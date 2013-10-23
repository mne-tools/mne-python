# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from os import path as op
import os
import copy as cp
import numpy as np
import re
from scipy import linalg, sparse

from .utils import get_subjects_dir, _check_subject, logger, verbose
from .source_estimate import (_read_stc, mesh_edges, mesh_dist, morph_data,
                              SourceEstimate, spatial_src_connectivity)
from .surface import read_surface
from .parallel import parallel_func, check_n_jobs
from .stats.cluster_level import _find_clusters


class Label(object):
    """A FreeSurfer/MNE label with vertices restricted to one hemisphere

    Labels can be combined with the ``+`` operator:
     - Duplicate vertices are removed.
     - If duplicate vertices have conflicting position values, an error is
       raised.
     - Values of duplicate vertices are summed.

    Parameters
    ----------
    vertices : array (length N)
        vertex indices (0 based).
    pos : array (N by 3) | None
        locations in meters. If None, then zeros are used.
    values : array (length N) | None
        values at the vertices. If None, then ones are used.
    hemi : 'lh' | 'rh'
        Hemisphere to which the label applies.
    comment, name, fpath : str
        Kept as information but not used by the object itself.
    subject : str | None
        Name of the subject the label is from.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes
    ----------
    comment : str
        Comment from the first line of the label file.
    hemi : 'lh' | 'rh'
        Hemisphere.
    name : None | str
        A name for the label. It is OK to change that attribute manually.
    pos : array, shape = (n_pos, 3)
        Locations in meters.
    subject : str | None
        Subject name. It is best practice to set this to the proper
        value on initialization, but it can also be set manually.
    values : array, len = n_pos
        Values at the vertices.
    verbose : bool, str, int, or None
        See above.
    vertices : array, len = n_pos
        Vertex indices (0 based)
    """
    @verbose
    def __init__(self, vertices, pos=None, values=None, hemi=None, comment="",
                 name=None, filename=None, subject=None, verbose=None):
        if not isinstance(hemi, basestring):
            raise ValueError('hemi must be a string, not %s' % type(hemi))
        vertices = np.asarray(vertices)
        if np.any(np.diff(vertices.astype(int)) <= 0):
            raise ValueError('Vertices must be ordered in increasing '
                             'order.')

        if values is None:
            values = np.ones(len(vertices))
        if pos is None:
            pos = np.zeros((len(vertices), 3))
        values = np.asarray(values)
        pos = np.asarray(pos)
        if not (len(vertices) == len(values) == len(pos)):
            err = ("vertices, values and pos need to have same length (number "
                   "of vertices)")
            raise ValueError(err)

        # name
        if name is None and filename is not None:
            name = op.basename(filename[:-6])

        self.vertices = vertices
        self.pos = pos
        self.values = values
        self.hemi = hemi
        self.comment = comment
        self.verbose = verbose
        self.subject = _check_subject(None, subject, False)
        self.name = name
        self.filename = filename

    def __setstate__(self, state):
        self.vertices = state['vertices']
        self.pos = state['pos']
        self.values = state['values']
        self.hemi = state['hemi']
        self.comment = state['comment']
        self.verbose = state['verbose']
        self.subject = state.get('subject', None)
        self.name = state['name']
        self.filename = state['filename']

    def __getstate__(self):
        out = dict(vertices=self.vertices,
                   pos=self.pos,
                   values=self.values,
                   hemi=self.hemi,
                   comment=self.comment,
                   verbose=self.verbose,
                   subject=self.subject,
                   name=self.name,
                   filename=self.filename)
        return out

    def __repr__(self):
        name = 'unknown, ' if self.subject is None else self.subject + ', '
        name += repr(self.name) if self.name is not None else "unnamed"
        n_vert = len(self)
        return "<Label  |  %s, %s : %i vertices>" % (name, self.hemi, n_vert)

    def __len__(self):
        return len(self.vertices)

    def __add__(self, other):
        if isinstance(other, BiHemiLabel):
            return other + self
        elif isinstance(other, Label):
            if self.subject != other.subject:
                raise ValueError('Label subject parameters must match, got '
                                 '"%s" and "%s". Consider setting the '
                                 'subject parameter on initialization, or '
                                 'setting label.subject manually before '
                                 'combining labels.' % (self.subject,
                                                        other.subject))
            if self.hemi != other.hemi:
                name = '%s + %s' % (self.name, other.name)
                if self.hemi == 'lh':
                    lh, rh = self.copy(), other.copy()
                else:
                    lh, rh = other.copy(), self.copy()
                return BiHemiLabel(lh, rh, name=name)
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        # check for overlap
        duplicates = np.intersect1d(self.vertices, other.vertices)
        n_dup = len(duplicates)
        if n_dup:
            self_dup = [np.where(self.vertices == d)[0][0]
                        for d in duplicates]
            other_dup = [np.where(other.vertices == d)[0][0]
                         for d in duplicates]
            if not np.all(self.pos[self_dup] == other.pos[other_dup]):
                err = ("Labels %r and %r: vertices overlap but differ in "
                       "position values" % (self.name, other.name))
                raise ValueError(err)

            isnew = np.array([v not in duplicates for v in other.vertices])

            vertices = np.hstack((self.vertices, other.vertices[isnew]))
            pos = np.vstack((self.pos, other.pos[isnew]))

            # find position of other's vertices in new array
            tgt_idx = [np.where(vertices == v)[0][0] for v in other.vertices]
            n_self = len(self.values)
            n_other = len(other.values)
            new_len = n_self + n_other - n_dup
            values = np.zeros(new_len, dtype=self.values.dtype)
            values[:n_self] += self.values
            values[tgt_idx] += other.values
        else:
            vertices = np.hstack((self.vertices, other.vertices))
            pos = np.vstack((self.pos, other.pos))
            values = np.hstack((self.values, other.values))

        name0 = self.name if self.name else 'unnamed'
        name1 = other.name if other.name else 'unnamed'

        indcs = np.argsort(vertices)
        vertices, pos, values = vertices[indcs], pos[indcs, :], values[indcs]

        label = Label(vertices, pos=pos, values=values, hemi=self.hemi,
                      comment="%s + %s" % (self.comment, other.comment),
                      name="%s + %s" % (name0, name1))
        return label

    def save(self, filename):
        "calls write_label to write the label to disk"
        write_label(filename, self)

    def copy(self):
        """Copy the label instance.

        Returns
        -------
        label : instance of Label
            The copied label.
        """
        return cp.deepcopy(self)

    @verbose
    def smooth(self, subject=None, smooth=2, grade=None,
               subjects_dir=None, n_jobs=1, copy=True, verbose=None):
        """Smooth the label

        Useful for filling in labels made in a
        decimated source space for display.

        Parameters
        ----------
        subject : str | None
            The name of the subject used. If None, the value will be
            taken from self.subject.
        smooth : int
            Number of iterations for the smoothing of the surface data.
            Cannot be None here since not all vertices are used. For a
            grade of 5 (e.g., fsaverage), a smoothing of 2 will fill a
            label.
        grade : int, list (of two arrays), array, or None
            Resolution of the icosahedral mesh (typically 5). If None, all
            vertices will be used (potentially filling the surface). If a list,
            values will be morphed to the set of vertices specified in grade[0]
            and grade[1], assuming that these are vertices for the left and
            right hemispheres. Note that specifying the vertices (e.g.,
            grade=[np.arange(10242), np.arange(10242)] for fsaverage on a
            standard grade 5 source space) can be substantially faster than
            computing vertex locations. If one array is used, it is assumed
            that all vertices belong to the hemisphere of the label. To create
            a label filling the surface, use None.
        subjects_dir : string, or None
            Path to SUBJECTS_DIR if it is not set in the environment.
        n_jobs : int
            Number of jobs to run in parallel
        copy : bool
            If False, smoothing is done in-place.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Returns
        -------
        label : instance of Label
            The smoothed label.

        Notes
        -----
        This function will set label.pos to be all zeros. If the positions
        on the new surface are required, consider using mne.read_surface
        with label.vertices.
        """
        subject = _check_subject(self.subject, subject)
        return self.morph(subject, subject, smooth, grade, subjects_dir,
                          n_jobs, copy)

    @verbose
    def morph(self, subject_from=None, subject_to=None, smooth=5, grade=None,
              subjects_dir=None, n_jobs=1, copy=True, verbose=None):
        """Morph the label

        Useful for transforming a label from one subject to another.

        Parameters
        ----------
        subject_from : str | None
            The name of the subject of the current label. If None, the
            initial subject will be taken from self.subject.
        subject_to : str
            The name of the subject to morph the label to. This will
            be put in label.subject of the output label file.
        smooth : int
            Number of iterations for the smoothing of the surface data.
            Cannot be None here since not all vertices are used.
        grade : int, list (of two arrays), array, or None
            Resolution of the icosahedral mesh (typically 5). If None, all
            vertices will be used (potentially filling the surface). If a list,
            values will be morphed to the set of vertices specified in grade[0]
            and grade[1], assuming that these are vertices for the left and
            right hemispheres. Note that specifying the vertices (e.g.,
            grade=[np.arange(10242), np.arange(10242)] for fsaverage on a
            standard grade 5 source space) can be substantially faster than
            computing vertex locations. If one array is used, it is assumed
            that all vertices belong to the hemisphere of the label. To create
            a label filling the surface, use None.
        subjects_dir : string, or None
            Path to SUBJECTS_DIR if it is not set in the environment.
        n_jobs : int
            Number of jobs to run in parallel.
        copy : bool
            If False, the morphing is done in-place.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        label : instance of Label
            The morphed label.

        Notes
        -----
        This function will set label.pos to be all zeros. If the positions
        on the new surface are required, consider using mne.read_surface
        with label.vertices.
        """
        subject_from = _check_subject(self.subject, subject_from)
        if not isinstance(subject_to, basestring):
            raise TypeError('"subject_to" must be entered as a string')
        if not isinstance(smooth, int):
            raise ValueError('smooth must be an integer')
        if np.all(self.values == 0):
            raise ValueError('Morphing label with all zero values will result '
                             'in the label having no vertices. Consider using '
                             'something like label.values.fill(1.0).')
        if(isinstance(grade, np.ndarray)):
            if self.hemi == 'lh':
                grade = [grade, np.array([])]
            else:
                grade = [np.array([]), grade]
        if self.hemi == 'lh':
            vertices = [self.vertices, np.array([])]
        else:
            vertices = [np.array([]), self.vertices]
        data = self.values[:, np.newaxis]
        stc = SourceEstimate(data, vertices, tmin=1, tstep=1,
                             subject=subject_from)
        stc = morph_data(subject_from, subject_to, stc, grade=grade,
                         smooth=smooth, subjects_dir=subjects_dir,
                         n_jobs=n_jobs)
        inds = np.nonzero(stc.data)[0]
        if copy is True:
            label = self.copy()
        else:
            label = self
        label.values = stc.data[inds, :].ravel()
        label.pos = np.zeros((len(inds), 3))
        if label.hemi == 'lh':
            label.vertices = stc.vertno[0][inds]
        else:
            label.vertices = stc.vertno[1][inds]
        label.subject = subject_to
        return label


class BiHemiLabel(object):
    """A freesurfer/MNE label with vertices in both hemispheres

    Parameters
    ----------
    lh, rh : Label
        Label objects representing the left and the right hemisphere,
        respectively
    name : None | str
        name for the label

    Attributes
    ----------
    lh, rh : Label
        Labels for the left and right hemisphere, respectively.
    name : None | str
        A name for the label. It is OK to change that attribute manually.
    subject : str | None
        Subject the label is from.
    """
    hemi = 'both'

    def __init__(self, lh, rh, name=None):
        if lh.subject != rh.subject:
            raise ValueError('lh.subject (%s) and rh.subject (%s) must '
                             'agree' % (lh.subject, rh.subject))
        self.lh = lh
        self.rh = rh
        self.name = name
        self.subject = lh.subject

    def __repr__(self):
        temp = "<BiHemiLabel  |  %s, lh : %i vertices,  rh : %i vertices>"
        name = 'unknown, ' if self.subject is None else self.subject + ', '
        name += repr(self.name) if self.name is not None else "unnamed"
        return temp % (name, len(self.lh), len(self.rh))

    def __len__(self):
        return len(self.lh) + len(self.rh)

    def __add__(self, other):
        if isinstance(other, Label):
            if other.hemi == 'lh':
                lh = self.lh + other
                rh = self.rh
            else:
                lh = self.lh
                rh = self.rh + other
        elif isinstance(other, BiHemiLabel):
            lh = self.lh + other.lh
            rh = self.rh + other.rh
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        name = '%s + %s' % (self.name, other.name)
        return BiHemiLabel(lh, rh, name=name)


def read_label(filename, subject=None):
    """Read FreeSurfer Label file

    Parameters
    ----------
    filename : string
        Path to label file.
    subject : str | None
        Name of the subject the data are defined for.
        It is good practice to set this attribute to avoid combining
        incompatible labels and SourceEstimates (e.g., ones from other
        subjects). Note that due to file specification limitations, the
        subject name isn't saved to or loaded from files written to disk.

    Returns
    -------
    label : Label
        Instance of Label object with attributes:
            comment        comment from the first line of the label file
            vertices       vertex indices (0 based, column 1)
            pos            locations in meters (columns 2 - 4 divided by 1000)
            values         values at the vertices (column 5)
    """
    fid = open(filename, 'r')
    comment = fid.readline().replace('\n', '')[1:]
    if subject is not None and not isinstance(subject, basestring):
        raise TypeError('subject must be a string')

    nv = int(fid.readline())
    data = np.empty((5, nv))
    for i, line in enumerate(fid):
        data[:, i] = line.split()

    basename = op.basename(filename)
    if basename.endswith('lh.label') or basename.startswith('lh.'):
        hemi = 'lh'
    elif basename.endswith('rh.label') or basename.startswith('rh.'):
        hemi = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label')
    fid.close()

    # let's make sure everything is ordered correctly
    vertices = np.array(data[0], dtype=np.int32)
    pos = 1e-3 * data[1:4].T
    values = data[4]
    order = np.argsort(vertices)
    vertices = vertices[order]
    pos = pos[order]
    values = values[order]

    label = Label(vertices=vertices, pos=pos, values=values, hemi=hemi,
                  comment=comment, filename=filename, subject=subject)

    return label


@verbose
def write_label(filename, label, verbose=None):
    """Write a FreeSurfer label

    Parameters
    ----------
    filename : string
        Path to label file to produce.
    label : Label
        The label object to save.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    hemi = label.hemi
    path_head, name = op.split(filename)
    if name.endswith('.label'):
        name = name[:-6]
    if not (name.startswith(hemi) or name.endswith(hemi)):
        name += '-' + hemi
    filename = op.join(path_head, name) + '.label'

    logger.info('Saving label to : %s' % filename)

    fid = open(filename, 'wb')
    n_vertices = len(label.vertices)
    data = np.zeros((n_vertices, 5), dtype=np.float)
    data[:, 0] = label.vertices
    data[:, 1:4] = 1e3 * label.pos
    data[:, 4] = label.values
    fid.write("#%s\n" % label.comment)
    fid.write("%d\n" % n_vertices)
    for d in data:
        fid.write("%d %f %f %f %f\n" % tuple(d))

    return label


def label_time_courses(labelfile, stcfile):
    """Extract the time courses corresponding to a label file from an stc file

    Parameters
    ----------
    labelfile : string
        Path to the label file.
    stcfile : string
        Path to the stc file. The name of the stc file (must be on the
        same subject and hemisphere as the stc file).

    Returns
    -------
    values : 2d array
        The time courses.
    times : 1d array
        The time points.
    vertices : array
        The indices of the vertices corresponding to the time points.
    """
    stc = _read_stc(stcfile)
    lab = read_label(labelfile)

    vertices = np.intersect1d(stc['vertices'], lab.vertices)
    idx = [k for k in range(len(stc['vertices']))
           if stc['vertices'][k] in vertices]

    if len(vertices) == 0:
        raise ValueError('No vertices match the label in the stc file')

    values = stc['data'][idx]
    times = stc['tmin'] + stc['tstep'] * np.arange(stc['data'].shape[1])

    return values, times, vertices


def label_sign_flip(label, src):
    """Compute sign for label averaging

    Parameters
    ----------
    label : Label
        A label.
    src : list of dict
        The source space over which the label is defined.

    Returns
    -------
    flip : array
        Sign flip vector (contains 1 or -1)
    """
    if len(src) != 2:
        raise ValueError('Only source spaces with 2 hemisphers are accepted')

    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    # get source orientations
    if label.hemi == 'lh':
        vertno_sel = np.intersect1d(lh_vertno, label.vertices)
        if len(vertno_sel) == 0:
            return np.array([])
        ori = src[0]['nn'][vertno_sel]
    elif label.hemi == 'rh':
        vertno_sel = np.intersect1d(rh_vertno, label.vertices)
        if len(vertno_sel) == 0:
            return np.array([])
        ori = src[1]['nn'][vertno_sel]
    else:
        raise Exception("Unknown hemisphere type")

    _, _, Vh = linalg.svd(ori, full_matrices=False)

    # Comparing to the direction of the first right singular vector
    flip = np.sign(np.dot(ori, Vh[:, 0] if len(vertno_sel) > 3 else Vh[0]))
    return flip


def stc_to_label(stc, src=None, smooth=5, connected=False, subjects_dir=None):
    """Compute a label from the non-zero sources in an stc object.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates.
    src : list of dict | string | None
        The source space over which the source estimates are defined.
        If it's a string it should the subject name (e.g. fsaverage).
        Can be None if stc.subject is not None.
    smooth : int
        Number of smoothing steps to use.
    connected : bool
        If True a list of connected labels will be returned in each
        hemisphere. The labels are ordered in decreasing order depending
        of the maximum value in the stc.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    labels : list of Labels | list of list of Labels
        The generated labels. If connected is False, it returns
        a list of Labels (One per hemisphere). If no Label is available
        in an hemisphere, None is returned. If connected is True,
        it returns for each hemisphere a list of connected labels
        ordered in decreasing order depending of the maximum value in the stc.
        If no Label is available in an hemisphere, an empty list is returned.
    """
    src = stc.subject if src is None else src
    if src is None:
        raise ValueError('src cannot be None if stc.subject is None')
    if isinstance(src, basestring):
        subject = src
    else:
        subject = stc.subject

    if not isinstance(stc, SourceEstimate):
        raise ValueError('SourceEstimate should be surface source estimates')

    if isinstance(src, basestring):
        subjects_dir = get_subjects_dir(subjects_dir)
        surf_path_from = op.join(subjects_dir, src, 'surf')
        rr_lh, tris_lh = read_surface(op.join(surf_path_from,
                                      'lh.white'))
        rr_rh, tris_rh = read_surface(op.join(surf_path_from,
                                      'rh.white'))
        rr = [rr_lh, rr_rh]
        tris = [tris_lh, tris_rh]
        if connected:
            raise ValueError('The option to return only connected labels'
                             ' is only available if a source space is passed'
                             ' as parameter.')
    else:
        if len(src) != 2:
            raise ValueError('source space should contain the 2 hemispheres')
        rr = [1e3 * src[0]['rr'], 1e3 * src[1]['rr']]
        tris = [src[0]['tris'], src[1]['tris']]
        src_conn = spatial_src_connectivity(src).tocsr()

    labels = []
    cnt = 0
    cnt_full = 0
    for hemi_idx, (hemi, this_vertno, this_tris, this_rr) in enumerate(
                                    zip(['lh', 'rh'], stc.vertno, tris, rr)):

        this_data = stc.data[cnt:cnt + len(this_vertno)]

        e = mesh_edges(this_tris)
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        e = e + sparse.eye(n_vertices, n_vertices)

        if connected:
            if not isinstance(src, basestring):  # XXX : ugly
                inuse = np.where(src[hemi_idx]['inuse'])[0]
                tmp = np.zeros((len(inuse), this_data.shape[1]))
                this_vertno_idx = np.searchsorted(inuse, this_vertno)
                tmp[this_vertno_idx] = this_data
                this_data = tmp
            offset = cnt_full + len(this_data)
            this_src_conn = src_conn[cnt_full:offset, cnt_full:offset].tocoo()
            this_data_abs_max = np.abs(this_data).max(axis=1)
            clusters, _ = _find_clusters(this_data_abs_max, 0.,
                                         connectivity=this_src_conn)
            cnt_full += len(this_data)
            # Then order clusters in descending order based on maximum value
            clusters_max = np.argsort([np.max(this_data_abs_max[c])
                                       for c in clusters])[::-1]
            clusters = [clusters[k] for k in clusters_max]
            clusters = [inuse[c] for c in clusters]
        else:
            clusters = [this_vertno[np.any(this_data, axis=1)]]

        cnt += len(this_vertno)

        clusters = [c for c in clusters if len(c) > 0]

        if len(clusters) == 0:
            if not connected:
                this_labels = None
            else:
                this_labels = []
        else:
            this_labels = []
            for c in clusters:
                idx_use = c
                for k in range(smooth):
                    e_use = e[:, idx_use]
                    data1 = e_use * np.ones(len(idx_use))
                    idx_use = np.where(data1)[0]

                label = Label(vertices=idx_use,
                              pos=this_rr[idx_use],
                              values=np.ones(len(idx_use)),
                              hemi=hemi,
                              comment='Label from stc',
                              subject=subject)
                this_labels.append(label)

            if not connected:
                this_labels = this_labels[0]

        labels.append(this_labels)

    return labels


def _verts_within_dist(graph, source, max_dist):
    """Find all vertices wihin a maximum geodesic distance from source

    Parameters
    ----------
    graph : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
    source : int
        Source vertex
    max_dist: float
        Maximum geodesic distance

    Returns
    -------
    verts : array
        Vertices within max_dist
    dist : array
        Distances from source vertex
    """
    dist_map = {}
    dist_map[source] = 0
    verts_added_last = [source]
    # add neighbors until no more neighbors within max_dist can be found
    while len(verts_added_last) > 0:
        verts_added = []
        for i in verts_added_last:
            v_dist = dist_map[i]
            row = graph[i, :]
            neighbor_vert = row.indices
            neighbor_dist = row.data
            for j, d in zip(neighbor_vert, neighbor_dist):
                n_dist = v_dist + d
                if j in dist_map:
                    if n_dist < dist_map[j]:
                        dist_map[j] = n_dist
                else:
                    if n_dist <= max_dist:
                        dist_map[j] = n_dist
                        # we found a new vertex within max_dist
                        verts_added.append(j)
        verts_added_last = verts_added

    verts = np.sort(np.array(dist_map.keys(), dtype=np.int))
    dist = np.array([dist_map[v] for v in verts])

    return verts, dist


def _grow_labels(seeds, extents, hemis, dist, vert):
    """Helper for parallelization of grow_labels
    """
    labels = []
    for seed, extent, hemi in zip(seeds, extents, hemis):
        label_verts, label_dist = _verts_within_dist(dist[hemi], seed, extent)

        # create a label
        comment = 'Circular label: seed=%d, extent=%0.1fmm' % (seed, extent)
        label = Label(vertices=label_verts,
                      pos=vert[hemi][label_verts],
                      values=label_dist,
                      hemi=hemi,
                      comment=comment)
        labels.append(label)
    return labels


def grow_labels(subject, seeds, extents, hemis, subjects_dir=None,
                n_jobs=1):
    """Generate circular labels in source space with region growing

    This function generates a number of labels in source space by growing
    regions starting from the vertices defined in "seeds". For each seed, a
    label is generated containing all vertices within a maximum geodesic
    distance on the white matter surface from the seed.

    Note: "extents" and "hemis" can either be arrays with the same length as
          seeds, which allows using a different extent and hemisphere for each
          label, or integers, in which case the same extent and hemisphere is
          used for each label.

    Parameters
    ----------
    subject : string
        Name of the subject as in SUBJECTS_DIR.
    seeds : array or int
        Seed vertex numbers.
    extents : array or float
        Extents (radius in mm) of the labels.
    hemis : array or int
        Hemispheres to use for the labels (0: left, 1: right).
    subjects_dir : string
        Path to SUBJECTS_DIR if not set in the environment.
    n_jobs : int
        Number of jobs to run in parallel. Likely only useful if tens
        or hundreds of labels are being expanded simultaneously.

    Returns
    -------
    labels : list of Labels. The labels' ``comment`` attribute contains
        information on the seed vertex and extent; the ``values``  attribute
        contains distance from the seed in millimeters

    """
    subjects_dir = get_subjects_dir(subjects_dir)
    n_jobs = check_n_jobs(n_jobs)

    # make sure the inputs are arrays
    seeds = np.atleast_1d(seeds)
    extents = np.atleast_1d(extents)
    hemis = np.atleast_1d(hemis)

    n_seeds = len(seeds)

    if len(extents) != 1 and len(extents) != n_seeds:
        raise ValueError('The extents parameter has to be of length 1 or '
                         'len(seeds)')

    if len(hemis) != 1 and len(hemis) != n_seeds:
        raise ValueError('The hemis parameter has to be of length 1 or '
                         'len(seeds)')

    # make the arrays the same length as seeds
    if len(extents) == 1:
        extents = np.tile(extents, n_seeds)

    if len(hemis) == 1:
        hemis = np.tile(hemis, n_seeds)

    hemis = ['lh' if h == 0 else 'rh' for h in hemis]

    # load the surfaces and create the distance graphs
    tris, vert, dist = {}, {}, {}
    for hemi in set(hemis):
        surf_fname = op.join(subjects_dir, subject, 'surf', hemi + '.white')
        vert[hemi], tris[hemi] = read_surface(surf_fname)
        dist[hemi] = mesh_dist(tris[hemi], vert[hemi])

    # create the patches
    parallel, my_grow_labels, _ = parallel_func(_grow_labels, n_jobs)
    seeds = np.array_split(seeds, n_jobs)
    extents = np.array_split(extents, n_jobs)
    hemis = np.array_split(hemis, n_jobs)
    labels = sum(parallel(my_grow_labels(s, e, h, dist, vert)
                          for s, e, h in zip(seeds, extents, hemis)), [])
    return labels


def _read_annot(fname):
    """Read a Freesurfer annotation from a .annot file.

    Note : Copied from PySurfer

    Parameters
    ----------
    fname : str
        Path to annotation file

    Returns
    -------
    annot : numpy array, shape=(n_verts)
        Annotation id at each vertex
    ctab : numpy array, shape=(n_entries, 5)
        RGBA + label id colortable array
    names : list of str
        List of region names as stored in the annot file

    """
    if not op.isfile(fname):
        dir_name = op.split(fname)[0]
        if not op.isdir(dir_name):
            raise IOError('Directory for annotation does not exist: %s',
                          fname)
        cands = os.listdir(dir_name)
        cands = [c for c in cands if '.annot' in c]
        if len(cands) == 0:
            raise IOError('No such file %s, no candidate parcellations '
                          'found in directory' % fname)
        else:
            raise IOError('No such file %s, candidate parcellations in '
                          'that directory: %s' % (fname, ', '.join(cands)))
    with open(fname, "rb") as fid:
        n_verts = np.fromfile(fid, '>i4', 1)[0]
        data = np.fromfile(fid, '>i4', n_verts * 2).reshape(n_verts, 2)
        annot = data[data[:, 0], 1]
        ctab_exists = np.fromfile(fid, '>i4', 1)[0]
        if not ctab_exists:
            raise Exception('Color table not found in annotation file')
        n_entries = np.fromfile(fid, '>i4', 1)[0]
        if n_entries > 0:
            length = np.fromfile(fid, '>i4', 1)[0]
            orig_tab = np.fromfile(fid, '>c', length)
            orig_tab = orig_tab[:-1]

            names = list()
            ctab = np.zeros((n_entries, 5), np.int)
            for i in xrange(n_entries):
                name_length = np.fromfile(fid, '>i4', 1)[0]
                name = np.fromfile(fid, "|S%d" % name_length, 1)[0]
                names.append(name)
                ctab[i, :4] = np.fromfile(fid, '>i4', 4)
                ctab[i, 4] = (ctab[i, 0] + ctab[i, 1] * (2 ** 8) +
                              ctab[i, 2] * (2 ** 16) +
                              ctab[i, 3] * (2 ** 24))
        else:
            ctab_version = -n_entries
            if ctab_version != 2:
                raise Exception('Color table version not supported')
            n_entries = np.fromfile(fid, '>i4', 1)[0]
            ctab = np.zeros((n_entries, 5), np.int)
            length = np.fromfile(fid, '>i4', 1)[0]
            _ = np.fromfile(fid, "|S%d" % length, 1)[0]  # Orig table path
            entries_to_read = np.fromfile(fid, '>i4', 1)[0]
            names = list()
            for i in xrange(entries_to_read):
                _ = np.fromfile(fid, '>i4', 1)[0]  # Structure
                name_length = np.fromfile(fid, '>i4', 1)[0]
                name = np.fromfile(fid, "|S%d" % name_length, 1)[0]
                names.append(name)
                ctab[i, :4] = np.fromfile(fid, '>i4', 4)
                ctab[i, 4] = (ctab[i, 0] + ctab[i, 1] * (2 ** 8) +
                              ctab[i, 2] * (2 ** 16))

        # convert to more common alpha value
        ctab[:, 3] = 255 - ctab[:, 3]

    return annot, ctab, names


def _get_annot_fname(annot_fname, subject, hemi, parc, subjects_dir):
    """Helper function to get the .annot filenames and hemispheres"""
    if annot_fname is not None:
        # we use use the .annot file specified by the user
        hemis = [op.basename(annot_fname)[:2]]
        if hemis[0] not in ['lh', 'rh']:
            raise ValueError('Could not determine hemisphere from filename, '
                             'filename has to start with "lh" or "rh".')
        annot_fname = [annot_fname]
    else:
        # construct .annot file names for requested subject, parc, hemi
        if hemi not in ['lh', 'rh', 'both']:
            raise ValueError('hemi has to be "lh", "rh", or "both"')
        if hemi == 'both':
            hemis = ['lh', 'rh']
        else:
            hemis = [hemi]

        annot_fname = list()
        for hemi in hemis:
            fname = op.join(subjects_dir, subject, 'label',
                            '%s.%s.annot' % (hemi, parc))
            annot_fname.append(fname)

    return annot_fname, hemis


@verbose
def labels_from_parc(subject, parc='aparc', hemi='both', surf_name='white',
                     annot_fname=None, regexp=None, subjects_dir=None,
                     verbose=None):
    """Read labels from FreeSurfer parcellation

    Note: Only cortical labels will be returned.

    Parameters
    ----------
    subject : str
        The subject for which to read the parcellation for.
    parc : str
        The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
    hemi : str
        The hemisphere to read the parcellation for, can be 'lh', 'rh',
        or 'both'.
    surf_name : str
        Surface used to obtain vertex locations, e.g., 'white', 'pial'
    annot_fname : str or None
        Filename of the .annot file. If not None, only this file is read
        and 'parc' and 'hemi' are ignored.
    regexp : str
        Regular expression or substring to select particular labels from the
        parcellation. E.g. 'superior' will return all labels in which this
        substring is contained.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    Returns
    -------
    labels : list of Label
        The labels, sorted by label name (ascending).
    colors : list of tuples
        RGBA color for obtained from the parc color table for each label.
    """
    logger.info('Reading labels from parcellation..')

    subjects_dir = get_subjects_dir(subjects_dir)

    # get the .annot filenames and hemispheres
    annot_fname, hemis = _get_annot_fname(annot_fname, subject, hemi, parc,
                                          subjects_dir)

    # now we are ready to create the labels
    n_read = 0
    labels = list()
    label_colors = list()
    for fname, hemi in zip(annot_fname, hemis):
        # read annotation
        annot, ctab, label_names = _read_annot(fname)
        label_rgbas = ctab[:, :4]
        label_ids = ctab[:, -1]

        # load the vertex positions from surface
        fname_surf = op.join(subjects_dir, subject, 'surf',
                             '%s.%s' % (hemi, surf_name))
        vert_pos, _ = read_surface(fname_surf)
        vert_pos /= 1e3  # the positions in labels are in meters
        for label_id, label_name, label_rgba in\
                zip(label_ids, label_names, label_rgbas):
            vertices = np.where(annot == label_id)[0]
            if len(vertices) == 0:
                # label is not part of cortical surface
                continue
            pos = vert_pos[vertices, :]
            values = np.zeros(len(vertices))
            name = label_name + '-' + hemi
            label = Label(vertices, pos, values, hemi, name=name,
                          subject=subject)
            labels.append(label)

            # store the color
            label_rgba = tuple(label_rgba / 255.)
            label_colors.append(label_rgba)

        n_read = len(labels) - n_read
        logger.info('   read %d labels from %s' % (n_read, fname))

    if regexp is not None:
        # allow for convenient substring match
        r_ = (re.compile('.*%s.*' % regexp if regexp.replace('_', '').isalnum()
              else regexp))

    # sort the labels and colors by label name
    names = [label.name for label in labels]
    labels_ = zip(*((label, color) for (name, label, color) in sorted(
                    zip(names, labels, label_colors))
                        if (r_.match(name) if regexp else True)))
    if labels_:
        labels, label_colors = labels_
    else:
        raise RuntimeError('The regular expression supplied did not match.')
    # convert tuples to lists
    labels = list(labels)
    label_colors = list(label_colors)
    logger.info('[done]')

    return labels, label_colors


def _write_annot(fname, annot, ctab, names):
    """Write a Freesurfer annotation to a .annot file.

    Parameters
    ----------
    fname : str
        Path to annotation file
    annot : numpy array, shape=(n_verts)
        Annotation id at each vertex. Note: IDs must be computed from
        RGBA colors, otherwise the mapping will be invalid.
    ctab : numpy array, shape=(n_entries, 4)
        RGBA colortable array.
    names : list of str
        List of region names to be stored in the annot file
    """

    with open(fname, 'wb') as fid:
        n_verts = len(annot)
        np.array(n_verts, dtype='>i4').tofile(fid)

        data = np.zeros((n_verts, 2), dtype='>i4')
        data[:, 0] = np.arange(n_verts)
        data[:, 1] = annot
        data.ravel().tofile(fid)

        # indicate that color table exists
        np.array(1, dtype='>i4').tofile(fid)

        # color table version 2
        np.array(-2, dtype='>i4').tofile(fid)

        # write color table
        n_entries = len(ctab)
        np.array(n_entries, dtype='>i4').tofile(fid)

        # write dummy color table name
        table_name = 'MNE-Python Colortable'
        np.array(len(table_name), dtype='>i4').tofile(fid)
        np.fromstring(table_name, dtype=np.uint8).tofile(fid)

        # number of entries to write
        np.array(n_entries, dtype='>i4').tofile(fid)

        # write entries
        for ii, (name, color) in enumerate(zip(names, ctab)):
            np.array(ii, dtype='>i4').tofile(fid)
            np.array(len(name), dtype='>i4').tofile(fid)
            np.fromstring(name, dtype=np.uint8).tofile(fid)
            np.array(color[:4], dtype='>i4').tofile(fid)


@verbose
def parc_from_labels(labels, colors, subject=None, parc=None,
                     annot_fname=None, overwrite=False, subjects_dir=None,
                     verbose=None):
    """Create a FreeSurfer parcellation from labels

    Parameters
    ----------
    labels : list with instances of mne.Label
        The labels to create a parcellation from.
    colors : list of tuples | None
        RGBA color to write into the colortable for each label. If None,
        the colors are created based on the alphabetical order of the label
        names. Note: Per hemisphere, each label must have a unique color,
        otherwise the stored parcellation will be invalid.
    subject : str | None
        The subject for which to write the parcellation for.
    parc : str | None
        The parcellation name to use.
    annot_fname : str | None
        Filename of the .annot file. If not None, only this file is written
        and 'parc' and 'subject' are ignored.
    overwrite : bool
        Overwrite files if they already exist.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    logger.info('Writing labels to parcellation..')

    # do some input checking
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[1] != 4:
            raise ValueError('Each color must have 4 values')
        if len(colors) != len(labels):
            raise ValueError('colors must have the same length as labels')
        if np.any(colors < 0) or np.any(colors > 1):
            raise ValueError('color values must be between 0 and 1')

    subjects_dir = get_subjects_dir(subjects_dir)

    # get the .annot filenames and hemispheres
    annot_fname, hemis = _get_annot_fname(annot_fname, subject, 'both', parc,
                                          subjects_dir)

    if not overwrite:
        for fname in annot_fname:
            if op.exists(fname):
                raise ValueError('File %s exists. Use "overwrite=True" to '
                                 'overwrite it' % fname)

    names = ['%s-%s' % (label.name, label.hemi) for label in labels]

    for hemi, fname in zip(hemis, annot_fname):
        hemi_labels = [label for label in labels if label.hemi == hemi]
        n_hemi_labels = len(hemi_labels)
        if n_hemi_labels == 0:
            # no labels for this hemisphere
            continue
        hemi_labels.sort(key=lambda label: label.name)
        if colors is not None:
            hemi_colors = [colors[names.index('%s-%s' % (label.name, hemi))]
                           for label in hemi_labels]
        else:
            import matplotlib.pyplot as plt
            hemi_colors = plt.cm.spectral(np.linspace(0, 1, n_hemi_labels))

        # Creat annot and color table array to write
        max_vert = 0
        for label in hemi_labels:
            max_vert = max(max_vert, np.max(label.vertices))
        n_vertices = max_vert + 1
        annot = np.zeros(n_vertices, dtype=np.int)
        ctab = np.zeros((n_hemi_labels, 4), dtype=np.int32)
        for ii, (label, color) in enumerate(zip(hemi_labels, hemi_colors)):
            ctab[ii] = np.round(255 * np.asarray(color))
            if np.all(ctab[ii, :3] == 0):
                # we cannot have an all-zero color, otherw. e.g. tksurfer
                # refuses to read the parcellation
                if colors is not None:
                    logger.warning('    Colormap contains color with, "r=0, '
                                   'g=0, b=0" value. Some FreeSurfer tools '
                                   'may fail to read the parcellation')
                else:
                    ctab[ii, :3] = 1

            # create the annotation id from the color
            annot_id = (ctab[ii, 0] + ctab[ii, 1] * 2 ** 8
                        + ctab[ii, 2] * 2 ** 16)

            annot[label.vertices] = annot_id

        # convert to FreeSurfer alpha values
        ctab[:, 3] = 255 - ctab[:, 3]

        hemi_names = [label.name for label in hemi_labels]

        # remove hemi ending in names
        hemi_names = [name[:-3] if name.endswith(hemi) else name
                      for name in hemi_names]
        # write it
        logger.info('   writing %d labels to %s' % (n_hemi_labels, fname))
        _write_annot(fname, annot, ctab, hemi_names)

    logger.info('[done]')
