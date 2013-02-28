# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import copy as cp
import numpy as np
from scipy import linalg
import warnings

import logging
logger = logging.getLogger('mne')

from .utils import get_subjects_dir
from .source_estimate import read_stc, mesh_edges, mesh_dist, morph_data, \
                             SourceEstimate
from .surface import read_surface
from . import verbose


def _aslabel(label):
    "For maintaining backwards-compatibility: convert dict to Label"
    if hasattr(label, 'hemi'):
        return label
    else:
        warnings.warn('Representing labels as dicts is deprecated and will be '
                      'removed in v0.6. Use Label objects.')
        return Label(**label)


class Label(dict):
    """An freesurfer/MNE label with vertices restricted to one hemisphere

    Labels can be combined with the ``+`` operator:
     - Duplicate vertices are removed.
     - If duplicate vertices have conflicting position values, an error is
       raised.
     - Values of duplicate vertices are summed.

    Parameters
    ----------
    vertices : array (length N)
        vertex indices (0 based).
    pos : array (N by 3)
        locations in meters.
    values : array (length N)
        values at the vertices.
    hemi : 'lh' | 'rh'
        Hemisphere to which the label applies.
    comment, name, fpath : str
        Kept as information but not used by the object itself.
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
    values : array, len = n_pos
        Values at the vertices.
    verbose : bool, str, int, or None
        See above.
    vertices : array, len = n_pos
        Vertex indices (0 based)

    Notes
    -----
    For backwards compatibility, the following attributes are stored as
    dictionary entries: ``'vertices', 'pos', 'values', 'hemi', 'comment'``
    """
    @verbose
    def __init__(self, vertices, pos, values, hemi, comment="", name=None,
                 filename=None, verbose=None):
        vertices = np.asarray(vertices)
        values = np.asarray(values)
        pos = np.asarray(pos)
        if not (len(vertices) == len(values) == len(pos)):
            err = ("vertices, values and pos need to have same length (number "
                   "of vertices)")
            raise ValueError(err)

        self.vertices = vertices
        self.pos = pos
        self.values = values
        self.hemi = hemi
        self.comment = comment

        # also set dict entries (for backwards compatibility)
        self._dep_warn = False
        self.update(vertices=vertices, pos=pos, values=values, hemi=hemi,
                    comment=comment)
        self._dep_warn = True

        # name
        if name is None and filename is not None:
            name = os.path.basename(filename[:-6])
        self.name = name
        self.filename = filename

    def __repr__(self):
        name = repr(self.name) if self.name is not None else "unnamed"
        n_vert = len(self)
        return "<Label  |  %s, %s : %i vertices>" % (name, self.hemi, n_vert)

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, key):
        warnings.warn('Dictionary-like usage of Label objects is deprecated '
                      'and will be removed in v0.6. Use attributes.')
        return super(Label, self).__getitem__(key)

    def _update_attr_from_dict(self):
        "backwards compatibility"
        if self._dep_warn:
            warnings.warn('Dictionary-like usage of Label objects is '
                          'deprecated and will be removed in v0.6.')
        for key in ['vertices', 'pos', 'values', 'hemi', 'comment']:
            setattr(self, key, self.get(key, None))

    def __setitem__(self, key, value):
        super(Label, self).__setitem__(key, value)
        self._update_attr_from_dict()

    def set(self, key, value):
        "Deprecated. Create a new Label."
        super(Label, self).set(key, value)
        self._update_attr_from_dict()

    def update(self, **D):
        "Deprecated. Create a new Label."
        super(Label, self).update(D)
        self._update_attr_from_dict()

    def __add__(self, other):
        if isinstance(other, BiHemiLabel):
            return other + self
        elif isinstance(other, Label):
            if self.hemi != other.hemi:
                name = '%s + %s' % (self.name, other.name)
                # catch warnings here to suppress deprecation warning
                # XXX this can be removed once dict useage is removed
                with warnings.catch_warnings(record=True) as w:
                    if self.hemi == 'lh':
                        lh, rh = cp.deepcopy(self), cp.deepcopy(other)
                    else:
                        lh, rh = cp.deepcopy(other), cp.deepcopy(self)
                return BiHemiLabel(lh, rh, name=name)
        else:
            raise TypeError("Need: Label or BiHemiLabel. Got: %r" % other)

        # check for overlap
        duplicates = np.intersect1d(self.vertices, other.vertices)
        n_dup = len(duplicates)
        if n_dup:
            self_dup = [np.where(self.vertices == d)[0][0] for d in duplicates]
            other_dup = [np.where(other.vertices == d)[0][0] for d in duplicates]
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

        label = Label(vertices, pos=pos, values=values, hemi=self.hemi,
                      comment="%s + %s" % (self.comment, other.comment),
                      name="%s + %s" % (name0, name1))
        return label

    def save(self, filename):
        "calls write_label to write the label to disk"
        write_label(filename, self)

    @verbose
    def smooth(self, subject, smooth=2, grade=None,
               subjects_dir=None, n_jobs=1, verbose=None):
        """Smooth the label. Useful for filling in labels made in a
        decimated source space for display.

        Parameters
        ----------
        subject : str
            The name of the subject used.
        smooth : int
            Number of iterations for the smoothing of the surface data.
            Cannot be None here since not all vertices are used. For a
            grade of 5 (e.g., fsaverage), a smoothing of 2 will fill a
            label.
        grade : int, or None
            Resolution of the icosahedral mesh to smooth to. To fill a
            label for display, use None.
        subjects_dir : string
            See morph_data.
        n_jobs: int
            See morph_data.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """

        if self.hemi == 'lh':
            vertices = [self.vertices, np.array([])]
        else:
            vertices = [np.array([]), self.vertices]
        data = self.values[:, np.newaxis]
        stc = SourceEstimate(data, vertices, tmin=1, tstep=1)
        stc = morph_data(subject, subject, stc, grade=grade, smooth=smooth,
                         subjects_dir=subjects_dir, n_jobs=n_jobs)
        inds = np.nonzero(stc.data)[0]
        self.values = stc.data[inds, :].ravel()
        self.pos = np.zeros((len(inds), 3))
        if self.hemi == 'lh':
            self.vertices = stc.vertno[0][inds]
        else:
            self.vertices = stc.vertno[1][inds]


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
        Labels for the left and right hemisphere, respectively

    name : None | str
        A name for the label. It is OK to change that attribute manually.

    """
    hemi = 'both'

    def __init__(self, lh, rh, name=None):
        self.lh = lh
        self.rh = rh
        self.name = name

    def __repr__(self):
        temp = "<BiHemiLabel  |  %s, lh : %i vertices,  rh : %i vertices>"
        name = repr(self.name) if self.name is not None else "unnamed"
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


def read_label(filename):
    """Read FreeSurfer Label file

    Parameters
    ----------
    filename : string
        Path to label file.

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
    nv = int(fid.readline())
    data = np.empty((5, nv))
    for i, line in enumerate(fid):
        data[:, i] = line.split()

    basename = os.path.basename(filename)
    if basename.endswith('lh.label') or basename.startswith('lh.'):
        hemi = 'lh'
    elif basename.endswith('rh.label') or basename.startswith('rh.'):
        hemi = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label')
    fid.close()

    label = Label(vertices=np.array(data[0], dtype=np.int32),
                  pos=1e-3 * data[1:4].T, values=data[4], hemi=hemi,
                  comment=comment, filename=filename)

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
    label = _aslabel(label)
    hemi = label.hemi
    path_head, name = os.path.split(filename)
    if name.endswith('.label'):
        name = name[:-6]
    if not (name.startswith(hemi) or name.endswith(hemi)):
        name += '-' + hemi
    filename = os.path.join(path_head, name) + '.label'

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
        Path to the label file

    stcfile : string
        Path to the stc file. The name of the stc file (must be on the
        same subject and hemisphere as the stc file)

    Returns
    -------
    values : 2d array
        The time courses
    times : 1d array
        The time points
    vertices : array
        The indices of the vertices corresponding to the time points
    """
    stc = read_stc(stcfile)
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
        A label
    src : list of dict
        The source space over which the label is defined

    Returns
    -------
    flip : array
        Sign flip vector (contains 1 or -1)
    """
    if len(src) != 2:
        raise ValueError('Only source spaces with 2 hemisphers are accepted')

    label = _aslabel(label)

    lh_vertno = src[0]['vertno']
    rh_vertno = src[1]['vertno']

    # get source orientations
    if label.hemi == 'lh':
        vertno_sel = np.intersect1d(lh_vertno, label.vertices)
        ori = src[0]['nn'][vertno_sel]
    elif label.hemi == 'rh':
        vertno_sel = np.intersect1d(rh_vertno, label.vertices)
        ori = src[1]['nn'][vertno_sel]
    else:
        raise Exception("Unknown hemisphere type")

    _, _, Vh = linalg.svd(ori, full_matrices=False)

    # Comparing to the direction of the first right singular vector
    flip = np.sign(np.dot(ori, Vh[:, 0]))
    return flip


def stc_to_label(stc, src, smooth=5):
    """Compute a label from the non-zero sources in an stc object.

    Parameters
    ----------
    stc : SourceEstimate
        The source estimates
    src : list of dict or string
        The source space over which the source estimates are defined.
        If it's a string it should the subject name (e.g. fsaverage).

    Returns
    -------
    labels : list of Labels
        The generated labels. One per hemisphere containing sources.
    """
    from scipy import sparse

    if not stc.is_surface():
        raise ValueError('SourceEstimate should be surface source estimates')

    if isinstance(src, basestring):
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            raise ValueError('SUBJECTS_DIR environment variable not set')
        surf_path_from = os.path.join(subjects_dir, src, 'surf')
        rr_lh, tris_lh = read_surface(os.path.join(surf_path_from,
                                      'lh.white'))
        rr_rh, tris_rh = read_surface(os.path.join(surf_path_from,
                                      'rh.white'))
        rr = [rr_lh, rr_rh]
        tris = [tris_lh, tris_rh]
    else:
        if len(src) != 2:
            raise ValueError('source space should contain the 2 hemispheres')
        tris = [src[0]['tris'], src[1]['tris']]
        rr = [1e3 * src[0]['rr'], 1e3 * src[1]['rr']]

    labels = []
    cnt = 0
    for hemi, this_vertno, this_tris, this_rr in \
                                    zip(['lh', 'rh'], stc.vertno, tris, rr):
        if len(this_vertno) == 0:
            continue
        e = mesh_edges(this_tris)
        e.data[e.data == 2] = 1
        n_vertices = e.shape[0]
        this_data = stc.data[cnt:cnt + len(this_vertno)]
        cnt += len(this_vertno)
        e = e + sparse.eye(n_vertices, n_vertices)
        idx_use = this_vertno[np.any(this_data, axis=1)]
        if len(idx_use) == 0:
            continue
        for k in range(smooth):
            e_use = e[:, idx_use]
            data1 = e_use * np.ones(len(idx_use))
            idx_use = np.where(data1)[0]

        label = Label(vertices=idx_use,
                      pos=this_rr[idx_use],
                      values=np.ones(len(idx_use)),
                      hemi=hemi,
                      comment='Label from stc')

        labels.append(label)

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


def grow_labels(subject, seeds, extents, hemis, subjects_dir=None):
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
        Name of the subject as in SUBJECTS_DIR
    seeds : array or int
        Seed vertex numbers
    extents : array or float
        Extents (radius in mm) of the labels
    hemis : array or int
        Hemispheres to use for the labels (0: left, 1: right)
    subjects_dir : string
        Path to SUBJECTS_DIR if not set in the environment

    Returns
    -------
    labels : list of Labels. The labels' ``comment`` attribute contains
        information on the seed vertex and extent; the ``values``  attribute
        contains distance from the seed in millimeters

    """
    if subjects_dir is None:
        if 'SUBJECTS_DIR' in os.environ:
            subjects_dir = os.environ['SUBJECTS_DIR']
        else:
            raise ValueError('SUBJECTS_DIR environment variable not set')

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
        surf_fname = os.path.join(subjects_dir, subject, 'surf',
                                  hemi + '.white')
        vert[hemi], tris[hemi] = read_surface(surf_fname)
        dist[hemi] = mesh_dist(tris[hemi], vert[hemi])

    # create the patches
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
    ctab : numpy array, shape=(n_verts, 5)
        RGBA + label id colortable array
    names : list of str
        List of region names as stored in the annot file

    """
    with open(fname, "rb") as fid:
        n_verts = np.fromfile(fid, '>i4', 1)[0]
        data = np.fromfile(fid, '>i4', n_verts * 2).reshape(n_verts, 2)
        annot = data[:, 1]
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
        ctab[:, 3] = 255

    return annot, ctab, names


def labels_from_parc(subject, parc='aparc', hemi='both', surf_name='white',
                     annot_fname=None, subjects_dir=None, verbose=None):
    """ Read labels from FreeSurfer parcellation

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
    if annot_fname is not None:
        # we use use the .annot file specified by the user
        hemis = [os.path.basename(annot_fname)[:2]]
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
            fname = os.path.join(subjects_dir, subject, 'label',
                                 '%s.%s.annot' % (hemi, parc))
            annot_fname.append(fname)

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
        fname_surf = os.path.join(subjects_dir, subject, 'surf',
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
            label = Label(vertices, pos, values, hemi, name=name)
            labels.append(label)

            # store the color
            label_rgba = tuple(label_rgba / 255.)
            label_colors.append(label_rgba)

        n_read = len(labels) - n_read
        logger.info('   read %d labels from %s' % (n_read, fname))

    # sort the labels and colors by label name
    names = [label.name for label in labels]
    labels, label_colors = zip(*((label, color) for (name, label, color)
                               in sorted(zip(names, labels, label_colors))))
    # convert tuples to lists
    labels = list(labels)
    label_colors = list(label_colors)

    logger.info('[done]')

    return labels, label_colors
