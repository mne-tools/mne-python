# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import numpy as np
from scipy import linalg

from .source_estimate import read_stc, mesh_edges, mesh_dist
from .surface import read_surface


def read_label(filename):
    """Read FreeSurfer Label file

    Parameters
    ----------
    filename : string
        Path to label file.

    Returns
    -------
    label : dict
        Label dictionaries with keys:
            comment        comment from the first line of the label file
            vertices       vertex indices (0 based, column 1)
            pos            locations in meters (columns 2 - 4 divided by 1000)
            values         values at the vertices (column 5)

    """
    fid = open(filename, 'r')
    comment = fid.readline().replace('\n', '')
    nv = int(fid.readline())
    data = np.empty((5, nv))
    for i, line in enumerate(fid):
        data[:, i] = line.split()

    label = dict()
    label['comment'] = comment[1:]
    label['vertices'] = np.array(data[0], dtype=np.int32)
    label['pos'] = 1e-3 * data[1:4].T
    label['values'] = data[4]

    basename = os.path.basename(filename)
    if basename.endswith('lh.label') or basename.startswith('lh.'):
        label['hemi'] = 'lh'
    elif basename.endswith('rh.label') or basename.startswith('rh.'):
        label['hemi'] = 'rh'
    else:
        raise ValueError('Cannot find which hemisphere it is. File should end'
                         ' with lh.label or rh.label')
    fid.close()

    return label


def write_label(filename, label):
    """Write a FreeSurfer label

    Parameters
    ----------
    filename : string
        Path to label file to produce.
    label : dict
        The label structure.
    """
    if not filename.endswith('lh.label') or not filename.endswith('rh.label'):
        filename += '-' + label['hemi'] + '.label'

    print 'Saving label to : %s' % filename

    fid = open(filename, 'wb')
    n_vertices = len(label['vertices'])
    data = np.zeros((n_vertices, 5), dtype=np.float)
    data[:, 0] = label['vertices']
    data[:, 1:4] = 1e3 * label['pos']
    data[:, 4] = label['values']
    fid.write("#%s\n" % label['comment'])
    fid.write("%d\n" % n_vertices)
    for d in data:
        fid.write("%d %f %f %f %f\n" % tuple(d))

    print '[done]'
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

    vertices = np.intersect1d(stc['vertices'], lab['vertices'])
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
    label : dict
        A label read with the read_label function
    src : list of dict
        The source space over which the label is defined

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
    if label['hemi'] == 'lh':
        vertno_sel = np.intersect1d(lh_vertno, label['vertices'])
        ori = src[0]['nn'][vertno_sel]
    elif label['hemi'] == 'rh':
        vertno_sel = np.intersect1d(rh_vertno, label['vertices'])
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
        The source space over which are defined the source estimates.
        If it's a string it should the subject name (e.g. fsaverage).

    Returns
    -------
    labels : list of dict
        The generated labels. One per hemisphere containing sources.
    """
    from scipy import sparse

    if not stc.is_surface():
        raise ValueError('SourceEstimate should be surface source '
                         'estimates')

    if isinstance(src, str):
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

        label = dict()
        label['comment'] = 'Label from stc'
        label['vertices'] = idx_use
        label['pos'] = this_rr[idx_use]
        label['values'] = np.ones(len(idx_use))
        label['hemi'] = hemi
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
    labels : list
        List with lables. Each label is a dictionary with keys:
            comment     Comment with seed vertex and extent
            hemi        Hemisphere of the label ('lh', or 'rh')
            vertices    Vertex indices (0 based)
            pos         Locations in millimeters
            values      Distances in millimeters from seed
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
        label = dict()
        label['comment'] = 'Circular label: seed=%d, extent=%0.1fmm' %\
                           (seed, extent)
        label['vertices'] = label_verts
        label['pos'] = vert[hemi][label_verts]
        label['values'] = label_dist
        label['hemi'] = hemi

        labels.append(label)

    return labels
