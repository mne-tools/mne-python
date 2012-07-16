# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
import os

import numpy as np
from scipy.sparse import csr_matrix

from ..minimum_norm.inverse import _make_stc
from ..utils import check_random_state
from ..surface import read_surface
from ..source_estimate import mesh_edges


def select_source_in_label(fwd, label, random_state=None):
    """Select source positions using a label

    Parameters
    ----------
    fwd : dict
        a forward solution
    label : dict
        the label (read with mne.read_label)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    lh_vertno : list
        selected source coefficients on the left hemisphere
    rh_vertno : list
        selected source coefficients on the right hemisphere
    """
    lh_vertno = list()
    rh_vertno = list()

    rng = check_random_state(random_state)

    if label['hemi'] == 'lh':
        src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label['vertices'])
        idx_select = rng.randint(0, len(src_sel_lh), 1)
        lh_vertno.append(src_sel_lh[idx_select][0])
    else:
        src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label['vertices'])
        idx_select = rng.randint(0, len(src_sel_rh), 1)
        rh_vertno.append(src_sel_rh[idx_select][0])

    return lh_vertno, rh_vertno


def generate_sparse_stc(fwd, labels, stc_data, tmin, tstep, random_state=0):
    """Generate sources time courses from waveforms and labels

    Parameters
    ----------
    fwd : dict
        a forward solution
    labels : list of dict
        The labels
    stc_data : array
        The waveforms
    tmin : float
        The beginning of the timeseries
    tstep : float
        The time step (1 / sampling frequency)
    random_state : None | int | np.random.RandomState
        To specify the random generator state.

    Returns
    -------
    stc : SourceEstimate
        The generated source time courses.
    """
    rng = check_random_state(random_state)
    vertno = [[], []]
    for label in labels:
        lh_vertno, rh_vertno = select_source_in_label(fwd, label, rng)
        vertno[0] += lh_vertno
        vertno[1] += rh_vertno
    vertno = map(np.array, vertno)
    stc = _make_stc(stc_data, tmin, tstep, vertno)
    return stc


def mesh_dist(tris, vert):
    """ Generate an adjacency matrix where the entries are the distances
        between neighboring vertices

    Parameters:
    -----------
    tris : array (n_tris x 3)
        Mesh triangulation
    vert : array (n_vert x 3)
        Vertex locations

    Returns:
    --------
    dist_matrix : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
    """
    edges = mesh_edges(tris).tocoo()

    # Euclidean distances between neighboring vertices
    dist = np.sqrt(np.sum((vert[edges.row, :] - vert[edges.col, :]) ** 2,
                          axis=1))

    dist_matrix = csr_matrix((dist, (edges.row, edges.col)), shape=edges.shape)

    return dist_matrix


def _verts_within_dist(graph, source, max_dist):
    """ Find all vertices wihin a maximum geodesic distance from source

    Parameters:
    -----------
    graph : scipy.sparse.csr_matrix
        Sparse matrix with distances between adjacent vertices
    source : int
        Source vertex
    max_dist: float
        Maximum geodesic distance

    Returns:
    --------
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


def circular_source_labels(subject, seeds, extents, hemis, subjects_dir=None):
    """ Generate circular labels in source space

    This function generates a number of labels in source space by growing
    regions starting from the vertices defined in "seeds". For each seed, a
    label is generated containing all vertices within a maximum geodesic
    distance on the white matter surface from the seed.

    Note: "extents" and "hemis" can either be arrays with the same length as
          seeds, which allows using a different extent and hemisphere for each
          label, or integers, in which case the same extent and hemisphere is
          used for each label.

    Parameters:
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

    Returns:
    --------
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

