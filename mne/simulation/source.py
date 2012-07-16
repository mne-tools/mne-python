# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from scipy.sparse import csr_matrix

from ..surface import read_surface
from ..source_estimate import mesh_edges


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

    This function generates a number of labels in source space by growing regions
    starting from the vertices defined in "seeds". For each seed, a label is generated
    containing all vertices within a maximum geodesic distance on the white matter
    surface from the seed.

    Note: "extents" and "hemis" can either be arrays with the same length as seeds,
          which allows using a different extent and hemisphere for each label, or
          integers, in which case the same extent and hemisphere is used for each
          label.

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
            values      Values at the vertices (not used, all 1)
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
        patch_verts, _ = _verts_within_dist(dist[hemi], seed, extent)

        # create a label
        label = dict()
        label['comment'] = 'Circular label: seed=%d, extent=%0.1fmm' %\
                           (seed, extent)
        label['vertices'] = patch_verts
        label['pos'] = vert[hemi][patch_verts]
        label['values'] = np.ones(len(patch_verts))
        label['hemi'] = hemi

        labels.append(label)

    return labels

