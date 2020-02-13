# Authors: Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: Simplified BSD


import numpy as np
from scipy import sparse

from ..utils import logger, verbose


def mesh_edges(faces):
    """Return sparse matrix with edges as an adjacency matrix.

    Parameters
    ----------
    faces : array, shape (n_triangles, 3)
        The mesh faces

    Returns
    -------
    edges : sparse matrix
        The adjacency matrix
    """
    npoints = np.max(faces) + 1
    nfaces = len(faces)
    a, b, c = faces.T
    edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                              shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                      shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                      shape=(npoints, npoints))
    edges = edges + edges.T
    edges = edges.tocoo()
    return edges


def _nearest(vertices, adj_mat):
    from scipy.sparse.csgraph import dijkstra
    # Vertices can be out of order, so sort them to start ...
    order = np.argsort(vertices)
    vertices = vertices[order]
    _, _, sources = dijkstra(adj_mat, False, indices=vertices, min_only=True,
                             return_predecessors=True)
    col = np.searchsorted(vertices, sources)
    # ... then get things back to the correct configuration.
    col = order[col]
    row = np.arange(len(col))
    data = np.ones(len(col))
    mat = sparse.coo_matrix((data, (row, col)))
    assert mat.shape == (adj_mat.shape[0], len(vertices)), mat.shape
    return mat


@verbose
def smoothing_matrix(vertices, adj_mat, smoothing_steps=20, verbose=None):
    """Create a smoothing matrix.

    This smoothing matrix can be used to interpolate data defined
    for a subset of vertices onto mesh with an adjancency matrix given by
    adj_mat.

    If smoothing_steps is None, as many smoothing steps are applied until
    the whole mesh is filled with with non-zeros. Only use this option if
    the vertices correspond to a subsampled version of the mesh.

    Parameters
    ----------
    vertices : 1d array
        vertex indices
    adj_mat : sparse matrix
        N x N adjacency matrix of the full mesh
    smoothing_steps : int or None
        number of smoothing steps (Default: 20)
    %(verbose)s

    Returns
    -------
    smooth_mat : sparse matrix
        smoothing matrix with size N x len(vertices)
    """
    from scipy import sparse

    logger.info("Updating smoothing matrix, be patient..")

    if smoothing_steps == 0:
        return _nearest(vertices, adj_mat)

    e = adj_mat.copy()
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    e = e + sparse.eye(n_vertices, n_vertices)
    idx_use = vertices
    smooth_mat = 1.0
    n_iter = smoothing_steps if smoothing_steps is not None else 1000
    for k in range(n_iter):
        e_use = e[:, idx_use]

        data1 = e_use * np.ones(len(idx_use))
        idx_use = np.where(data1)[0]
        scale_mat = sparse.dia_matrix((1 / data1[idx_use], 0),
                                      shape=(len(idx_use), len(idx_use)))

        smooth_mat = scale_mat * e_use[idx_use, :] * smooth_mat

        logger.info("Smoothing matrix creation, step %d" % (k + 1))
        if smoothing_steps is None and len(idx_use) >= n_vertices:
            break

    # Make sure the smoothing matrix has the right number of rows
    # and is in COO format
    smooth_mat = smooth_mat.tocoo()
    smooth_mat = sparse.coo_matrix((smooth_mat.data,
                                    (idx_use[smooth_mat.row],
                                     smooth_mat.col)),
                                   shape=(n_vertices,
                                          len(vertices)))

    return smooth_mat
