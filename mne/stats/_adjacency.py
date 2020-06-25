# -*- coding: utf-8 -*-

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
from scipy import sparse

from ..utils import _validate_type, _check_option
from ..utils.check import int_like


def combine_adjacency(*structure):
    """Create a sparse binary adjacency/neighbors matrix.

    Parameters
    ----------
    *structure : list
        The adjacency along each dimension. Each entry can be:

        - ndarray or sparse matrix
            A square binary adjacency matrix for the given dimension.
        - int
            The number of elements along the given dimension. A lattice
            adjacency will be generated.

    Returns
    -------
    adjacency : scipy.sparse.coo_matrix, shape (n_features, n_features)
        The adjacency matrix.
    """
    structure = list(structure)
    for di, dim in enumerate(structure):
        name = f'structure[{di}]'
        _validate_type(dim, ('int-like', np.ndarray, sparse.spmatrix), name)
        if isinstance(dim, int_like):
            dim = int(dim)
            # Don't add the diagonal, because we explicitly remove it later:
            # dim = sparse.eye(dim, format='coo')
            # dim += sparse.eye(dim.shape[0], k=1, format='coo')
            # dim += sparse.eye(dim.shape[0], k=-1, format='coo')
            ii, jj = np.arange(0, dim - 1), np.arange(1, dim)
            edges = np.vstack([np.hstack([ii, jj]), np.hstack([jj, ii])])
            dim = sparse.coo_matrix(
                (np.ones(edges.shape[1]), edges), (dim, dim), float)
        else:
            _check_option(f'{name}.ndim', dim.ndim, [2])
            if dim.shape[0] != dim.shape[1]:
                raise ValueError(
                    f'{name} must be square, got shape {dim.shape}')
            if not isinstance(dim, sparse.coo_matrix):
                dim = sparse.coo_matrix(dim)
            else:
                dim = dim.copy()
        dim.data[dim.row == dim.col] = 0.  # remove diagonal, will add later
        dim.eliminate_zeros()
        if not (dim.data == 1).all():
            raise ValueError('All adjacency values must be 0 or 1')
        structure[di] = dim
    # list of coo
    assert all(isinstance(dim, sparse.coo_matrix) for dim in structure)
    shape = np.array([d.shape[0] for d in structure], int)
    n_others = np.array([np.prod(np.concatenate([shape[:di], shape[di + 1:]]))
                         for di in range(len(structure))], int)
    n_each = np.array([dim.data.size for dim in structure], int) * n_others
    n_off = n_each.sum()  # off-diagonal terms
    n_diag = np.prod(shape)
    vertices = np.arange(n_diag).reshape(shape)
    edges = np.empty((2, n_off + n_diag), int)
    used = np.zeros(n_off, bool)
    weights = np.empty(n_off + n_diag, float)  # even though just 0/1
    offset = 0
    for di, dim in enumerate(structure):
        s_l = [slice(None)] * len(shape)
        s_r = [slice(None)] * len(shape)
        s_l[di] = dim.row
        s_r[di] = dim.col
        assert dim.row.shape == dim.col.shape == dim.data.shape
        sl = slice(offset, offset + n_each[di])
        edges[:, sl] = [vertices[tuple(s_l)].ravel(),
                        vertices[tuple(s_r)].ravel()]
        weights[sl] = np.tile(dim.data, n_others[di])
        offset += n_each[di]
        assert not used[sl].any()
        used[sl] = True
    assert used.all()
    # Handle the diagonal separately at the end to avoid duplicate entries
    edges[:, n_off:] = vertices.ravel()
    weights[n_off:] = 1.
    graph = sparse.coo_matrix((weights, edges),
                              (vertices.size, vertices.size))
    return graph
