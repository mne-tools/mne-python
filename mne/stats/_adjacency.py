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
            dim = sparse.eye(dim, format='coo')
            if dim.shape[0] > 1:
                dim += sparse.eye(dim.shape[0], k=1, format='coo')
                dim += sparse.eye(dim.shape[0], k=-1, format='coo')
        else:
            _check_option(f'{name}.ndim', dim.ndim, [2])
            if dim.shape[0] != dim.shape[1]:
                raise ValueError(
                    f'{name} must be square, got shape {dim.shape}')
        if not isinstance(dim, sparse.coo_matrix):
            dim = sparse.coo_matrix(dim)
        else:
            dim = dim.copy()
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
    nnz = n_each.sum()
    vertices = np.arange(np.prod(shape)).reshape(shape)
    edges = np.empty((2, nnz), int)
    used = np.zeros(nnz, bool)
    weights = np.empty(nnz, float)  # even though just 0/1
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
    assert weights.shape == (nnz,)
    assert edges.shape == (2,) + (nnz,)
    graph = sparse.coo_matrix((weights, edges),
                              (vertices.size, vertices.size))
    return graph
