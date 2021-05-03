# -*- coding: utf-8 -*-

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np

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

    Notes
    -----
    For 4-dimensional data with shape ``(n_obs, n_times, n_freqs, n_chans)``,
    you can specify **no** connections among elements in a particular
    dimension by passing a matrix of zeros. For example:

    >>> import numpy as np
    >>> from scipy.sparse import diags
    >>> from mne.stats import combine_adjacency
    >>> n_times, n_freqs, n_chans = (50, 7, 16)
    >>> chan_adj = diags([1., 1.], offsets=(-1, 1), shape=(n_chans, n_chans))
    >>> combine_adjacency(
    ...     n_times,  # regular lattice adjacency for times
    ...     np.zeros((n_freqs, n_freqs)),  # no adjacency between freq. bins
    ...     chan_adj,  # custom matrix, or use mne.channels.find_ch_adjacency
    ...     )  # doctest: +NORMALIZE_WHITESPACE
    <5600x5600 sparse matrix of type '<class 'numpy.float64'>'
            with 27076 stored elements in COOrdinate format>
    """
    from scipy import sparse
    structure = list(structure)
    for di, dim in enumerate(structure):
        name = f'structure[{di}]'
        _validate_type(dim, ('int-like', np.ndarray, sparse.spmatrix), name)
        if isinstance(dim, int_like):
            # Don't add the diagonal, because we explicitly remove it later
            dim = sparse.diags([1, 1],
                               offsets=(-1, 1),
                               shape=(int(dim), int(dim)),
                               dtype=float).tocoo()
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
