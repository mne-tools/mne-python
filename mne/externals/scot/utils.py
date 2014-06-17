# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Utility functions """

from __future__ import division

import numpy as np

from functools import partial


def cuthill_mckee(matrix):
    """ Cuthill-McKee algorithm

    Permute a symmetric binary matrix into a band matrix form with a small bandwidth.

    Parameters
    ----------
    matrix : ndarray, dtype=bool, shape = [n, n]
        The matrix is internally converted to a symmetric matrix by setting each element [i,j] to True if either
        [i,j] or [j,i] evaluates to true.

    Returns
    -------
    order : list of int
        Permutation intices

    Examples
    --------
    >>> A = np.array([[0,0,1,1], [0,0,0,0], [1,0,1,0], [1,0,0,0]])
    >>> p = cuthill_mckee(A)
    >>> A
    array([[0, 0, 1, 1],
           [0, 0, 0, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 0]])
    >>> A[p,:][:,p]
    array([[0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 1, 1]])
    """
    matrix = np.atleast_2d(matrix)
    n, m = matrix.shape
    assert(n == m)

    # make sure the matrix is really symmetric. This is equivalent to
    # converting a directed adjacency matrix into a undirected adjacency matrix.
    matrix = np.logical_or(matrix, matrix.T)

    degree = np.sum(matrix, 0)
    order = [np.argmin(degree)]

    for i in range(n):
        adj = np.nonzero(matrix[order[i]])[0]
        adj = [a for a in adj if a not in order]
        if not adj:
            idx = [i for i in range(n) if i not in order]
            order.append(idx[np.argmin(degree[idx])])
        else:
            if len(adj) == 1:
                order.append(adj[0])
            else:
                adj = np.asarray(adj)
                i = adj[np.argsort(degree[adj])]
                order.extend(i.tolist())
        if len(order) == n:
            break

    return order


def acm(x, l):
    """ Autocovariance matrix at lag l

    This function calculates the autocovariance matrix of `x` at lag `l`.

    Parameters
    ----------
    x : ndarray, shape = [n_samples, n_channels, (n_trials)]
        Signal data (2D or 3D for multiple trials)
    l : int
        Lag

    Returns
    -------
    c : ndarray, shape = [nchannels, n_channels]
        Autocovariance matrix of `x` at lag `l`.
    """
    x = np.atleast_3d(x)

    if l > x.shape[0]-1:
        raise AttributeError("lag exceeds data length")

    ## subtract mean from each trial
    #for t in range(x.shape[2]):
    #    x[:, :, t] -= np.mean(x[:, :, t], axis=0)

    if l == 0:
        a, b = x, x
    else:
        a = x[l:, :, :]
        b = x[0:-l, :, :]

    c = np.zeros((x.shape[1], x.shape[1]))
    for t in range(x.shape[2]):
        c += a[:, :, t].T.dot(b[:, :, t]) / x.shape[0]
    c /= x.shape[2]

    return c


#noinspection PyPep8Naming
class memoize(object):
    """cache the return value of a method

    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    """

    def __init__(self, func):
        self.func = func

    #noinspection PyUnusedLocal
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    References
    ----------
    http://stackoverflow.com/a/1235363/3005167

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out