# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
import numpy as np


def check_indices(indices):
    """Check indices parameter."""
    if not isinstance(indices, tuple) or len(indices) != 2:
        raise ValueError('indices must be a tuple of length 2')

    if len(indices[0]) != len(indices[1]):
        raise ValueError('Index arrays indices[0] and indices[1] must '
                         'have the same length')

    return indices


def seed_target_indices(seeds, targets):
    """Generate indices parameter for seed based connectivity analysis.

    Parameters
    ----------
    seeds : array of int | int
        Seed indices.
    targets : array of int | int
        Indices of signals for which to compute connectivity.

    Returns
    -------
    indices : tuple of array
        The indices parameter used for connectivity computation.
    """
    # make them arrays
    seeds = np.asarray((seeds,)).ravel()
    targets = np.asarray((targets,)).ravel()

    n_seeds = len(seeds)
    n_targets = len(targets)

    indices = (np.concatenate([np.tile(i, n_targets) for i in seeds]),
               np.tile(targets, n_seeds))

    return indices


def degree(connectivity, threshold=1.):
    """Compute the undirected degree of a connectivity matrix.

    Parameters
    ----------
    connectivity : ndarray, shape (n_nodes, n_nodes)
        The connectivity matrix.
    threshold : float
        The proportion of activations to keep before computing
        the degree.

    Returns
    -------
    degree : ndarray, shape (n_nodes,)
        The computed degree.
    """
    connectivity = np.array(connectivity)
    threshold = float(threshold)
    if not 0 < threshold <= 1:
        raise ValueError('threshold must be 0 <= threshold < 1, got %s'
                         % (threshold,))
    if connectivity.ndim != 2 or \
            connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError('connectivity must be have shape (n_nodes, n_nodes), '
                         'got %s' % (connectivity.shape,))
    degree = connectivity.ravel()  # no need to copy because np.array does
    degree[::connectivity.shape[0] + 1] = 0
    n_keep = int(round((degree.size - len(connectivity)) * threshold))
    degree[np.argsort(degree)[:-n_keep]] = 0
    degree.shape = connectivity.shape
    degree = np.sum(degree > 0, axis=0)
    return degree
