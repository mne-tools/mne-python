# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Source decomposition with ICA.
"""

import numpy as np

from . import config
from .datatools import cat_trials


class ResultICA:
    """ Result of :func:`plainica`

    Attributes
    ----------
    `mixing` : array
        estimate of the mixing matrix
    `unmixing` : array
        estimate of the unmixing matrix
    """
    def __init__(self, mx, ux):
        self.mixing = mx
        self.unmixing = ux


def plainica(x, reducedim=0.99, backend=None):
    """ Source decomposition with ICA.

    Apply ICA to the data x, with optional PCA dimensionality reduction.

    Parameters
    ----------
    x : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
        data set
    reducedim : {int, float, 'no_pca'}, optional
        A number of less than 1 in interpreted as the fraction of variance that should remain in the data. All
        components that describe in total less than `1-reducedim` of the variance are removed by the PCA step.
        An integer numer of 1 or greater is interpreted as the number of components to keep after applying the PCA.
        If set to 'no_pca' the PCA step is skipped.
    backend : dict-like, optional
        Specify backend to use. When set to None the backend configured in config.backend is used.

    Returns
    -------
    result : ResultICA
        Source decomposition
    """

    x = np.atleast_3d(x)
    l, m, t = np.shape(x)

    if backend is None:
        backend = config.backend

    # pre-transform the data with PCA
    if reducedim == 'no pca':
        c = np.eye(m)
        d = np.eye(m)
        xpca = x
    else:
        c, d, xpca = backend['pca'](x, reducedim)

    # run on residuals ICA to estimate volume conduction    
    mx, ux = backend['ica'](cat_trials(xpca))

    # correct (un)mixing matrix estimatees
    mx = mx.dot(d)
    ux = c.dot(ux)

    class Result:
        unmixing = ux
        mixing = mx

    return Result
