# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""principal component analysis (PCA) implementation"""

import numpy as np
from ..datatools import cat_trials


def pca_svd(data):
    """calculate PCA using SVD
    
    Parameters
    ----------
    data : array, shape = [n_samples, n_channels]
        Two dimensional data array.
        
    Returns
    -------
    w : array
        Eigenvectors
    s : array
        Eigenvalues
    """

    (w, s, v) = np.linalg.svd(data.transpose())

    return w, s ** 2


def pca_eig(x):
    """calculate PCA using Eigenvalue decomposition
    
    Parameters
    ----------
    data : array, shape = [n_samples, n_channels]
        Two dimensional data array.
        
    Returns
    -------
    w : array
        Eigenvectors
    s : array
        Eigenvalues
    """

    [s, w] = np.linalg.eigh(x.transpose().dot(x))

    return w, s


def pca(x, subtract_mean=False, normalize=False, sort_components=True, reducedim=None, algorithm=pca_eig):
    """ Calculate principal component analysis (PCA)
    
    Parameters
    ----------
    x : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
        EEG data set
    subtract_mean : bool, optional
        Subtract sample mean from x.
    normalize : bool, optional
        Normalize variances to 1 before applying PCA.
    sort_components : bool, optional
        Sort principal components in order of decreasing eigenvalues.
    reducedim : {float, int}, optional
        A number of less than 1 in interpreted as the fraction of variance that should remain in the data. All
        components that describe in total less than `1-reducedim` of the variance are removed by the PCA step.
        An integer numer of 1 or greater is interpreted as the number of components to keep after applying the PCA.
    algorithm : func, optional
        Specify function to use for eigenvalue decomposition (:func:`pca_eig` or :func:`pca_svd`)
        
    Returns
    -------
    w : array, shape = [n_channels, n_components]
        PCA transformation matrix
    v : array, shape = [n_components, n_channels]
        PCA backtransformation matrix
    """

    x = cat_trials(np.atleast_3d(x))

    if reducedim:
        sort_components = True

    if subtract_mean:
        for i in range(np.shape(x)[1]):
            x[:, i] -= np.mean(x[:, i])

    k, l = None, None
    if normalize:
        l = np.std(x, 0, ddof=1)
        k = np.diag(1.0 / l)
        l = np.diag(l)
        x = x.dot(k)

    w, latent = algorithm(x)

    #v = np.linalg.inv(w)
    # PCA is just a rotation, so inverse is equal transpose...
    v = w.T

    if normalize:
        w = k.dot(w)
        v = v.dot(l)

    latent /= sum(latent)

    if sort_components:
        order = np.argsort(latent)[::-1]
        w = w[:, order]
        v = v[order, :]
        latent = latent[order]

    if reducedim and reducedim < 1:
        selected = np.nonzero(np.cumsum(latent) < reducedim)[0]
        try:
            selected = np.concatenate([selected, [selected[-1] + 1]])
        except IndexError:
            selected = [0]
        if selected[-1] >= w.shape[1]:
            selected = selected[0:-1]
        w = w[:, selected]
        v = v[selected, :]

    if reducedim and reducedim >= 1:
        w = w[:, np.arange(reducedim)]
        v = v[np.arange(reducedim), :]

    return w, v
