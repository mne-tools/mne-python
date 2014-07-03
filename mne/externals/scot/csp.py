# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""common spatial patterns (CSP) implementation"""

import numpy as np
from scipy.linalg import eig
    

def csp(x, cl, numcomp=np.inf):
    """ Calculate common spatial patterns (CSP)

    Parameters
    ----------
    x : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
        EEG data set
    cl : list of valid dict keys
        Class labels associated with each trial. Currently only two classes are supported.
    numcomp : {int}, optional
        Number of patterns to keep after applying the CSP. If `numcomp` is greater than n_channels, all n_channels
        patterns are returned.

    Returns
    -------
    w : array, shape = [n_channels, n_components]
        CSP weight matrix
    v : array, shape = [n_components, n_channels]
        CSP projection matrix
    """
    
    x = np.atleast_3d(x)
    cl = np.asarray(cl).ravel()
    
    n, m, t = x.shape
    
    if t != cl.size:
        raise AttributeError('CSP only works with multiple classes. Number of'
                             ' elemnts in cl (%d) must equal 3rd dimension of X (%d)' % (cl.size, t))

    labels = np.unique(cl)
    
    if labels.size != 2:
        raise AttributeError('CSP is currently ipmlemented for 2 classes (got %d)' % labels.size)
        
    x1 = x[:, :, cl == labels[0]]
    x2 = x[:, :, cl == labels[1]]
    
    sigma1 = np.zeros((m, m))
    for t in range(x1.shape[2]):
        sigma1 += np.cov(x1[:, :, t].transpose()) / x1.shape[2]
    sigma1 /= sigma1.trace()
    
    sigma2 = np.zeros((m, m))
    for t in range(x2.shape[2]):
        sigma2 += np.cov(x2[:, :, t].transpose()) / x2.shape[2]
    sigma2 /= sigma2.trace()
        
    e, w = eig(sigma1, sigma1 + sigma2, overwrite_a=True, overwrite_b=True, check_finite=False)

    order = np.argsort(e)[::-1]
    w = w[:, order]
    # e = e[order]
        
    v = np.linalg.inv(w)
   
    # subsequently remove unwanted components from the middle of w and v
    while w.shape[1] > numcomp:
        i = int(np.floor(w.shape[1]/2))
        w = np.delete(w, i, 1)
        v = np.delete(v, i, 0)
        
    return w, v
