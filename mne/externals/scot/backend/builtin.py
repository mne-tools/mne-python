# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Use internally implemented functions as backend.
"""

import numpy as np

from .. import config
from .. import datatools

from ..builtin import binica, pca, csp
from ..builtin.var import VAR
from ..builtin import utils


def wrapper_binica(data):
    """ Call binica for ICA calculation.
    """
    w, s = binica.binica(datatools.cat_trials(data))
    u = s.dot(w)
    m = np.linalg.inv(u)
    return m, u

def wrapper_pca(x, reducedim):
    """ Call SCoT's PCA algorithm.
    """
    c, d = pca.pca(datatools.cat_trials(x), subtract_mean=False, reducedim=reducedim)
    y = datatools.dot_special(x, c)
    return c, d, y
    
def wrapper_csp(x, cl, reducedim):
    c, d = csp.csp(x, cl, numcomp=reducedim)
    y = datatools.dot_special(x,c)
    return c, d, y


backend = {
    'ica': wrapper_binica,
    'pca': wrapper_pca,
    'csp': wrapper_csp,
    'var': VAR,
    'utils': utils
}


def activate():
    config.backend = backend


activate()
