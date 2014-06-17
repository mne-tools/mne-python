# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Use scikit-learn routines as backend.
"""

from __future__ import absolute_import

from sklearn.decomposition import FastICA, PCA
from sklearn import linear_model
import scipy as sp
from . import backend_builtin as builtin
from . import config, datatools
from .varbase import VARBase


def wrapper_fastica(data):
    """ Call FastICA implementation from scikit-learn.
    """
    ica = FastICA()
    ica.fit(datatools.cat_trials(data))
    u = ica.components_.T
    m = ica.mixing_.T
    return m, u


def wrapper_pca(x, reducedim):
    """ Call PCA implementation from scikit-learn.
    """
    pca = PCA(n_components=reducedim)
    pca.fit(datatools.cat_trials(x))
    d = pca.components_
    c = pca.components_.T
    y = datatools.dot_special(x,c)
    return c, d, y


class VAR(VARBase):
    """ Scikit-learn based implementation of VARBase.

    This class fits VAR models using various implementations of generalized linear model fitting available in scikit-learn.
    
    Parameters    
    ----------
    model_order : int
        Autoregressive model order
    fitobj : class, optional
        Instance of a linear model implementation.
    """
    def __init__(self, model_order, fitobj=linear_model.LinearRegression()):
        VARBase.__init__(self, model_order)
        self.fitting_model = fitobj

    def fit(self, data):
        """ Fit VAR model to data.
        
        Parameters
        ----------
        data : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
            Continuous or segmented data set.
            
        Returns
        -------
        self : :class:`VAR`
            The :class:`VAR` object.
        """
        data = sp.atleast_3d(data)
        (x, y) = self._construct_eqns(data)
        self.fitting_model.fit(x, y)

        self.coef = self.fitting_model.coef_

        self.residuals = data - self.predict(data)
        self.rescov = sp.cov(datatools.cat_trials(self.residuals[self.p:, :, :]), rowvar=False)

        return self


backend = builtin.backend.copy()
backend.update({
    'ica': wrapper_fastica,
    'pca': wrapper_pca,
    'var': VAR
})


def activate():
    """ Set backend attribute in the config module.
    """
    config.backend = backend


activate()
