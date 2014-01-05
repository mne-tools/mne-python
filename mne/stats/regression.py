# Author: Tal Linzen <linzen@nyu.edu>
#
# License: Simplified BSD

import numpy as np
import scipy
from scipy import linalg

from mne.source_estimate import SourceEstimate

class OLSFit(object):
    def __init__(self, beta, stderr, t, p):
        self.beta = beta
        self.stderr = stderr
        self.t = t
        self.p = p

class OLS(object):
    
    def __init__(self, data, design_matrix, names):
        self.data = data
        self.design_matrix = design_matrix
        self.names = names

    def fit(self):
        n_trials = self.data.shape[0]
        n_rows, n_predictors = self.design_matrix.shape

        if n_trials != n_rows:
            raise ValueError('Number of rows in design matrix must be equal '
                             'to number of observations')
        if n_predictors != len(self.names):
            raise ValueError('Number of predictor names must be equal to '
                             'number of predictors')

        y = np.reshape(self.data, (n_trials, -1))
        betas, resid_sum_squares, _, _ = linalg.lstsq(self.design_matrix, y)

        df = n_rows - n_predictors
        sqrt_noise_var = np.sqrt(resid_sum_squares / df)
        sqrt_noise_var = sqrt_noise_var.reshape(self.data.shape[1:])
        design_invcov = linalg.inv(np.dot(self.design_matrix.T,
                                          self.design_matrix))
        unscaled_stderrs = np.sqrt(np.diag(design_invcov))

        beta = {}
        stderr = {}
        t = {}
        p = {}
        for x, unscaled_stderr, pred in zip(betas, unscaled_stderrs, 
                                            self.names):
            beta_map = x.reshape(self.data.shape[1:])
            beta[pred] = self.create_object(beta_map)

            stderr_map = sqrt_noise_var * unscaled_stderr
            stderr[pred] = self.create_object(stderr_map)

            t_map = beta_map / stderr_map
            t[pred] = self.create_object(t_map)

            cdf = scipy.stats.t.cdf(np.abs(t_map), df)
            p[pred] = self.create_object((1 - cdf) * 2)

        return OLSFit(beta, stderr, t, p)

    def create_object(self, data):
        return data

class OLSEpochs(OLS):

    def __init__(self, epochs, design_matrix, names):
        self.epochs = epochs
        self.evoked = epochs.average()
        data = epochs.get_data()
        super(OLSEpochs, self).__init__(data, design_matrix, names)

    def create_object(self, data):
        ev = self.evoked.copy()
        ev.data = data
        return ev

class OLSSourceEstimates(OLS):

    def __init__(self, source_estimates, design_matrix, names):
        self.source_estimates = source_estimates
        data = np.array([x.data for x in source_estimates])
        super(OLSSourceEstimates, self).__init__(data, design_matrix, names)

    def create_object(self, data):
        s = self.source_estimates[0]
        stc = SourceEstimate(data, vertices=s.vertno, tmin=s.tmin, 
                             tstep=s.tstep, subject=s.subject)
        return stc
