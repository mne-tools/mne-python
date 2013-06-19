# Authors: Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg
from . import Epochs
from .cov import compute_covariance, compute_raw_data_covariance
from .fiff.pick import pick_types

class CSP(object):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)
    algorithm [1] 
        
    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.

    Parameters
    ----------
    n_components : int
        The number of maximum components.
    cov : instance of Covariance, sklearn.covariance or None
        The method to estimate covariance matrix of the signals
        if None MNE Covariance is used.
    
    [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.

    """
    
    def __init__(self, n_components=64):
        
        self.n_components = n_components
    
    
    def decompose_epochs(self, epoch_list):
        """Run the CSP decomposition on epochs.
        
        Parameters
        ----------
        epoch_list : list of Epochs
            The epochs for 2 classes. 
            The CSP is estimated on the concatenated epochs.
        
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        
        if len(epoch_list) != 2:
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        if not all([isinstance(e, Epochs) for e in epoch_list]):
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        covariance = []
        for epoch in epoch_list:
            # compute covariance for class a
            covariance.append(compute_covariance(epoch).data)
        
        # call decomposition algorithm
        self._decompose(*covariance)
        return self
    
    def decompose_covariance(self, cov_list):
        """Run the CSP decomposition on covariance matrix.
        
        Parameters
        ----------
        cov_list : list of Covariance
            The Covariance of 2 classes. 
        
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        
        if len(cov_list) != 2:
            raise ValueError('CSP decomposition needs 2 Covariance object')
        
        if not all([isinstance(c, Covariance) for c in cov_list]):
            raise ValueError('CSP decomposition needs 2 Covariance object')
        
        covariance = []
        for c in cov_list:
            # compute covariance for class a
            covariance.append(c.data)
        
        # call decomposition algorithm
        self._decompose(*covariance)
        return self
    
    
    def _decompose(self, cov_a, cov_b):
        """ Aux Function """
        
        cov_a /= np.trace(cov_a)
        cov_b /= np.trace(cov_b)
        # computes the eigen values
        (lambda_, u) = linalg.eig(cov_a + cov_b)
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = np.sort(lambda_)[::-1]
        
        u  = u[:,ind]
        p  = np.sqrt(linalg.pinv(np.diag(lambda2_.real))) * u.T
        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
        (g, b) = linalg.eig(w_a, w_b)
        # sort eigen values
        ind = np.argsort(g.real)
        b = b[:,ind].real
        # and project
        w = np.dot(b.T,p)
        
        self.filters  = w
        self.patterns = linalg.pinv(w).T
    
    
    def get_sources_epochs(self, epochs, pick_components = [0, -1]):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        pick_components : array
            The list of CSP filters to estimate the sources.
            default is first and last components corresponding to the largest
            and the smallest eigen values.
        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch.
        """
        if not hasattr(self, 'filters'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')
        
        if len(pick_components) > self.n_components:
            raise ValueError('picked components must be < n_components')
        
        return self._get_sources_epochs(epochs.get_data(), pick_components)
    
    def _get_sources_epochs(self, data, pick_components):
        """ Aux Function """
        return np.asarray([np.dot(self.filters, trial) for trial in data])
    
        
