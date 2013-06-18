# Authors: Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

from cov import compute_covariance, compute_raw_data_covariance
import numpy as np
import scipy.linalg

class CSP(object):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)
    algorithm 
    
    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problems.

    Parameters
    ----------
    components : list, array
        The list of components (aka filters) to decompose the signals
    n_components : int
        The number of maximum components
    cov_func : instance of Covariance, sklearn.covariance or None
        The method to estimate covariance matrix of the signals
        if None MNE Covariance is used
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    
    def __init__(self, components, n_components=64, cov_func=None, 
                 verbose=None):
        
        if components is not None and \
                len(components) > n_components:
            raise ValueError('components number must be smaller than '
                             'n_components')
        
        self.components   = components
        self.n_components = n_components
        self.cov_func = cov_func

    
    def decompose_raw(self, raws, cov_func=None, picks=None, verbose=None):
        """Run the CSP decomposition on raw objects
        
        Parameters
        ----------
        raws : list of Raw objects
            The CSP is estimated on raw signals.
        cov_func : instance of Covariance, sklearn.covariance or None
            The method to estimate covariance matrix of the signals
            if None MNE Covariance is used.
        picks : array-like
            Channels to be included relative to the channels already picked on
            epochs-initialization.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        
        n = data_a.shape[1]
        if cov_func == None:
            # compute covariance for class a
            cov_a = compute_raw_data_covariance(raws[0]).data
            cov_a /= np.trace(cov_a)
            # and for class b
            cov_b = compute_raw_data_covariance(raws[1]).data
            cov_b /= np.trace(cov_b)
        else:
            if (raw_signals[0]._preloaded) and (raw_signals[1]._preloaded):
                data_a = raw_signals[0]._data
                data_b = raw_signals[1]._data
            else:
                raise RuntimeError('Raw object needs to be preloaded before '
                                   'CSP decomposition.')
            
            # compute covariance for class a
            cov_func.fit(data_a)
            cov_a = cov_func.covariance_ / np.trace(cov.covariance_)
            # and for class b
            cov_func.fit(data_b)
            cov_b = cov_func.covariance_ / np.trace(cov.covariance_)
        # call decomposition algorithm
        self._decompose(cov_a,cov_b)
        return self
    
    
    def decompose_epochs(self, epochs, cov_func = None,
                         picks=None, verbose=None):
        """Run the CSP decomposition on epochs
        
        Parameters
        ----------
        epochs : list of Epochs
            The epochs for 2 classes. 
            The CSP is estimated on the concatenated epochs.
        cov_func : instance of Covariance, sklearn.covariance or None
            The method to estimate covariance matrix of the signals
            if None MNE Covariance is used.
        picks : array-like
            Channels to be included relative to the channels already picked on
            epochs-initialization.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        
        if len(epochs) != 2:
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        if not all([isinstance(e, mne.Epochs) for e in epochs]):
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        if picks is None:
            # just use epochs good data channels and avoid double picking
            picks = pick_types(epochs[0].info, include=epochs.ch_names,
                               exclude='bads')
        
        if cov_func == None:
            # compute covariance for class a
            cov_a = compute_covariance(epochs[0]).data
            cov_a /= np.trace(cov_a)
            # and for class b
            cov_b = compute_covariance(epochs[1]).data
            cov_b /= np.trace(cov_b)
        else:
            # compute concatenated epochs 
            data_a = np.transpose(epochs[0].get_data(),
                                  [1, 0, 2]).reshape(n, -1).T
            data_b = np.transpose(epochs[1].get_data(),
                                  [1, 0, 2]).reshape(n, -1).T
            # compute covariance for class a
            cov_func.fit(data_a)
            cov_a = cov_func.covariance_ / np.trace(cov.covariance_)
            # and for class b
            cov_func.fit(data_b)
            cov_b = cov_func.covariance_ / np.trace(cov.covariance_)
        # call decomposition algorithm
        self._decompose(cov_a,cov_b)
        return self
    
    
    def _decompose(self, cov_a, cov_b):
        """ Aux Function """
        # computes the eigen values
        (lambda_, u) = np.linalg.eig(cov_a + cov_b)
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = np.sort(lambda_)[::-1]
        
        u  = u[:,ind]
        p  = np.sqrt(np.linalg.pinv(np.diag(lambda2_.real))) * u.T
        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
        (g, b) = scipy.linalg.eig(w_a, w_b)
        # sort eigen values
        ind = np.argsort(g.real)
        b = b[:,ind].real
        # and project
        w = np.dot(b.T,p)
        
        self.csp_filters  = w
        self.csp_patterns = np.linalg.pinv(w).T
    
    
    def get_sources_epochs(self, epochs, components):
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        components : array of shape (n_components)
            The list of CSP filters to estimate the sources
        
        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch
        """
        if not hasattr(self, 'csp_filters'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        return self._get_sources_epochs(epochs, components)
    
    def _get_sources_epochs(self, epochs, components):
        """ Aux Function """
        data = epochs.get_data()
        (nEpoch,nSensor,nTime) = data.shape
        csp_data = np.zeros_like(data)
        for i in np.arange(nEpoch):
            csp_data[i,:,:] = np.dot(self.csp_filters,data[i,:,:])
        
        return csp_data
    
    
    def plot_sources_maps(self):
        """Plot CSP sources topographies
        """
        
