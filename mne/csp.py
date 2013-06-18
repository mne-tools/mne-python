# Authors: Romain Trachel <romain.trachel@inria.fr>
#
# License: BSD (3-clause)

<<<<<<< HEAD
import numpy as np
from scipy import linalg
from . import Epochs
from .cov import compute_covariance, compute_raw_data_covariance
from .fiff.pick import pick_types
=======
from cov import compute_covariance, compute_raw_data_covariance
import numpy as np
import scipy.linalg
>>>>>>> 5460d76145923c12aa54da1b090222469868cead

class CSP(object):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP)
    algorithm 
    
    This object can be used as a supervised decomposition to estimate
<<<<<<< HEAD
    spatial filters for feature extraction in a 2 class decoding problem.

    Parameters
    ----------
    n_components : int
        The number of maximum components
    cov : instance of Covariance, sklearn.covariance or None
        The method to estimate covariance matrix of the signals
        if None MNE Covariance is used
    """
    
    def __init__(self, n_components=64, cov=None):
        
        self.n_components = n_components
        self.cov = cov
    
    
    def decompose_epochs(self, epochs, picks=None, verbose=None):
=======
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
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        """Run the CSP decomposition on epochs
        
        Parameters
        ----------
        epochs : list of Epochs
            The epochs for 2 classes. 
            The CSP is estimated on the concatenated epochs.
<<<<<<< HEAD
=======
        cov_func : instance of Covariance, sklearn.covariance or None
            The method to estimate covariance matrix of the signals
            if None MNE Covariance is used.
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
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
        
<<<<<<< HEAD
        if not all([isinstance(e, Epochs) for e in epochs]):
=======
        if not all([isinstance(e, mne.Epochs) for e in epochs]):
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
            raise ValueError('CSP decomposition needs 2 sets of epochs')
        
        if picks is None:
            # just use epochs good data channels and avoid double picking
<<<<<<< HEAD
            picks = pick_types(epochs[0].info, include=epochs[0].ch_names,
                               exclude='bads')
        
        if self.cov is None:
=======
            picks = pick_types(epochs[0].info, include=epochs.ch_names,
                               exclude='bads')
        
        if cov_func == None:
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
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
<<<<<<< HEAD
            self.cov.fit(data_a)
            cov_a = self.cov.covariance_ / np.trace(cov.covariance_)
            # and for class b
            self.cov.fit(data_b)
            cov_b = self.cov.covariance_ / np.trace(cov.covariance_)
        
=======
            cov_func.fit(data_a)
            cov_a = cov_func.covariance_ / np.trace(cov.covariance_)
            # and for class b
            cov_func.fit(data_b)
            cov_b = cov_func.covariance_ / np.trace(cov.covariance_)
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        # call decomposition algorithm
        self._decompose(cov_a,cov_b)
        return self
    
    
    def _decompose(self, cov_a, cov_b):
        """ Aux Function """
        # computes the eigen values
<<<<<<< HEAD
        (lambda_, u) = linalg.eig(cov_a + cov_b)
=======
        (lambda_, u) = np.linalg.eig(cov_a + cov_b)
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        # sort them
        ind = np.argsort(lambda_)[::-1]
        lambda2_ = np.sort(lambda_)[::-1]
        
        u  = u[:,ind]
<<<<<<< HEAD
        p  = np.sqrt(linalg.pinv(np.diag(lambda2_.real))) * u.T
=======
        p  = np.sqrt(np.linalg.pinv(np.diag(lambda2_.real))) * u.T
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        # Compute the generalized eigen value problem
        w_a = np.dot(np.dot(p, cov_a), p.T)
        w_b = np.dot(np.dot(p, cov_b), p.T)
        # and solve it
<<<<<<< HEAD
        (g, b) = linalg.eig(w_a, w_b)
=======
        (g, b) = scipy.linalg.eig(w_a, w_b)
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        # sort eigen values
        ind = np.argsort(g.real)
        b = b[:,ind].real
        # and project
        w = np.dot(b.T,p)
        
<<<<<<< HEAD
        self.filters  = w
        self.patterns = linalg.pinv(w).T
    
    
    def get_sources_epochs(self, epochs, pick_components = [0, -1]):
=======
        self.csp_filters  = w
        self.csp_patterns = np.linalg.pinv(w).T
    
    
    def get_sources_epochs(self, epochs, components):
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        """Estimate epochs sources given the CSP filters

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
<<<<<<< HEAD
        pick_components : array
            The list of CSP filters to estimate the sources
            default is first and last components corresponding to the largest
            and the smallest eigen values
=======
        components : array of shape (n_components)
            The list of CSP filters to estimate the sources
        
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch
        """
<<<<<<< HEAD
        if not hasattr(self, 'filters'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')
        
        if len(pick_components) > self.n_components:
            raise ValueError('picked components must be < n_components')
        
        return self._get_sources_epochs(epochs, pick_components)
    
    def _get_sources_epochs(self, epochs, pick_components):
=======
        if not hasattr(self, 'csp_filters'):
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        return self._get_sources_epochs(epochs, components)
    
    def _get_sources_epochs(self, epochs, components):
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
        """ Aux Function """
        data = epochs.get_data()
        (nEpoch,nSensor,nTime) = data.shape
        csp_data = np.zeros_like(data)
        for i in np.arange(nEpoch):
<<<<<<< HEAD
            csp_data[i,:,:] = np.dot(self.filters,data[i,:,:])
        
        return csp_data[:,pick_components,:]
=======
            csp_data[i,:,:] = np.dot(self.csp_filters,data[i,:,:])
        
        return csp_data
>>>>>>> 5460d76145923c12aa54da1b090222469868cead
    
    
    def plot_sources_maps(self):
        """Plot CSP sources topographies
        """
        
