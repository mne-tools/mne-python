# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from sklearn.decomposition import FastICA
import numpy as np
from ..cov import compute_whitener
from scipy.stats import kurtosis


class ICA(object):
    """ MEG signal decomposition and denoising workflow
    Paramerters
    -----------
    raw : instance of Raw
        MEG raw data
    noise_cov : ndarray
        noise covariance used for whitening
    picks : array-like
        indices for channel selection as returned from mne.fiff.pick_channels
    n_components : integer
        number of components to extract. If none, no dimension
        reduciton will be applied.
    start : integer
        starting time slice
    stop : integer
        final time slice
    sort_method : string
        function used to sort the sources and the mixing matrix.
        If None, the output will be left unsorted

    Attributes
    ----------
    raw : instance of Raw
        raw object used for initializing the ica seed
    n_components : integer
        number of components to be extracted
    comp_picks : ndrarray
        components selection array
    whitener : ndrarray
        whiter used for preprocessing
    seed : instance of FastICA
        FastICA instance used to perform decomposition
    sort_func : function
        function used for sorting the components
    """
    def __init__(self, raw, picks, noise_cov=None, start=None, stop=None,
                 n_components=None, sort_method='kurtosis', exclude=None):

        self.raw = raw
        self.n_components = n_components
        self._cov = True if noise_cov != None else False
        _n_comps = n_components if n_components != None else picks.shape[0]

        self.comp_picks = np.zeros(_n_comps, dtype=np.bool)
        self.comp_picks.fill(True)

        if noise_cov != None:
            self.whitener, _ = compute_whitener(noise_cov, raw.info,
                                                picks, exclude)
            del _

        elif noise_cov == None:
            if raw._preloaded:
                self.raw_data = raw._data[picks]
            else:
                start = raw.first_samp if start == None else start
                stop = raw.last_samp if stop == None else stop
                self.raw_data = raw[picks, start:stop]
            std_chan = np.std(self.raw_data, axis=1) ** -1
            self.whitener = np.array([std_chan]).T

        self.seed = FastICA(n_components)

        self.sort_func = None
        if sort_method == 'var':
            self.sort_func = np.var
        elif sort_method == 'kurtosis':
            self.sort_func = kurtosis
        elif isinstance(sort_method, str):
            print ('\n    This is not a valid sorting function'
                   '\n    --- No sort will be applied.')

        # if self.sort_func != None:
    def __repr__(self):
        out = 'ICA object.\n    %i components' % self.n_components
        if self.n_selected == self.n_components:
            nsel_msg = 'all'
        else:
            nsel_msg = str(self.n_selected)
        out += ' (%s active)' % nsel_msg
        if hasattr(self, 'raw_sources'):
            out += ('\n    %i raw time slices' % self.n_selected)
        return out

    def _sort(self, X):
        sorter = self.sort_func(X, 1)
        X_sorted = X[np.argsort(sorter)]
        return X_sorted

    def fit_raw(self):
        """ Run the ica decomposition for raw data
        """
        print ('\nComputing signal decomposition on raw data.'
               '\n    Please be patient. This may take some time')
        S = self.seed.fit_transform(self.raw_data.T)
        A = self.seed.get_mixing_matrix()
        self.raw_sources = self._sort(S).T
        self.raw_mixing = self._sort(A).T

    def fit_epochs(self, epochs, picks):
        pass

    def denoise_raw(self, bads=[]):
        """ Recompose raw data

        Paramerters
        -----------
        bads : list-like
            Indices for transient component deselection

        Returns
        -------
        out : n pickked channels x n time slices ndrarray
            denoised raw data
        """
        comp_picks = self.comp_picks.copy()
        if bads != None:
            comp_picks.put(bads, False)
        w = self.whitener ** -1 if self._cov == False else self.whitener
        raw_sources = self.raw_sources.copy()
        raw_sources.put(comp_picks, 0)
        out = np.dot(raw_sources.T, w.T * self.raw_mixing)
        return out.T

    def denoise_epochs(self, bads=[]):
        pass

    def set_comps(self, comps, state=False):
        """ Permanently set good and bad components

        Paramerters
        -----------
        comps: list-like
            indices for component selection
        """
        self.comp_picks.put(comps, state)

    @property
    def n_selected(self):
        """ Return number of selected sources
        """
        return self.comp_picks.nonzero()[0].shape[0]
