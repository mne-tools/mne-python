# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from inspect import getargspec
from copy import copy
import numpy as np
from scipy.stats import kurtosis, skew
from scipy import linalg

from ..cov import compute_whitener
from ..fiff.raw import RawFromMerge
from ..epochs import EpochsFromMerge


class ICA(object):
    """MEG signal decomposition and denoising workflow

    Paramerters
    -----------
    noise_cov : ndarray
        noise covariance used for whitening
    n_components : integer
        number of components to be extracted. If None, no dimensionality
        reduction will be applied.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results.

    Attributes
    ----------
    pre_whitener : ndrarray | instance of mne.cov.Covariance
        whiter used for preprocessing
    source_ids : np.ndrarray
        array of source ids for sorting-book-keeping
    sorted_by : str
        flag informing about the active
    last_fit : str
        flag informing about which type was last fit.
    """
    def __init__(self, noise_cov=None, n_components=None, random_state=None):
        from sklearn.decomposition import FastICA
        self.noise_cov = noise_cov
        self._fast_ica = FastICA(n_components, random_state=random_state)
        self.n_components = n_components
        self.last_fit = 'unfitted'
        self.sorted_by = 'unsorted'
        self.sources = None

    def __repr__(self):
        out = 'ICA '
        if self.last_fit == 'unfitted':
            msg = '(no decomposition, '
        elif self.last_fit == 'raw':
            msg = '(raw data decomposition, '
        else:
            msg = '(epochs decomposition, '

        out += msg + '%i components' % self.n_components

        if self.last_fit != 'unfitted':
            out += (', %i time slices' % self.sources.shape[1])

        if self.sorted_by == 'unsorted':
            sorted_by = self.sorted_by
        else:
            sorted_by = 'sorted by %s' % self.sorted_by
        out += ', %s)' % sorted_by

        return out

    def fit_raw(self, raw, picks, start=None, stop=None,
                sort_method='skew'):
        """Run the ica decomposition for raw data

        Paramerters
        -----------
        raw : instance of mne.fiff.Raw
            raw measurments to be decomposed
        start : integer
            starting time slice
        stop : integer
            final time slice
        picks : array-like
            channels to be included.
        sort_method : string
            function used to sort the sources and the mixing matrix.
            If None, the output will be left unsorted

        Returns
        -------
        self : instance of ICA
            returns the instance for chaining
        """
        print ('\nComputing signal decomposition on raw data.'
               '\n    Please be patient. This may take some time')

        self.source_ids = (np.arange(self.n_components) if self.n_components
                           is not None else np.arange(picks.shape[0]))

        start = raw.first_samp if start is None else start
        stop = raw.last_samp if stop is None else stop
        data = raw[picks, start:stop][0]

        data, self.pre_whitener = self._pre_whiten(data, picks)

        self._fit_data(data, sort_method=sort_method)
        self.last_fit = 'raw'
        self.data = data
        self.picks = picks
        self._raw = raw

        return self

    def fit_epochs(self, epochs, picks=None, sort_method='skew'):
        """Run the ica decomposition for epochs

        Paramerters
        -----------
        epochs : instance of Epochs
            The epochs. The ICA is estimated on the concatenated epochs.

        sort_method : string
            function used to sort the sources and the mixing matrix.
            If None, the output will be left unsorted

        Returns
        -------
        self : instance of ICA
            returns the instance for chaining
        """
        data = np.hstack(epochs.get_data())
        self.picks = epochs.picks
        self.source_ids = (np.arange(self.n_components) if self.n_components
                           is not None else np.arange(self.picks.shape[0]))

        print ('\nComputing signal decomposition on epochs.'
               '\n    Please be patient. This may take some time')

        data, self.pre_whitener = self._pre_whiten(data, picks=epochs.picks)
        self._fit_data(data, sort_method=sort_method)
        self.last_fit = 'epochs'
        self._epochs = epochs
        return self

    def denoise_raw(self, bads=[], copy=True):
        """Recompose raw data

        Paramerters
        -----------
        bads : list-like
            Indices for transient component deselection
        make_raw : boolean
            Either return denoised data as nd array or newly instantiated
            Raw object.

        Returns
        -------
        denoised : depends on input arguments
            denoised raw data as ndarray or as instance of Raw
        """
        if self.last_fit != 'raw':
            raise ValueError('Currently no raw data fitted.'
                             'Please fit raw data first.')

        denoised = self._denoise(bads)

        if copy:
            return RawFromMerge(self._raw, data=denoised, picks=self.picks)

        self._raw._data[self.picks] = denoised

    def denoise_epochs(self, bads=[], copy=True):
        """Recompose epochs

        Paramerters
        -----------
        bads : list-like
            Indices for transient component deselection
        copy : boolean
            Either return denoised data as nd array or newly instantiated
            Epochs object.

        Returns
        -------
        denoised : depends on input arguments
            denoised raw data as ndarray or as instance of Raw
        """
        if self.last_fit != 'epochs':
            raise ValueError('Currently no epochs fitted.'
                             'Please fit epochs first.')

        denoised = self._denoise(bads)
        neps = self._epochs.events.shape[0]
        denoised = np.array(np.split(denoised, neps, 1))

        if copy:
            return EpochsFromMerge(self._epochs, denoised)
        else:
            self._epochs._data = denoised
            self._epochs._preload = True

            return self._epochs
        #TODO  alternative epochs constructor to restore epochs object

    def sort_sources(self, sort_method, inplace=True):
        """Sort sources accoroding to criteria such as skewness or kurtosis

        Paramerters
        -----------
        sources : str
            string for selecting the sources
        sort_method : string
            function used to sort the sources and the mixing matrix.
            If None, the output will be left unsorted
        """
        if self.sources is None:
            print ('No sources availble. First fit ica decomposition first.')

        if sort_method == 'skew':
            sort_func = skew
        elif sort_method == 'kurtosis':
            sort_func = kurtosis
        elif sort_method == 'back':
            sort_func = lambda x, y: self.source_ids
            sort_func.__name__ = 'unsorted'
        elif callable(sort_method):
            args = getargspec(sort_method).args
            if len(args) > 1:
                if args[:2] == ['a', 'axis']:
                    sort_func = sort_method
            else:
                ValueError('%s is not a valid function.'
                           'The function needs an array and'
                           'an axis argument' % sort_method.__name__)
        elif isinstance(sort_method, str):
            ValueError('%s is not a valid sorting option' % sort_method)

        self.sorted_by = sort_func.__name__

        sort_args = np.argsort(sort_func(self.sources, 1))

        S = self.sources[sort_args]
        A = self.mixing[sort_args]
        ID = self.source_ids[sort_args]

        if inplace:
            self.sources = S
            self.mixing = A
            self.source_ids = ID

        return S, A, ID

    def _pre_whiten(self, data, picks):
        """Helper function"""
        if self.noise_cov is not None:
            assert data.shape[0] == self.noise_cov.data.shape[0]
            pre_whitener, _ = compute_whitener(self.noise_cov, self.raw.info,
                                               picks)
            data = np.dot(pre_whitener, data)

        elif self.noise_cov is None:  # use standardization as whitener
            std_chan = np.std(data, axis=1) ** -1
            pre_whitener = np.array([std_chan]).T
            data *= pre_whitener
        else:
            raise ValueError('This is not a valid valur for noise_cov')

        return data, pre_whitener

    def _fit_data(self, data, sort_method):
        """Helper function"""
        self.sources = self._fast_ica.fit_transform(data.T).T
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.sort_sources(sort_method=sort_method)

    def _denoise(self, bads):
        """Helper function"""
        if self.sorted_by != 'unsorted':
            sources, mixing, ids = self.sort_sources(sort_method='back',
                                                     inplace=False)
        else:
            sources = self.sources.copy()
            mixing = self.mixing.copy()
            ids = self.source_ids.copy()

        pre_whitener = self.pre_whitener
        if self.noise_cov is None:  # revert standardization
            pre_whitener **= -1
            mixing *= pre_whitener.T
        else:
            mixing = np.dot(mixing, linalg.pinv(pre_whitener))

        if bads is not None:
            ids = ids.tolist()
            bads_idx = [ids.index(bad) for bad in bads]
            sources[bads_idx, :] = 0.

        out = np.dot(sources.T, mixing).T

        return out
