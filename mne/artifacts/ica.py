# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from inspect import getargspec
import numpy as np
from scipy.stats import kurtosis, skew
from scipy import linalg
from ..cov import compute_whitener
from copy import deepcopy


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
    sorted_by : str
        flag informing about the active
    last_fit : str
        flag informing about which type was last fit.
    ch_names : list-like
        ch_names resulting from initial picking
    """
    def __init__(self, noise_cov=None, n_components=None, random_state=None):
        from sklearn.decomposition import FastICA
        self.noise_cov = noise_cov
        self._fast_ica = FastICA(n_components, random_state=random_state)
        self.n_components = n_components
        self.last_fit = 'unfitted'
        self.sorted_by = 'unsorted'
        self.ch_names = None

    def __repr__(self):
        out = 'ICA '
        if self.last_fit == 'unfitted':
            msg = '(no decomposition, '
        elif self.last_fit == 'raw':
            msg = '(raw data decomposition, '
        else:
            msg = '(epochs decomposition, '

        out += msg + '%i components' % self.n_components

        if self.sorted_by == 'unsorted':
            sorted_by = self.sorted_by
        else:
            sorted_by = 'sorted by %s' % self.sorted_by
        out += ', %s)' % sorted_by

        return out

    def decompose_raw(self, raw, picks, start=None, stop=None):
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

        Returns
        -------
        self : instance of ICA
            returns the instance for chaining
        """
        print ('\nComputing signal decomposition on raw data.'
               '\n    Please be patient. This may take some time')

        self.ch_names = np.array(raw.ch_names)[picks].tolist()
        self._sort_idx = (np.arange(self.n_components) if self.n_components
                           is not None else np.arange(picks.shape[0]))

        data, self.pre_whitener = self._get_raw_data(raw, picks, start, stop)

        self._fast_ica.fit(data.T)
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.last_fit = 'raw'

        return self

    def decompose_epochs(self, epochs, picks=None):
        """Run the ica decomposition for epochs

        Paramerters
        -----------
        epochs : instance of Epochs
            The epochs. The ICA is estimated on the concatenated epochs.
        picks : array-like
            channels to be included.

        Returns
        -------
        self : instance of ICA
            returns the instance for chaining
        """
        if picks is None:
            picks = epochs.picks
        self.ch_names = np.array(epochs.ch_names)[picks].tolist()
        self._sort_idx = (np.arange(self.n_components) if self.n_components
                           is not None else np.arange(self.picks.shape[0]))

        print ('\nComputing signal decomposition on epochs.'
               '\n    Please be patient. This may take some time')

        data, self.pre_whitener = self._get_epochs_data(epochs, picks)
        self._fast_ica.fit(data.T)
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.last_fit = 'epochs'

        return self

    def get_sources_raw(self, raw, picks='previous', start=None, stop=None,
                        sort_method='skew'):
        """ Uncover raw sources
        Paramerters
        -----------
        raw : instance of Raw
            Raw object to draw sources from
        start : integer
            starting time slice
        stop : integer
            final time slice
        picks : array-like
            channels to be included
        sort_method : str | function
            method used for sorting the sources. Options are 'skew',
            'kurtosis', 'unsorted' or a custom function that takes an
            array and an axis argument.
        """
        if self.last_fit is 'unfitted':
            print ('No fit availble. Please first fit ica decomposition.')
            return

        picks = self._check_picks(raw, picks)
        data, _ = self._get_raw_data(raw, picks, start, stop)
        raw_sources = self._fast_ica.transform(data.T).T

        return self.sort_sources(raw_sources, sort_method=sort_method)

    def get_sources_epochs(self, epochs, picks=None, sort_method='skew'):
        """ Uncover raw sources
        Paramerters
        -----------
        raw : instance of Raw
            Raw object to draw sources from
        picks : array-like
            channels to be included
        sort_method : str | function
            method used for sorting the sources. Options are 'skew',
            'kurtosis', 'unsorted' or a custom function that takes an
            array and an axis argument.

        Returns
        -------
        epochs_sources : ndarray
            epochs x sources x timeslices array

        """
        if self.last_fit is 'unfitted':
            print ('No fit availble. Please first fit ica decomposition.')
            return

        picks = self._check_picks(epochs, picks)
        data, _ = self._get_epochs_data(epochs, picks)
        sources = self._fast_ica.transform(data.T).T
        sources = self.sort_sources(sources, sort_method=sort_method)
        epochs_sources = np.array(np.split(sources, len(epochs.events), 1))

        return epochs_sources

    def pick_sources_raw(self, raw, exclude=[], include=None, start=None, stop=None,
                         copy=True, sort_method='skew'):
        """Recompose raw data

        Paramerters
        -----------
        raw : instance of Raw
            raw object to pick to remove ica components from
        exclude : list-like
            Indices for transient component deselection
        include : array-like
            use channel subset as specified
        copy: boolean
            modify raw instance in place or return modified copy
        sort_method : str | function
            method used for sorting the sources. Options are 'skew',
            'kurtosis', 'unsorted' or a custom function that takes an
            array and an axis argument.

        Returns
        -------
        raw : instance of Raw
            raw instance with selected ica components removed
        """
        if self.last_fit != 'raw':
            raise ValueError('Currently no raw data fitted.'
                             'Please fit raw data first.')

        include = self._check_picks(raw, include)
        if sort_method not in (None, self.sorted_by):
            print ('\n    Sort method demanded is different from last sort'
                   '\n    ... reordering the sources accorodingly')
            sort_method = self.sorted_by

        sources = self.get_sources_raw(raw, picks=include, start=start,
                                       stop=stop, sort_method=sort_method)
        recomposed = self._pick_sources(sources, exclude,
                                        include)
        if not raw._preloaded:
            raw._preload_data(True)
            raw._preloaded = True

        if copy is True:
            raw = raw.copy()

        raw[include, start:stop] = recomposed

        return raw

    def pick_sources_epochs(self, epochs, exclude=[], include=None, copy=True,
                            sort_method='skew'):
        """Recompose epochs

        Paramerters
        -----------
        epochs : instance of Epochs
            epochs object to pick to remove ica components from
        exclude : list-like
            Indices for transient component deselection
        copy : boolean
            Either return denoised data as nd array or newly instantiated
            Epochs object.
        sort_method : str | function
            method used for sorting the sources. Options are 'skew',
            'kurtosis', 'unsorted' or a custom function that takes an
            array and an axis argument.

        Returns
        -------
        epochs : instance of Epochs
            epochs with selected ica components removed
        """

        include = self._check_picks(epochs, include)
        if sort_method not in (None, self.sorted_by):
            print ('\n    Sort method demanded is different from last sort'
                   '\n    ... reordering the sources accorodingly')
            sort_method = self.sorted_by

        sources = self.get_sources_epochs(epochs)
        if copy is True:
            epochs = epochs.copy()

        recomposed = self._pick_sources(sources.swapaxes(0, 1), exclude, include)
        epochs._data = recomposed.swapaxes(0, 1)
        epochs.preload = True

        return epochs

    def sort_sources(self, sources, sort_method='skew'):
        """Sort sources accoroding to criteria such as skewness or kurtosis

        Paramerters
        -----------
        sources : str
            string for selecting the sources
        sort_method : str | function
            method used for sorting the sources. Options are 'skew',
            'kurtosis', 'unsorted' or a custom function that takes an
            array and an axis argument.
        """
        if sources.shape[0] != self.n_components:
            raise ValueError('Sources have to match the number of components')

        if self.last_fit is 'unfitted':
            print ('No fit availble. Please first fit ica decomposition.')
            return

        if sort_method == 'skew':
            sort_func = skew
        elif sort_method == 'kurtosis':
            sort_func = kurtosis
        elif sort_method == 'unsorted':
            sort_func = lambda x, y: self._sort_idx
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

        sort_args = np.argsort(sort_func(sources, 1))
        self._sort_idx = self._sort_idx[sort_args]
        self.sorted_by = (sort_func.__name__ if not callable(sort_method)
                          else sort_method)
        print '\n    sources reordered by %s' % self.sorted_by

        return sources[sort_args]

    def _pre_whiten(self, data, info, picks):
        """Helper function"""
        if hasattr(self.noise_cov, 'data'):
            # pick cov
            ncov = deepcopy(self.noise_cov)
            if not ncov.ch_names == self.ch_names:
                ncov['data'] = ncov.data[picks][:, picks]
            # check whether cov matches channels
            assert data.shape[0] == ncov.data.shape[0]

            pre_whitener, _ = compute_whitener(ncov, info,
                                               picks)
            data = np.dot(pre_whitener, data)

        elif self.noise_cov is None:  # use standardization as whitener
            std_chan = np.std(data, axis=1) ** -1
            pre_whitener = np.array([std_chan]).T
            data *= pre_whitener
        else:
            raise ValueError('This is not a valid value for noise_cov')

        return data, pre_whitener

    def _get_raw_data(self, raw, picks, start, stop):
        """Helper function"""
        start = 0 if start is None else start
        stop = (raw.last_samp - raw.first_samp) + 1 if stop is None else stop
        return self._pre_whiten(raw[picks, start:stop][0], raw.info, picks)

    def _get_epochs_data(self, epochs, picks):
        """Helper function"""
        data = epochs._data if epochs.preload else epochs.get_data()
        data, pre_whitener = self._pre_whiten(np.hstack(data), epochs.info,
                                              picks)
        return data, pre_whitener

    def _pick_sources(self, sources, exclude, include):
        """Helper function"""
        mixing = self.mixing.copy()
        pre_whitener = self.pre_whitener.copy()
        if not hasattr(self.noise_cov, 'data'):  # revert standardization
            pre_whitener **= -1
            mixing *= pre_whitener.T
        else:
            mixing = np.dot(mixing, linalg.pinv(pre_whitener))

        if exclude not in (None, []):
            sources[exclude, :] = 0.

        out = np.dot(sources.T, mixing).T

        return out

    def _check_picks(self, pickable, picks):
        """Helper function"""
        out = None
        if picks is None:
            intersect = np.intersect1d(np.array(pickable.ch_names),
                                       self.ch_names)
            out = np.where(intersect)[0]
        elif not np.in1d(pickable.ch_names, self.ch_names)[picks].all():
            raise ValueError('Channel picks have to match '
                             'the previous fit.')
        else:
            out = picks

        return out
