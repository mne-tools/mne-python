# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from copy import deepcopy
import inspect
import warnings

import numpy as np
from scipy import stats
from scipy import linalg

from ..cov import compute_whitener
from ..fiff import pick_types


class ICA(object):
    """M/EEG signal decomposition using Independant Component Analysis (ICA)

    This object can be used to estimate ICA components and then
    remove some from Raw or Epochs for data exploration or artifact
    correction.

    Parameters
    ----------
    noise_cov : None | instance of mne.cov.Covariance
        Noise covariance used for whitening. If None, channels are just
        z-scored.
    n_components : int
        Number of components to be extracted. If None, no dimensionality
        reduction will be applied.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results.
    algorithm : {'parallel', 'deflation'}
        Apply parallel or deflational algorithm for FastICA
    fun : string or function, optional. Default: 'logcosh'
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point.
    fun_args: dictionary, optional
        Arguments to send to the functional form.
        If empty and if fun='logcosh', fun_args will take value
        {'alpha' : 1.0}

    Attributes
    ----------
    pre_whitener : instance of np.float | instance of mne.cov.Covariance
        Whitener used for preprocessing.
    sorted_by : str
        Flag informing about the active.
    last_fit : str
        Flag informing about which type was last fit.
    ch_names : list-like
        Channel names resulting from initial picking.
    """
    def __init__(self, noise_cov=None, n_components=None, random_state=None,
                 algorithm='parallel', fun='logcosh', fun_args=None):
        try:
            from sklearn.decomposition import FastICA  # to avoid strong dep.
        except ImportError:
            raise Exception('the scikit-learn package is missing and '
                            'required for ICA')
        self.noise_cov = noise_cov

        # sklearn < 0.11 does not support random_state argument for FastICA
        kwargs = {'algorithm': algorithm, 'fun': fun, 'fun_args': fun_args}

        if random_state is not None:
            aspec = inspect.getargspec(FastICA.__init__)
            if 'random_state' not in aspec.args:
                warnings.warn('random_state argument ignored, update '
                              'scikit-learn to version 0.11 or newer')
            else:
                kwargs['random_state'] = random_state

        self._fast_ica = FastICA(n_components, **kwargs)

        self.n_components = n_components
        self.last_fit = 'unfitted'
        self.sorted_by = 'unsorted'
        self.ch_names = None
        self.mixing = None

    def __repr__(self):
        out = 'ICA '
        if self.last_fit == 'unfitted':
            msg = '(no decomposition, '
        elif self.last_fit == 'raw':
            msg = '(raw data decomposition, '
        else:
            msg = '(epochs decomposition, '

        out += (msg + '%s components' % str(self.n_components) if
                self.n_components else 'no dimension reduction')

        if self.sorted_by == 'unsorted':
            sorted_by = self.sorted_by
        else:
            sorted_by = 'sorted by %s' % self.sorted_by
        out += ', %s)' % sorted_by

        return out

    def decompose_raw(self, raw, picks=None, start=None, stop=None):
        """Run the ica decomposition on raw data

        Parameters
        ----------
        raw : instance of mne.fiff.Raw
            Raw measurments to be decomposed.
        picks : array-like
            Channels to be included. This selecetion remains throught the
            initialized ICA session. If None only good data channels are used.
        start : int
            first sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int
            First sample to not include. If omitted, data is included to the
            end.

        Returns
        -------
        self : instance of ICA
            Returns the modified instance.
        """
        print ('Computing signal decomposition on raw data. '
               'Please be patient, this may take some time')

        if picks is None:  # just use good data channels
            picks = pick_types(raw.info, meg=True, eeg=True,
                               exclude=raw.info['bads'])

        self.ch_names = [raw.ch_names[k] for k in picks]
        if self.n_components is not None:
            self._sort_idx = np.arange(self.n_components)
        else:
            self._sort_idx = np.arange(len(picks))

        data, self.pre_whitener = self._get_raw_data(raw, picks, start, stop)

        self._fast_ica.fit(data.T)
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.last_fit = 'raw'
        return self

    def decompose_epochs(self, epochs, picks=None):
        """Run the ica decomposition on epochs

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. The ICA is estimated on the concatenated epochs.
        picks : array-like
            Channels to be included relative to the channels already picked on
            epochs-initialization. This selecetion remains throught the
            initialized ICA session.

        Returns
        -------
        self : instance of ICA
            Returns the modified instance.
        """
        print ('Computing signal decomposition on epochs. '
               'Please be patient, this may take some time')

        if picks is None:  # just use epochs good data channels
            picks = pick_types(epochs.info, meg=True, eeg=True,
                               exclude=epochs.info['bads'])

        self.ch_names = [epochs.ch_names[k] for k in picks]

        if self.n_components is not None:
            self._sort_idx = np.arange(self.n_components)
        else:
            self._sort_idx = np.arange(len(picks))

        data, self.pre_whitener = self._get_epochs_data(epochs, picks)
        self._fast_ica.fit(data.T)
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.last_fit = 'epochs'
        return self

    def get_sources_raw(self, raw, start=None, stop=None,
                        sort_func=stats.skew):
        """Estimate raw sources given the unmixing matrix

        Parameters
        ----------
        raw : instance of Raw
            Raw object to draw sources from
        start : int
            First sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int
            First sample to not include.
            If omitted, data is included to the end.
        sort_func : function
            Function used for sorting the sources. It should take an
            array and an axis argument.

        Returns
        -------
        sources : array, shape = (n_components, n_times)
            The ICA sources time series.
        """
        if self.mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        # this depends on the previous fit so I removed the arg
        picks = [raw.ch_names.index(k) for k in self.ch_names]
        data, _ = self._get_raw_data(raw, picks, start, stop)
        raw_sources = self._fast_ica.transform(data.T).T
        return self.sort_sources(raw_sources, sort_func=sort_func)

    def get_sources_epochs(self, epochs, sort_func=stats.skew):
        """Estimate epochs sources given the unmixing matrix

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from
        sort_func : function
            function used for sorting the sources. It should take an
            array and an axis argument.

        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch
        """
        if self.mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        picks = epochs.picks
        data, _ = self._get_epochs_data(epochs, picks)
        sources = self._fast_ica.transform(data.T).T
        sources = self.sort_sources(sources, sort_func=sort_func)
        epochs_sources = np.array(np.split(sources, len(epochs.events), 1))
        return epochs_sources

    def pick_sources_raw(self, raw, include=None, exclude=None, start=None,
                         stop=None, copy=True):
        """Recompose raw data including or excluding some sources

        Parameters
        ----------
        raw : instance of Raw
            Raw object to pick to remove ica components from.
        include : list-like | None
            The source indices to use. If None all are used.
        exclude : list-like | None
            The source indices to remove. If None  all are used.
        start : int | None
            The first time index to include.
        stop : int | None
            The first time index to exclude.
        copy: bool
            modify raw instance in place or return modified copy.

        Returns
        -------
        raw : instance of Raw
            raw instance with selected ica components removed
        """
        if not raw._preloaded:
            raise ValueError('raw data should be preloaded to have this '
                             'working. Please read raw data with '
                             'preload=True.')

        if self.last_fit != 'raw':
            raise ValueError('Currently no raw data fitted.'
                             'Please fit raw data first.')

        if self.sorted_by == 'unsorted':
            raise ValueError('Currently no sources reconstructed.'
                             'Please inspect sources first.')

        print '    ... restoring signals from selected sources'
        sources = self.get_sources_raw(raw, start=start, stop=stop,
                                       sort_func=self.sorted_by)
        recomposed = self._pick_sources(sources, include, exclude)

        if copy is True:
            raw = raw.copy()

        picks = [raw.ch_names.index(k) for k in self.ch_names]
        raw[picks, start:stop] = recomposed
        return raw

    def pick_sources_epochs(self, epochs, include=None, exclude=None,
                            copy=True):
        """Recompose epochs

        Parameters
        ----------
        epochs : instance of Epochs
            epochs object to pick to remove ica components from
        include : list-like | None
            The source indices to use. If None all are used.
        exclude : list-like | None
            The source indices to remove. If None  all are used.
        copy : bool
            Modify Epochs instance in place or return modified copy.

        Returns
        -------
        epochs : instance of Epochs
            Epochs with selected ica components removed.
        """
        if self.sorted_by == 'unsorted':
            raise ValueError('Currently no sources reconstructed.'
                             'Please inspect sources first.')

        sources = self.get_sources_epochs(epochs, sort_func=self.sorted_by)

        if copy is True:
            epochs = epochs.copy()
        # put sources-dimension first for selection
        recomposed = self._pick_sources(sources.swapaxes(0, 1),
                                        include, exclude)
        # restore epochs, channels, tsl order
        epochs._data = recomposed.swapaxes(0, 1)
        epochs.preload = True

        return epochs

    def sort_sources(self, sources, sort_func=stats.skew):
        """Sort sources accoroding to criteria such as skewness or kurtosis

        Parameters
        ----------
        sources : ndarray
            Previously reconstructed sources
        sort_func : function
            Function used for sorting the sources. It should take an
            array and an axis argument.

        Returns
        -------
        sorted_sources: ndarray
            The reorderd sources.
        """
        if sort_func is None:  # return sources
            return sources

        # select the appropriate dimension depending on input array
        sdim = 1 if sources.ndim > 2 else 0

        if self.n_components is not None:
            if sources.shape[sdim] != self.n_components:
                raise ValueError('Sources have to match the number'
                                 ' of components')

        if self.last_fit is 'unfitted':
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        sort_args = np.argsort(sort_func(sources, 1 + sdim))
        if sdim:
            sort_args = sort_args[0]
        if sort_func not in (self.sorted_by,):
            self._sort_idx = self._sort_idx[sort_args]
            print '    Sources reordered by %s' % sort_func

        self.sorted_by = sort_func

        return sources[:, sort_args] if sdim else sources[sort_args]

    def _pre_whiten(self, data, info, picks):
        """Helper function"""
        if self.noise_cov is None:  # use standardization as whitener
            pre_whitener = np.std(data) ** -1
            data *= pre_whitener
        else:  # pick cov
            ncov = deepcopy(self.noise_cov)
            if ncov.ch_names != self.ch_names:
                ncov['data'] = ncov.data[picks][:, picks]
            assert data.shape[0] == ncov.data.shape[0]
            pre_whitener, _ = compute_whitener(ncov, info, picks)
            data = np.dot(pre_whitener, data)

        return data, pre_whitener

    def _get_raw_data(self, raw, picks, start, stop):
        """Helper function"""
        return self._pre_whiten(raw[picks, start:stop][0], raw.info, picks)

    def _get_epochs_data(self, epochs, picks):
        """Helper function"""
        return self._pre_whiten(np.hstack(epochs.get_data()), epochs.info,
                                picks)

    def _pick_sources(self, sources, include, exclude):
        """Helper function"""
        mixing = self.mixing.copy()
        pre_whitener = self.pre_whitener.copy()
        if self.noise_cov is None:  # revert standardization
            pre_whitener **= -1
            mixing *= pre_whitener
        else:
            mixing = np.dot(mixing, linalg.pinv(pre_whitener))

        if include not in (None, []):
            mute = [i for i in xrange(len(sources)) if i not in include]
            sources[mute, :] = 0.  # include via exclusion
        elif exclude not in (None, []):
            sources[exclude, :] = 0.  # just exclude

        # restore initial sort, then mix back the souces
        restore_idx = np.argsort(self._sort_idx.copy())
        out = np.dot(sources[restore_idx].T, mixing).T

        return out
