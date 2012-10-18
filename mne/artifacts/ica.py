# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import stats
from scipy import linalg
from ..cov import compute_whitener


class ICA(object):
    """MEG signal decomposition and denoising workflow

    Paramerters
    -----------
    noise_cov : instance of mne.cov.Covariance
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
        whitener used for preprocessing
    sorted_by : str
        flag informing about the active
    last_fit : str
        flag informing about which type was last fit.
    ch_names : list-like
        ch_names resulting from initial picking
    """
    def __init__(self, noise_cov=None, n_components=None, random_state=None):
        from sklearn.decomposition import FastICA  # to avoid strong dependency
        self.noise_cov = noise_cov
        self._fast_ica = FastICA(n_components, random_state=random_state)
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

        out += msg + '%i components' % self.n_components

        if self.sorted_by == 'unsorted':
            sorted_by = self.sorted_by
        else:
            sorted_by = 'sorted by %s' % self.sorted_by
        out += ', %s)' % sorted_by

        return out

    def decompose_raw(self, raw, picks, start=None, stop=None):
        """Run the ica decomposition on raw data

        Paramerters
        -----------
        raw : instance of mne.fiff.Raw
            raw measurments to be decomposed
        start : integer
            starting time index
        stop : integer
            first time index to ignore.
        picks : array-like
            channels to be included.

        Returns
        -------
        self : instance of ICA
            returns the instance for chaining
        """
        print ('Computing signal decomposition on raw data. '
               'Please be patient, this may take some time')

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
        print ('Computing signal decomposition on epochs. '
               'Please be patient, this may take some time')

        if picks is None:
            picks = epochs.picks

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

    def get_sources_raw(self, raw, picks=None, start=None, stop=None,
                        sort_func=stats.skew):
        """Estimate raw sources given the unmixing matrix

        Paramerters
        -----------
        raw : instance of Raw
            Raw object to draw sources from
        start : integer
            starting time slice
        stop : integer
            final time slice
        picks : array-like | None
            channels to be included. If None the channels used during
            ICA estimation will be used.
        sort_func : function
            function used for sorting the sources. It should take an
            array and an axis argument.
        """
        if self.mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        picks = self._check_picks(raw, picks)
        data, _ = self._get_raw_data(raw, picks, start, stop)
        raw_sources = self._fast_ica.transform(data.T).T
        return self.sort_sources(raw_sources, sort_func=sort_func)

    def get_sources_epochs(self, epochs, picks=None, sort_func=stats.skew):
        """Estimate epochs sources given the unmixing matrix

        Paramerters
        -----------
        raw : instance of Raw
            Raw object to draw sources from
        picks : array-like
            channels to be included
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

        picks = self._check_picks(epochs, picks)
        data, _ = self._get_epochs_data(epochs, picks)
        sources = self._fast_ica.transform(data.T).T
        sources = self.sort_sources(sources, sort_func=sort_func)
        epochs_sources = np.array(np.split(sources, len(epochs.events), 1))
        return epochs_sources

    def pick_sources_raw(self, raw, include=None, exclude=[], start=None,
                         stop=None, copy=True, sort_func=stats.skew):
        """Recompose raw data including or excluding some sources

        Paramerters
        -----------
        raw : instance of Raw
            raw object to pick to remove ica components from
        include : array-like | None
            The source indices to use. If None all are used.
        exclude : list-like
            The source indices to remove.
        start : int | None
            The first time index to include
        stop : int | None
            The first time index to exclude
        copy: bool
            modify raw instance in place or return modified copy
        sort_func : function
            function used for sorting the sources. It should take an
            array and an axis argument.

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

        include = self._check_picks(raw, include)
        if sort_func not in (None, self.sorted_by):
            print ('\n    Sort method demanded is different from last sort'
                   '\n    ... reordering the sources accorodingly')
            sort_func = self.sorted_by

        print '    ... restoring signals from selected sources'
        sources = self.get_sources_raw(raw, picks=include, start=start,
                                       stop=stop, sort_func=sort_func)
        recomposed = self._pick_sources(sources, include, exclude)

        if copy is True:
            raw = raw.copy()

        raw[include, start:stop] = recomposed

        return raw

    def pick_sources_epochs(self, epochs, include=None, exclude=[], copy=True,
                            sort_func=stats.skew):
        """Recompose epochs

        Paramerters
        -----------
        epochs : instance of Epochs
            epochs object to pick to remove ica components from
        include : array-like | None
            The source indices to use. If None all are used.
        exclude : list-like
            The source indices to remove.
        copy : bool
            Modify Epochs instance in place or return modified copy
        sort_func : function | str
            function used for sorting the sources. It should take an
            array and an axis argument.

        Returns
        -------
        epochs : instance of Epochs
            epochs with selected ica components removed
        """
        include = self._check_picks(epochs, include)
        if sort_func not in (None, self.sorted_by):
            print ('\n    Sort method demanded is different from last sort'
                   '\n    ... reordering the sources accorodingly')
            sort_func = self.sorted_by

        print '    ... restoring signals from selected sources'
        sources = self.get_sources_epochs(epochs, sort_func=sort_func)

        if copy is True:
            epochs = epochs.copy()

        recomposed = self._pick_sources(sources.swapaxes(0, 1),
                                        include, exclude)
        epochs._data = recomposed.swapaxes(0, 1)
        epochs.preload = True

        return epochs

    def sort_sources(self, sources, sort_func=stats.skew):
        """Sort sources accoroding to criteria such as skewness or kurtosis

        Paramerters
        -----------
        sources : str
            string for selecting the sources
        sort_func : function
            function used for sorting the sources. It should take an
            array and an axis argument.
        """
        sdim = 1 if sources.ndim > 2 else 0
        if sources.shape[sdim] != self.n_components:
            raise ValueError('Sources have to match the number of components')

        if self.last_fit is 'unfitted':
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        sort_args = np.argsort(sort_func(sources, 1 + sdim))
        if sdim:
            sort_args = sort_args[0]
        self._sort_idx = self._sort_idx[sort_args]
        self.sorted_by = sort_func
        print '    Sources reordered by %s' % self.sorted_by

        return sources[:, sort_args] if sdim else sources[sort_args]

    def _pre_whiten(self, data, info, picks):
        """Helper function"""
        if self.noise_cov is None:  # use standardization as whitener
            std_chan = np.std(data, axis=1) ** -1
            pre_whitener = np.array([std_chan]).T
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
        start = 0 if start is None else start
        stop = (raw.last_samp - raw.first_samp) + 1 if stop is None else stop
        return self._pre_whiten(raw[picks, start:stop][0], raw.info, picks)

    def _get_epochs_data(self, epochs, picks):
        """Helper function"""
        data = epochs._data if epochs.preload else epochs.get_data()
        data, pre_whitener = self._pre_whiten(np.hstack(data), epochs.info,
                                              picks)
        return data, pre_whitener

    def _pick_sources(self, sources, include, exclude):
        """Helper function"""
        mixing = self.mixing.copy()
        pre_whitener = self.pre_whitener.copy()
        if self.noise_cov is None:  # revert standardization
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
        if picks is None:
            intersect = np.intersect1d(np.array(pickable.ch_names),
                                       self.ch_names)
            out = np.where(intersect)[0]
        else:
            out = picks

        return out
