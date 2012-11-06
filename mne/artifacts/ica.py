# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from copy import deepcopy
import inspect
import warnings
from inspect import getargspec, isfunction

import logging
logger = logging.getLogger('mne')

import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy import linalg

from .ecg import qrs_detector
from .eog import _find_eog_events

from ..cov import compute_whitener
from ..fiff import pick_types, pick_channels
from ..fiff.constants import Bunch
from ..viz import plot_ica_panel
from .. import verbose


def _make_xy_sfunc(func, ndim_output=False):
    """Helper Function"""
    if ndim_output:
        sfunc = lambda x, y: np.array([func(a, y.ravel()) for a in x])[:, 0]
    else:
        sfunc = lambda x, y: np.array([func(a, y.ravel()) for a in x])
    sfunc.__name__ = '.'.join(['score_func', func.__module__, func.__name__])
    sfunc.__doc__ = func.__doc__
    return sfunc

# makes score funcs attr accessible for users
score_funcs = Bunch()

xy_arg_dist_funcs = [(n, f) for n, f in vars(distance).items() if isfunction(f)
                     and not n.startswith('_')]

xy_arg_stats_funcs = [(n, f) for n, f in vars(stats).items() if isfunction(f)
                      and not n.startswith('_')]

score_funcs.update(dict((n, _make_xy_sfunc(f)) for n, f in xy_arg_dist_funcs
                   if getargspec(f).args == ['u', 'v']))

score_funcs.update(dict((n, _make_xy_sfunc(f, ndim_output=True))
                   for n, f in xy_arg_stats_funcs
                   if getargspec(f).args == ['x', 'y']))


__all__ = ['ICA', 'ica_find_ecg_events', 'ica_find_eog_events', 'score_funcs']


class ICA(object):
    """M/EEG signal decomposition using Independent Component Analysis (ICA)

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes
    ----------
    pre_whitener : instance of np.float | instance of mne.cov.Covariance
        Whitener used for preprocessing.
    last_fit : str
        Flag informing about which type was last fit.
    ch_names : list-like
        Channel names resulting from initial picking.
    index : ndarray
        Integer array representing the sources. This is usefull for different
        kinds of indexing and selection operations.
    verbose : bool, str, int, or None
        See above.
    """
    @verbose
    def __init__(self, noise_cov=None, n_components=None, random_state=None,
                 algorithm='parallel', fun='logcosh', fun_args=None,
                 verbose=None):
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
        self.index = np.arange(n_components) if n_components else None
        self.last_fit = 'unfitted'
        self.ch_names = None
        self.mixing = None
        self.verbose = verbose

    def __repr__(self):
        out = 'ICA '
        if self.last_fit == 'unfitted':
            msg = '(no'
        elif self.last_fit == 'raw':
            msg = '(raw data'
        else:
            msg = '(epochs'
        msg += ' decomposition, '

        out += msg + ('%s components' % str(self.n_components) if
               self.n_components else 'no dimension reduction') + ')'

        return out

    @verbose
    def decompose_raw(self, raw, picks=None, start=None, stop=None,
                      verbose=None):
        """Run the ICA decomposition on raw data

        Parameters
        ----------
        raw : instance of mne.fiff.Raw
            Raw measurements to be decomposed.
        picks : array-like
            Channels to be included. This selection remains throughout the
            initialized ICA session. If None only good data channels are used.
        start : int
            First sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int
            First sample to not include. If omitted, data is included to the
            end.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Returns
        -------
        self : instance of ICA
            Returns the modified instance.
        """
        logger.info('Computing signal decomposition on raw data. '
                    'Please be patient, this may take some time')

        if picks is None:  # just use good data channels
            picks = pick_types(raw.info, meg=True, eeg=True,
                               exclude=raw.info['bads'])

        self.ch_names = [raw.ch_names[k] for k in picks]

        if self.n_components is not None:
            self._sort_idx = np.arange(self.n_components)
        else:
            self._sort_idx = np.arange(len(picks))

        data, self.pre_whitener = self._pre_whiten(raw[picks, start:stop][0],
                                                   raw.info, picks)

        self._fast_ica.fit(data.T)
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.last_fit = 'raw'
        return self

    @verbose
    def decompose_epochs(self, epochs, picks=None, verbose=None):
        """Run the ICA decomposition on epochs

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. The ICA is estimated on the concatenated epochs.
        picks : array-like
            Channels to be included relative to the channels already picked on
            epochs-initialization. This selection remains throughout the
            initialized ICA session.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Returns
        -------
        self : instance of ICA
            Returns the modified instance.
        """
        logger.info('Computing signal decomposition on epochs. '
                    'Please be patient, this may take some time')

        if picks is None:  # just use epochs good data channels and avoid
            picks = pick_types(epochs.info, include=epochs.ch_names,  # double
                               exclude=epochs.info['bads'])  # picking

        meeg_picks = pick_types(epochs.info, meg=True, eeg=True,
                                exclude=epochs.info['bads'])

        picks = np.intersect1d(meeg_picks, picks)

        self.ch_names = [epochs.ch_names[k] for k in picks]

        if self.n_components is not None:
            self._sort_idx = np.arange(self.n_components)
        else:
            self._sort_idx = np.arange(len(picks))

        data, self.pre_whitener = self._pre_whiten(
                                np.hstack(epochs.get_data()[:, picks]),
                                epochs.info, picks)
        self._fast_ica.fit(data.T)
        self.mixing = self._fast_ica.get_mixing_matrix().T
        self.last_fit = 'epochs'
        return self

    def get_sources_raw(self, raw, start=None, stop=None):
        """Estimate raw sources given the unmixing matrix

        Parameters
        ----------
        raw : instance of Raw
            Raw object to draw sources from.
        start : int
            First sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int
            First sample to not include.
            If omitted, data is included to the end.

        Returns
        -------
        sources : array, shape = (n_components, n_times)
            The ICA sources time series.
        """
        if self.mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        # this depends on the previous fit so no pick arg
        picks = [raw.ch_names.index(k) for k in self.ch_names]
        data, _ = self._pre_whiten(raw[picks, start:stop][0], raw.info, picks)
        raw_sources = self._fast_ica.transform(data.T).T

        return raw_sources

    def get_sources_epochs(self, epochs, concatenate=False):
        """Estimate epochs sources given the unmixing matrix

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        concatenate : bool
            If true, epochs and time slices will be concatenated.

        Returns
        -------
        epochs_sources : ndarray of shape (n_epochs, n_sources, n_times)
            The sources for each epoch
        """
        if self.mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        picks = pick_types(epochs.info, include=self.ch_names,
                               exclude=epochs.info['bads'])

        data, _ = self._pre_whiten(np.hstack(epochs.get_data()[:, picks]),
                                   epochs.info, picks)
        sources = self._fast_ica.transform(data.T).T
        epochs_sources = np.array(np.split(sources, len(epochs.events), 1))

        return epochs_sources if not concatenate else np.hstack(epochs_sources)

    def plot_sources_raw(self, raw, order=None, start=None, stop=None,
                         n_components=None, source_idx=None, ncol=3, nrow=10,
                         show=True):
        """Create panel plots of ICA sources. Wrapper around viz.plot_ica_panel

        Parameters
        ----------
        raw : instance of mne.fiff.Raw
            Raw object to plot the sources from.
        order : ndarray | None.
            Index of length n_components. If None, plot will show the sources
            in the order as fitted.
            Example: arg_sort = np.argsort(np.var(sources)).
        start : int
            X-axis start index. If None from the beginning.
        stop : int
            X-axis stop index. If None to the end.
        n_components : int
            Number of components fitted.
        source_idx : array-like
            Indices for subsetting the sources.
        ncol : int
            Number of panel-columns.
        nrow : int
            Number of panel-rows.
        show : bool
            If True, plot will be shown, else just the figure is returned.

        Returns
        -------
        fig : instance of pyplot.Figure
        """

        sources = self.get_sources_raw(raw, start=start, stop=stop)

        if order is not None:
            if len(order) != sources.shape[0]:
                    raise ValueError('order and sources have to be of the '
                                     'same lenght.')
            else:
                sources = sources[order]

        fig = plot_ica_panel(sources, start=0 if start is not None else start,
                             stop=(stop - start) if stop is not None else stop,
                             n_components=n_components, source_idx=source_idx,
                             ncol=ncol, nrow=nrow)
        if show:
            import matplotlib.pylab as pl
            pl.show()

        return fig

    def plot_sources_epochs(self, epochs, epoch_idx=None, order=None,
                            start=None, stop=None, n_components=None,
                            source_idx=None, ncol=3, nrow=10, show=True):
        """Create panel plots of ICA sources. Wrapper around viz.plot_ica_panel

        Parameters
        ----------
        epochs : instance of mne.Epochs
            Epochs object to plot the sources from.
        epoch_idx : int
            Index to plot particular epoch.
        order : ndarray | None.
            Index of length n_components. If None, plot will show the sources
            in the order as fitted.
            Example: arg_sort = np.argsort(np.var(sources)).
        sources : ndarray
            Sources as drawn from self.get_sources.
        start : int
            X-axis start index. If None from the beginning.
        stop : int
            X-axis stop index. If None to the end.
        n_components : int
            Number of components fitted.
        source_idx : array-like
            Indices for subsetting the sources.
        ncol : int
            Number of panel-columns.
        nrow : int
            Number of panel-rows.
        show : bool
            If True, plot will be shown, else just the figure is returned.

        Returns
        -------
        fig : instance of pyplot.Figure
        """
        sources = self.get_sources_epochs(epochs, concatenate=True if epoch_idx
                                          is None else False)
        source_dim = 1 if sources.ndim > 2 else 0
        if order is not None:
            if len(order) != sources.shape[source_dim]:
                raise ValueError('order and sources have to be of the '
                                 'same lenght.')
            else:
                sources = (sources[:, order] if source_dim
                           else sources[order])

        fig = plot_ica_panel(sources[epoch_idx], start=start, stop=stop,
                             n_components=n_components, source_idx=source_idx,
                             ncol=ncol, nrow=nrow)
        if show:
            import matplotlib.pylab as pl
            pl.show()

        return fig

    def find_sources_raw(self, raw, target=None, score_func='pearsonr',
                         start=None, stop=None):
        """Find sources based on own distribution or based on similarity to
        other sources or between source and target.

        Parameters
        ----------
        raw : instance of Raw
            Raw object to draw sources from.
        target : array-like | ch_name | None
            Signal to which the sources shall be compared. It has to be of
            the same shape as the sources. If some string is supplied, a
            routine will try to find a matching channel. If None, a score
            function expecting only one input-array argument must be used,
            for instance, scipy.stats.skew (default).
        score_func : callable | str label
            Callable taking as arguments either two input arrays
            (e.g. pearson correlation) or one input
            array (e. g. skewness) and returns a float. For convenience the
            most common score_funcs are available via string labels: Currently,
            all distance metrics from scipy.spatial and all functions from
            scipy.stats taking compatible input arguments are supported. These
            function have been modified to support iteration over the rows of a
            2D array.
        start : int
            First sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int
            First sample to not include.
            If omitted, data is included to the end.
        scores : ndarray
            Scores for each source as returned from score_func.

        Returns
        -------
        scores : ndarray
            scores for each source as returned from score_func
        """
        # auto source drawing
        sources = self.get_sources_raw(raw=raw, start=start, stop=stop)

        # auto target selection
        if target is not None:
            if isinstance(target, str):
                pick = _get_target_ch(raw, target)
                target, _ = raw[pick, start:stop]
            if sources.shape[1] != target.shape[1]:
                raise ValueError('Source and targets do not have the same'
                                 'number of time slices.')
            target = target.ravel()

        return _find_sources(sources, target, score_func)

    def find_sources_epochs(self, epochs, target=None, score_func='pearsonr'):
        """Find sources based on relations between source and target

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to draw sources from.
        target : array-like | ch_name | None
            Signal to which the sources shall be compared. It has to be of
            the same shape as the sources. If some string is supplied, a
            routine will try to find a matching channel. If None, a score
            function expecting only one input-array argument must be used,
            for instance, scipy.stats.skew (default).
        score_func : callable | str label
            Callable taking as arguments either two input arrays
            (e.g. pearson correlation) or one input
            array (e. g. skewness) and returns a float. For convenience the
            most common score_funcs are available via string labels: Currently,
            all distance metrics from scipy.spatial and all functions from
            scipy.stats taking compatible input arguments are supported. These
            function have been modified to support iteration over the rows of a
            2D array.

        Returns
        -------
        scores : ndarray
            scores for each source as returned from score_func
        """
        sources = self.get_sources_epochs(epochs=epochs)
        # auto target selection
        if target is not None:
            if isinstance(target, str):
                pick = _get_target_ch(epochs, target)
                target = epochs.get_data()[:, pick]
            if sources.shape[2] != target.shape[2]:
                raise ValueError('Source and targets do not have the same'
                                 'number of time slices.')
            target = target.ravel()

        return _find_sources(np.hstack(sources), target, score_func)

    def pick_sources_raw(self, raw, include=None, exclude=None, start=None,
                         stop=None, copy=True):
        """Recompose raw data including or excluding some sources

        Parameters
        ----------
        raw : instance of Raw
            Raw object to pick to remove ICA components from.
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
            raw instance with selected ICA components removed
        """
        if not raw._preloaded:
            raise ValueError('raw data should be preloaded to have this '
                             'working. Please read raw data with '
                             'preload=True.')

        if self.last_fit != 'raw':
            raise ValueError('Currently no raw data fitted.'
                             'Please fit raw data first.')

        sources = self.get_sources_raw(raw, start=start, stop=stop)
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
            Epochs object to pick to remove ICA components from.
        include : list-like | None
            The source indices to use. If None all are used.
        exclude : list-like | None
            The source indices to remove. If None  all are used.
        copy : bool
            Modify Epochs instance in place or return modified copy.

        Returns
        -------
        epochs : instance of Epochs
            Epochs with selected ICA components removed.
        """
        if not epochs.preload:
            raise ValueError('raw data should be preloaded to have this '
                             'working. Please read raw data with '
                             'preload=True.')

        sources = self.get_sources_epochs(epochs)
        picks = pick_types(epochs.info, include=self.ch_names,
                               exclude=epochs.info['bads'])

        if copy is True:
            epochs = epochs.copy()

        # put sources-dimension first for selection
        recomposed = self._pick_sources(sources.swapaxes(0, 1),
                                        include, exclude)
        # restore epochs, channels, tsl order
        epochs._data[:, picks] = recomposed.swapaxes(0, 1)
        epochs.preload = True

        return epochs

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

        out = np.dot(sources.T, mixing).T

        return out


@verbose
def ica_find_ecg_events(raw, ecg_source, event_id=999,
                        tstart=0.0, l_freq=5, h_freq=35, qrs_threshold=0.6,
                        verbose=None):
    """Find ECG peaks from one selected ICA source

    Parameters
    ----------
    ecg_source : ndarray
        ICA source resembling ECG to find peaks from.
    event_id : int
        The index to assign to found events.
    raw : instance of Raw
        Raw object to draw sources from.
    start : int
        First sample to include (first is 0). If omitted, defaults to the
        first sample in data.
    stop : int
        First sample to not include.
        If omitted, data is included to the end.
    tstart : float
        Start detection after tstart seconds. Useful when beginning
        of run is noisy.
    l_freq : float
        Low pass frequency.
    h_freq : float
        High pass frequency.
    qrs_threshold : float
        Between 0 and 1. qrs detection threshold.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    ecg_events : array
        Events.
    ch_ECG : string
        Name of channel used.
    average_pulse : float.
        Estimated average pulse.
    """
    logger.info('Using ICA source to identify heart beats')

    # detecting QRS and generating event file
    ecg_events = qrs_detector(raw.info['sfreq'], ecg_source.ravel(),
                              tstart=tstart, thresh_value=qrs_threshold,
                              l_freq=l_freq, h_freq=h_freq)

    n_events = len(ecg_events)

    ecg_events = np.c_[ecg_events + raw.first_samp, np.zeros(n_events),
                       event_id * np.ones(n_events)]

    return ecg_events


@verbose
def ica_find_eog_events(raw, eog_source=None, event_id=998, l_freq=1,
                    h_freq=10, verbose=None):
    """Locate EOG artifacts

    Parameters
    ----------
    raw : instance of Raw
        The raw data.
    eog_source : ndarray
        ICA source resembling EOG to find peaks from.
    event_id : int
        The index to assign to found events.
    low_pass : float
        Low pass frequency.
    high_pass : float
        High pass frequency.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    eog_events : array
        Events
    """
    eog_events = _find_eog_events(eog_source[np.newaxis], event_id=event_id,
                                l_freq=l_freq, h_freq=h_freq,
                                sampling_rate=raw.info['sfreq'],
                                first_samp=raw.first_samp)
    return eog_events


def _get_target_ch(container, target):
    """Helper Function"""
    # auto target selection
    pick = pick_channels(container.ch_names, include=[target])
    if len(pick) == 0:
        raise ValueError('%s not in channel list (%s)' %
                        (target, container.ch_names))
    return pick


def _find_sources(sources, target, score_func):
    """Helper Function"""
    if isinstance(score_func, str):
        score_func = score_funcs.get(score_func, score_func)

    if not callable(score_func):
        raise ValueError('%s is not a valid score_func.' % score_func)

    scores = (score_func(sources, target) if target is not None
              else score_func(sources, 1))

    return scores
