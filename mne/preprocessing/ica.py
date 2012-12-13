# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from copy import deepcopy
import inspect
import warnings
from inspect import getargspec, isfunction

import os
import logging
logger = logging.getLogger('mne')

import numpy as np
from scipy import stats
from scipy.spatial import distance
from scipy import linalg

from .ecg import qrs_detector
from .eog import _find_eog_events

from ..cov import compute_whitener
from .. import Covariance
from ..fiff import pick_types, pick_channels
from ..fiff.write import write_double_matrix, write_string, \
                         write_name_list, start_block, end_block
from ..fiff.tree import dir_tree_find
from ..fiff.open import fiff_open
from ..fiff.tag import read_tag
from ..fiff.constants import Bunch, FIFF
from ..viz import plot_ica_panel
from .. import verbose
from ..fiff.write import start_file, end_file


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


__all__ = ['ICA', 'ica_find_ecg_events', 'ica_find_eog_events', 'score_funcs',
           'read_ica']


class ICA(object):
    """M/EEG signal decomposition using Independent Component Analysis (ICA)

    This object can be used to estimate ICA components and then
    remove some from Raw or Epochs for data exploration or artifact
    correction.

    Parameters
    ----------
    n_components : int | float | None
        The number of components used for ICA decomposition. If int, it must be
        smaller then max_n_components. If None, all PCA components will be
        used. If float between 0 and 1 components can will be selected by the
        cumulative percentage of explained variance.
    max_n_components : int | None
        The number of components used for PCA decomposition. If None, no
        dimension reduction will be applied and max_n_components will equal
        the number of channels supplied on decomposing data.
    noise_cov : None | instance of mne.cov.Covariance
        Noise covariance used for whitening. If None, channels are just
        z-scored.
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
    last_fit : str
        Flag informing about which type was last fit.
    ch_names : list-like
        Channel names resulting from initial picking.
    n_components : int
        The number of components used for ICA decomposition.
    max_n_components : int
        The number of PCA dimensions computed.
    verbose : bool, str, int, or None
        See above.
    """
    @verbose
    def __init__(self, n_components, max_n_components=100, noise_cov=None,
                 random_state=None, algorithm='parallel', fun='logcosh',
                 fun_args=None, verbose=None):
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

        if max_n_components is not None and n_components > max_n_components:
            raise ValueError('n_components must be smaller than '
                             'max_n_components')

        if isinstance(n_components, float):
            if not 0 < n_components <= 1:
                raise ValueError('For selecting ICA components by the '
                                 'explained variance of PCA components the'
                                 ' float value must be between 0.0 and 1.0 ')
            self._explained_var = n_components
            logger.info('Selecting pca_components via explained variance.')
        else:
            self._explained_var = 1.1
            logger.info('Selecting pca_components directly.')

        self._ica = FastICA(**kwargs)
        self.current_fit = 'unfitted'
        self.verbose = verbose
        self.n_components = n_components
        self.max_n_components = max_n_components
        self.ch_names = None
        self._mixing = None

    def __repr__(self):
        s = 'ICA '
        if self.current_fit == 'unfitted':
            msg = '(no'
        elif self.current_fit == 'raw':
            msg = '(raw data'
        else:
            msg = '(epochs'
        msg += ' decomposition, '

        s += msg + ('%s components' % str(self.n_components) if
               self.n_components else 'no dimension reduction') + ')'

        return s

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
        if self.current_fit != 'unfitted':
            raise RuntimeError('ICA decomposition has already been fitted. '
                               'Please start a new ICA session.')

        logger.info('Computing signal decomposition on raw data. '
                    'Please be patient, this may take some time')

        if picks is None:  # just use good data channels
            picks = pick_types(raw.info, meg=True, eeg=True, eog=False,
                               ecg=False, misc=False, stim=False,
                               exclude=raw.info['bads'])

        if self.max_n_components is None:
            self.max_n_components = len(picks)
            logger.info('Inferring max_n_components from picks.')

        self.ch_names = [raw.ch_names[k] for k in picks]

        data, self._pre_whitener = self._pre_whiten(raw[picks, start:stop][0],
                                                   raw.info, picks)

        to_ica, self._pca = self._prepare_pca(data, self.max_n_components)

        self._ica.fit(to_ica)
        self._mixing = self._ica.get_mixing_matrix().T
        self.current_fit = 'raw'

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
        if self.current_fit != 'unfitted':
            raise RuntimeError('ICA decomposition has already been fitted. '
                               'Please start a new ICA session.')

        logger.info('Computing signal decomposition on epochs. '
                    'Please be patient, this may take some time')

        if picks is None:  # just use epochs good data channels and avoid
            picks = pick_types(epochs.info, include=epochs.ch_names,  # double
                               exclude=epochs.info['bads'])  # picking

        meeg_picks = pick_types(epochs.info, meg=True, eeg=True, eog=False,
                                ecg=False, misc=False, stim=False,
                                exclude=epochs.info['bads'])

        # filter out all the channels the raw wouldn't have initialized
        picks = np.intersect1d(meeg_picks, picks)

        self.ch_names = [epochs.ch_names[k] for k in picks]

        if self.max_n_components is None:
            self.max_n_components = len(picks)
            logger.info('Inferring max_n_components from picks.')

        data, self._pre_whitener = self._pre_whiten(
                                np.hstack(epochs.get_data()[:, picks]),
                                epochs.info, picks)

        to_ica, self._pca = self._prepare_pca(data, self.max_n_components)

        self._ica.fit(to_ica)
        self._mixing = self._ica.get_mixing_matrix().T
        self.current_fit = 'epochs'

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
        if self._mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        return self._get_sources_raw(raw, start, stop)[0]

    def _get_sources_raw(self, raw, start, stop):
        picks = [raw.ch_names.index(k) for k in self.ch_names]
        data, _ = self._pre_whiten(raw[picks, start:stop][0], raw.info, picks)
        pca_data = self._pca.transform(data.T)
        raw_sources = self._ica.transform(pca_data[:, self._comp_idx]).T

        return raw_sources, pca_data

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
        if self._mixing is None:
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        return self._get_sources_epochs(epochs, concatenate)[0]

    def _get_sources_epochs(self, epochs, concatenate):

        picks = pick_types(epochs.info, include=self.ch_names,
                               exclude=epochs.info['bads'])

        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                  len(picks)))

        data, _ = self._pre_whiten(np.hstack(epochs.get_data()[:, picks]),
                                   epochs.info, picks)

        pca_data = self._pca.transform(data.T)
        sources = self._ica.transform(pca_data[:, self._comp_idx]).T
        sources = np.array(np.split(sources, len(epochs.events), 1))

        if concatenate:
            sources = np.hstack(sources)

        return sources, pca_data

    @verbose
    def save(self, fname):
        """ Store ICA session into a fiff file.

        Parameters
        ----------
        fname : str
            The absolute path of the file name to save the ICA session into.

        """
        if self.current_fit == 'unfitted':
            raise RuntimeError('No fit available. Please first fit ICA '
                               'decomposition.')

        logger.info('Wrting ica session to %s...' % fname)
        fid = start_file(fname)

        try:
            _write_ica(fid, self)
        except Exception as inst:
            os.remove(fname)
            raise inst
        end_file(fid)

    def export_sources(self, raw, picks=None, start=None, stop=None):
        """Export sources as raw object

        Parameters
        ----------
        raw : instance of Raw
            Raw object to export sources from.
        picks : array-like
            Channels to be included in addition to the sources. If None,
            artifact and stimulus channels will be included.
        start : int
            First sample to include (first is 0). If omitted, defaults to the
            first sample in data.
        stop : int
            First sample to not include. If omitted, data is included to the
            end.

        Returns
        -------
        out : instance of mne.Raw
            Container object for ICA sources

        """
        if not raw._preloaded:
            raise ValueError('raw data should be preloaded to have this '
                             'working. Please read raw data with '
                             'preload=True.')

        # include 'reference' channels for comparison with ICA
        if picks is None:
            picks = pick_types(raw.info, meg=False, eeg=False, misc=True,
                               ecg=True, eog=True, stim=True)

        # merge copied instance and picked data with sources
        out = raw.copy()
        out.fids = []
        sources = self.get_sources_raw(raw, start=start, stop=stop)
        out._data = np.r_[sources, raw[picks, start:stop][0]]

        # update first and last samples
        out.first_samp = raw.first_samp + (start if start else 0)
        out.last_samp = out.first_samp + stop if stop else raw.last_samp

        # set channel names and info
        ch_names = out.info['ch_names'] = []
        ch_info = out.info['chs'] = []
        for i in xrange(self.n_components):
            ch_names.append('ICA %03d' % (i + 1))
            ch_info.append(dict(ch_name='ICA %03d' % (i + 1), cal=1,
                logno=i + 1, coil_type=FIFF.FIFFV_COIL_NONE,
                kind=FIFF.FIFFV_MISC_CH, coord_Frame=FIFF.FIFFV_COORD_UNKNOWN,
                loc=np.array([0.,  0.,  0.,  1., 0.,  0.,  0.,  1.,
                              0.,  0.,  0.,  1.], dtype=np.float32),
                unit=FIFF.FIFF_UNIT_NONE, eeg_loc=None, range=1.0,
                scanno=i + 1, unit_mul=0, coil_trans=None))

        # re-append additionally picked ch_names
        ch_names += [raw.ch_names[k] for k in picks]
        # re-append additionally picked ch_info
        ch_info += [raw.info['chs'][k] for k in picks]

        # update number of channels
        out.info['nchan'] = len(picks) + self.n_components

        return out

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
                                     'same length.')
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
                                 'same length.')
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
            if hasattr(target, 'ndim'):
                if target.ndim < 2:
                    target = target.reshape(1, target.shape[-1])
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
            if hasattr(target, 'ndim'):
                if target.ndim < 3:
                    target = target.reshape(1, 1, target.shape[-1])
            if isinstance(target, str):
                pick = _get_target_ch(epochs, target)
                target = epochs.get_data()[:, pick]
            if sources.shape[2] != target.shape[2]:
                raise ValueError('Source and targets do not have the same'
                                 'number of time slices.')
            target = target.ravel()

        return _find_sources(np.hstack(sources), target, score_func)

    def pick_sources_raw(self, raw, include=None, exclude=None,
                         n_pca_components=64, start=None, stop=None,
                         copy=True):
        """Recompose raw data including or excluding some sources

        Parameters
        ----------
        raw : instance of Raw
            Raw object to pick to remove ICA components from.
        include : list-like | None
            The source indices to use. If None all are used.
        exclude : list-like | None
            The source indices to remove. If None  all are used.
        n_pca_components:
            The number of PCA components to be unwhitened, where n_components
            is the lower bound and max_n_components the upper bound.
            If greater than self.n_components, the PCA components that were not
            supplied to the ICA will get re-attached. This can be used to take
            back the PCA dimension reduction.
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

        if self.current_fit != 'raw':
            raise ValueError('Currently no raw data fitted.'
                             'Please fit raw data first.')

        sources, pca_data = self._get_sources_raw(raw, start=start, stop=stop)
        recomposed = self._pick_sources(sources, pca_data, include, exclude,
                                        n_pca_components)

        if copy is True:
            raw = raw.copy()

        picks = [raw.ch_names.index(k) for k in self.ch_names]
        raw[picks, start:stop] = recomposed
        return raw

    def pick_sources_epochs(self, epochs, include=None, exclude=None,
                            n_pca_components=64, copy=True):
        """Recompose epochs

        Parameters
        ----------
        epochs : instance of Epochs
            Epochs object to pick to remove ICA components from.
        include : list-like | None
            The source indices to use. If None all are used.
        exclude : list-like | None
            The source indices to remove. If None  all are used.
        n_pca_components:
            The number of PCA components to be unwhitened, where n_components
            is the lower bound and max_n_components the upper bound.
            If greater than self.n_components, the PCA components that were not
            supplied to the ICA will get re-attached. This can be used to take
            back the PCA dimension reduction.
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

        sources, pca_data = self._get_sources_epochs(epochs, True)
        picks = pick_types(epochs.info, include=self.ch_names,
                               exclude=epochs.info['bads'])

        if copy is True:
            epochs = epochs.copy()

        # put sources-dimension first for selection
        recomposed = self._pick_sources(sources, pca_data, include, exclude,
                                        n_pca_components)
        # restore epochs, channels, tsl order
        epochs._data[:, picks] = np.array(np.split(recomposed,
                                          len(epochs.events), 1))
        epochs.preload = True

        return epochs

    def _pre_whiten(self, data, info, picks):
        """Helper function"""
        if self.noise_cov is None:  # use standardization as whitener
            pre_whitener = np.atleast_1d(np.std(data)) ** -1
            data *= pre_whitener
        elif not hasattr(self, '_pre_whitener'):  # pick cov
            ncov = deepcopy(self.noise_cov)
            if data.shape[0] != ncov['data'].shape[0]:
                ncov['data'] = ncov['data'][picks][:, picks]
                assert data.shape[0] == ncov['data'].shape[0]

            pre_whitener, _ = compute_whitener(ncov, info, picks)
            data = np.dot(pre_whitener, data)
        else:
            data = np.dot(self._pre_whitener, data)
            pre_whitener = self._pre_whitener

        return data, pre_whitener

    def _prepare_pca(self, data, max_n_components):
        """ Helper Function """
        from sklearn.decomposition import RandomizedPCA

        # sklearn < 0.11 does not support random_state argument
        kwargs = {'n_components': max_n_components, 'whiten': False}

        aspec = inspect.getargspec(RandomizedPCA.__init__)
        if 'random_state' not in aspec.args:
            warnings.warn('RandomizedPCA does not support random_state '
                          'argument. Use scikit-learn to version 0.11 '
                          'or newer to get reproducible results.')
        else:
            kwargs['random_state'] = 0

        pca = RandomizedPCA(**kwargs)
        pca_data = pca.fit_transform(data.T)

        if self._explained_var > 1.0:
            if self.n_components is not None:  # normal n case
                self._comp_idx = np.arange(self.n_components)
                to_ica = pca_data[:, self._comp_idx]
            else:  # None case
                to_ica = pca_data
                self.n_components = pca_data.shape[1]
                self._comp_idx = np.arange(self.n_components)
        else:  # float case
            expl_var = pca.explained_variance_ratio_
            self._comp_idx = (np.where(expl_var.cumsum() <
                                      self._explained_var)[0])
            to_ica = pca_data[:, self._comp_idx]
            self.n_components = len(self._comp_idx)

        return to_ica, pca

    def _pick_sources(self, sources, pca_data, include, exclude,
                      n_pca_components):
        """Helper function"""
        if not(self.n_components <= n_pca_components <= self.max_n_components):
            raise ValueError('n_pca_components must be between n_components'
                             ' and max_n_components.')

        if include not in (None, []):
            mute = [i for i in xrange(len(sources)) if i not in include]
            sources[mute, :] = 0.  # include via exclusion
        elif exclude not in (None, []):
            sources[exclude, :] = 0.  # just exclude

        # restore pca data
        mixing = self._mixing.copy()
        pca_restored = np.dot(sources.T, mixing)

        # re-append deselected pca dimension if desired
        if n_pca_components - self.n_components > 0:
            add_components = np.arange(self.n_components, n_pca_components)
            pca_reappend = pca_data[:, add_components]
            pca_restored = np.c_[pca_restored, pca_reappend]

        # restore sensor space data
        out = _inverse_t_pca(pca_restored, self._pca)

        # restore scaling
        pre_whitener = self._pre_whitener.copy()
        if self.noise_cov is None:  # revert standardization
            pre_whitener **= -1
            out *= pre_whitener
        else:
            out = np.dot(out, linalg.pinv(pre_whitener))

        return out.T


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


def _inverse_t_pca(X, pca):
    """Helper Function"""
    components = pca.components_[np.arange(len(X.T))]
    X_orig = np.dot(X, components)

    if pca.mean_ is not None:
        X_orig += pca.mean_

    return X_orig


def _serialize(dict_, outer_sep=';', inner_sep=':'):
    """Aux function"""

    s = []
    for k, v in dict_.items():
        if callable(v):
            v = v.__name__
        for cls in (np.random.RandomState, Covariance):
            if isinstance(v, cls):
                v = cls.__name__
        else:
            v = str(v)
        s.append(k + inner_sep + v)

    return outer_sep.join(s)


def _deserialize(str_, outer_sep=';', inner_sep=':'):
    """Aux function"""

    out = {}
    for mapping in str_.split(outer_sep):
        k, v = mapping.split(inner_sep)
        if v == 'None':
            out[k] = None
        elif v == 'True':
            out[k] = True
        elif v == 'False':
            out[k] = False
        elif any([v.isdigit(), all([v[0] == '-', v[1:].isdigit()])]):
            out[k] = int(v)
        elif any([c.isdigit() for c in v]) and \
             any([e in v for e in ['e-', '.', 'e']]):
            out[k] = float(v)
        else:
            out[k] = v

    return out


def _write_ica(fid, ica):
    """Write an ICA object

    Parameters
    ----------
    fid: file
        The file descriptor
    ica:
        The instance of ICA to write
    """
    _pca, _ica = ica._pca, ica._ica
    _pca_params = _pca.get_params()
    _ica_params = _ica.get_params()

    for key in ('fun_args', 'fun_prime'):
        if _ica_params.get(key, None):
            _ica_params[key] = _serialize(_ica_params[key], '#')
        else:
            _ica_params[key] = str(None)

    ica_interface = dict(noise_cov=ica.noise_cov,
                         max_n_components=ica.max_n_components,
                         n_components=ica.n_components,
                         current_fit=ica.current_fit,
                         _explained_var=ica._explained_var
                         )

    unmixing_matrix_ = ica._ica.unmixing_matrix_

    start_block(fid, FIFF.FIFFB_ICA)

    #   ICA interface params
    write_string(fid, FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS,
                 _serialize(ica_interface))

    #   Channel names
    if ica.ch_names is not None:
        write_name_list(fid, FIFF.FIFF_MNE_ROW_NAMES, ica.ch_names)

    #   Whitener
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_WHITENER, ica._pre_whitener)

    #   _PCA parameters
    write_string(fid, FIFF.FIFF_MNE_ICA_PCA_PARAMS, _serialize(_pca_params))

    #   _PCA components_
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_COMPONENTS,
                        _pca.components_)

    #   _PCA explained_variance_
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR,
                       _pca.explained_variance_)
    #   _PCA mean_
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_PCA_MEAN, _pca.mean_)

    #   _ICA parameters
    write_string(fid, FIFF.FIFF_MNE_ICA_PARAMS, _serialize(_ica_params))

    #   _ICA unmixing
    write_double_matrix(fid, FIFF.FIFF_MNE_ICA_UNMIXING, unmixing_matrix_)

    ica_components_ = None
    try:  # try to write new attribute.
        ica_components_ = ica._ica.components_
    except:
        pass
    if ica_components_ is not None:
        write_double_matrix(fid, FIFF.FIFF_MNE_ICA_COMPONENTS, ica_components_)

    end_block(fid, FIFF.FIFFB_ICA)


@verbose
def read_ica(fname):
    """ Restore ICA sessions from fif file.

    Parameters
    ----------
    fname : str
        Absolute path to fif file containing ICA matrixes

    Returns
    -------
    ica : instance of ICA
    """
    try:
        from sklearn.decomposition import FastICA, RandomizedPCA
    except ImportError:
        raise Exception('the scikit-learn package is missing and '
                        'required for ICA')

    logger.info('Reading %s ...' % fname)
    fid, tree, _ = fiff_open(fname)
    ica_data = dir_tree_find(tree, FIFF.FIFFB_ICA)
    if len(ica_data) == 0:
        fid.close()
        raise ValueError('Could not find ICA data')

    my_ica_data = ica_data[0]
    ica_components_ = None
    for d in my_ica_data['directory']:
        kind = d.kind
        pos = d.pos
        if kind == FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS:
            tag = read_tag(fid, pos)
            ica_interface = tag.data
        elif kind == FIFF.FIFF_MNE_ROW_NAMES:
            tag = read_tag(fid, pos)
            ch_names = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_WHITENER:
            tag = read_tag(fid, pos)
            _pre_whitener = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_PARAMS:
            tag = read_tag(fid, pos)
            _pca_params = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_COMPONENTS:
            tag = read_tag(fid, pos)
            components_ = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR:
            tag = read_tag(fid, pos)
            explained_variance_ = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_MEAN:
            tag = read_tag(fid, pos)
            mean_ = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PARAMS:
            tag = read_tag(fid, pos)
            _ica_params = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_UNMIXING:
            tag = read_tag(fid, pos)
            unmixing_matrix_ = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_COMPONENTS:
            tag = read_tag(fid,pos)
            ica_components_ = tag.data

    fid.close()

    _pca_params = _deserialize(_pca_params)
    if _pca_params['random_state'] == np.random.RandomState.__name__:
        _pca_params['random_state'] = np.random.RandomState()
        logger.warning('Creating new RandomState object. The ensueing random '
                       'state will not match the random state from '
                       'PCA fit time.')

    interface = _deserialize(ica_interface)
    current_fit = interface.pop('current_fit')
    _explained_var = interface.pop('_explained_var')
    if interface['noise_cov'] == Covariance.__name__:
        logger.warning('The noise covariance used on fit cannot be restored.'
                       'The whitener drawn from the covariance will be used.')

    logger.info('Now restoring ICA session ...')

    _ica = FastICA(**_deserialize(_ica_params))

    _ica.unmixing_matrix_ = unmixing_matrix_
    # try to set an attribute (won't work with all sklearn versions)
    if ica_components_ is not None:
        _ica.components_ = ica_components_

    _pca = RandomizedPCA(**_pca_params)
    _pca.components_ = components_
    _pca.mean_ = mean_
    _pca.explained_variance_ = explained_variance_
    _pca.explained_variance_ratio_ = explained_variance_ / \
                                     explained_variance_.sum()
    ica = ICA(**interface)
    ica._ica, ica._pca = _ica, _pca
    ica.current_fit = current_fit
    ica.ch_names = ch_names.split(':')
    ica._comp_idx = np.arange(ica.n_components)
    ica._pre_whitener = _pre_whitener
    ica._explained_var = _explained_var
    ica._mixing = _ica.get_mixing_matrix().T

    logger.info('Ready.')

    return ica
