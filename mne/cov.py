# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import copy as cp
import os
from math import floor, ceil, log
import itertools as itt
import warnings

import numpy as np
from scipy import linalg

from .io.write import start_file, end_file
from .io.proj import (make_projector, _proj_equal, activate_proj,
                      _has_eeg_average_ref_proj)
from .io import fiff_open
from .io.pick import (pick_types, channel_indices_by_type, pick_channels_cov,
                      pick_channels, pick_info, _picks_by_type)

from .io.constants import FIFF
from .io.meas_info import read_bad_channels
from .io.proj import _read_proj, _write_proj
from .io.tag import find_tag
from .io.tree import dir_tree_find
from .io.write import (start_block, end_block, write_int, write_name_list,
                       write_double, write_float_matrix, write_string)
from .defaults import _handle_default
from .epochs import _is_good
from .utils import (check_fname, logger, verbose, estimate_rank,
                    _compute_row_norms, check_sklearn_version, _time_mask)

from .externals.six.moves import zip


def _check_covs_algebra(cov1, cov2):
    if cov1.ch_names != cov2.ch_names:
        raise ValueError('Both Covariance do not have the same list of '
                         'channels.')
    projs1 = [str(c) for c in cov1['projs']]
    projs2 = [str(c) for c in cov1['projs']]
    if projs1 != projs2:
        raise ValueError('Both Covariance do not have the same list of '
                         'SSP projections.')


def _get_tslice(epochs, tmin, tmax):
    """get the slice"""
    tstart, tend = None, None
    mask = _time_mask(epochs.times, tmin, tmax)
    tstart = np.where(mask)[0][0] if tmin is not None else None
    tend = np.where(mask)[0][-1] + 1 if tmax is not None else None
    tslice = slice(tstart, tend, None)
    return tslice


class Covariance(dict):
    """Noise covariance matrix

    Parameters
    ----------
    fname : string
        The name of the raw file.

    Attributes
    ----------
    data : array of shape (n_channels, n_channels)
        The covariance.
    ch_names : list of string
        List of channels' names.
    nfree : int
        Number of degrees of freedom i.e. number of time points used.
    """
    def __init__(self, fname):
        if fname is None:
            return

        # Reading
        fid, tree, _ = fiff_open(fname)
        self.update(_read_cov(fid, tree, FIFF.FIFFV_MNE_NOISE_COV))
        fid.close()

    @property
    def data(self):
        """Numpy array of Noise covariance matrix"""
        return self['data']

    @property
    def ch_names(self):
        """Channel names"""
        return self['names']

    @property
    def nfree(self):
        """Number of degrees of freedom"""
        return self['nfree']

    def save(self, fname):
        """Save covariance matrix in a FIF file

        Parameters
        ----------
        fname : str
            Output filename.
        """
        check_fname(fname, 'covariance', ('-cov.fif', '-cov.fif.gz'))

        fid = start_file(fname)

        try:
            _write_cov(fid, self)
        except Exception as inst:
            os.remove(fname)
            raise inst

        end_file(fid)

    def as_diag(self, copy=True):
        """Set covariance to be processed as being diagonal

        Parameters
        ----------
        copy : bool
            If True, return a modified copy of the covarince. If False,
            the covariance is modified in place.

        Returns
        -------
        cov : dict
            The covariance.

        Notes
        -----
        This function allows creation of inverse operators
        equivalent to using the old "--diagnoise" mne option.
        """
        if self['diag'] is True:
            return self.copy() if copy is True else self
        if copy is True:
            cov = cp.deepcopy(self)
        else:
            cov = self
        cov['diag'] = True
        cov['data'] = np.diag(cov['data'])
        cov['eig'] = None
        cov['eigvec'] = None
        return cov

    def __repr__(self):
        s = "size : %s x %s" % self.data.shape
        s += ", n_samples : %s" % self.nfree
        s += ", data : %s" % self.data
        return "<Covariance  |  %s>" % s

    def __add__(self, cov):
        """Add Covariance taking into account number of degrees of freedom"""
        _check_covs_algebra(self, cov)
        this_cov = cp.deepcopy(cov)
        this_cov['data'] = (((this_cov['data'] * this_cov['nfree']) +
                             (self['data'] * self['nfree'])) /
                            (self['nfree'] + this_cov['nfree']))
        this_cov['nfree'] += self['nfree']

        this_cov['bads'] = list(set(this_cov['bads']).union(self['bads']))

        return this_cov

    def __iadd__(self, cov):
        """Add Covariance taking into account number of degrees of freedom"""
        _check_covs_algebra(self, cov)
        self['data'][:] = (((self['data'] * self['nfree']) +
                            (cov['data'] * cov['nfree'])) /
                           (self['nfree'] + cov['nfree']))
        self['nfree'] += cov['nfree']

        self['bads'] = list(set(self['bads']).union(cov['bads']))

        return self

    @verbose
    def plot(self, info, exclude=[], colorbar=True, proj=False, show_svd=True,
             show=True, verbose=None):
        """Plot Covariance data

        Parameters
        ----------
        info: dict
            Measurement info.
        exclude : list of string | str
            List of channels to exclude. If empty do not exclude any channel.
            If 'bads', exclude info['bads'].
        colorbar : bool
            Show colorbar or not.
        proj : bool
            Apply projections or not.
        show_svd : bool
            Plot also singular values of the noise covariance for each sensor
            type. We show square roots ie. standard deviations.
        show : bool
            Call pyplot.show() as the end or not.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        fig_cov : instance of matplotlib.pyplot.Figure
            The covariance plot.
        fig_svd : instance of matplotlib.pyplot.Figure | None
            The SVD spectra plot of the covariance.
        """
        from .viz.misc import plot_cov
        return plot_cov(self, info, exclude, colorbar, proj, show_svd, show)


###############################################################################
# IO

@verbose
def read_cov(fname, verbose=None):
    """Read a noise covariance from a FIF file.

    Parameters
    ----------
    fname : string
        The name of file containing the covariance matrix. It should end with
        -cov.fif or -cov.fif.gz.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    cov : Covariance
        The noise covariance matrix.
    """
    check_fname(fname, 'covariance', ('-cov.fif', '-cov.fif.gz'))

    return Covariance(fname)


###############################################################################
# Estimate from data

@verbose
def make_ad_hoc_cov(info, verbose=None):
    """Create an ad hoc noise covariance

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        Measurement info.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    cov : instance of Covariance
        The ad hoc diagonal noise covariance for the M/EEG data channels.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    info = pick_info(info, pick_types(info, meg=True, eeg=True))

    # Standard deviations to be used
    grad_std = 5e-13
    mag_std = 20e-15
    eeg_std = 0.2e-6
    logger.info('Using standard noise values '
                '(MEG grad : %6.1f fT/cm MEG mag : %6.1f fT EEG : %6.1f uV)'
                % (1e13 * grad_std, 1e15 * mag_std, 1e6 * eeg_std))

    data = np.zeros(len(info['ch_names']))
    for meg, eeg, val in zip(('grad', 'mag', False), (False, False, True),
                             (grad_std, mag_std, eeg_std)):
        data[pick_types(info, meg=meg, eeg=eeg)] = val * val
    cov = Covariance(None)
    cov.update(kind=FIFF.FIFFV_MNE_NOISE_COV, diag=True, dim=len(data),
               names=info['ch_names'], data=data, projs=info['projs'],
               bads=info['bads'], nfree=0, eig=None, eigvec=None,
               info=info)
    return cov


def _check_n_samples(n_samples, n_chan):
    """Check to see if there are enough samples for reliable cov calc"""
    n_samples_min = 10 * (n_chan + 1) // 2
    if n_samples <= 0:
        raise ValueError('No samples found to compute the covariance matrix')
    if n_samples < n_samples_min:
        text = ('Too few samples (required : %d got : %d), covariance '
                'estimate may be unreliable' % (n_samples_min, n_samples))
        warnings.warn(text)
        logger.warning(text)


@verbose
def compute_raw_data_covariance(raw, tmin=None, tmax=None, tstep=0.2,
                                reject=None, flat=None, picks=None,
                                verbose=None):
    """Estimate noise covariance matrix from a continuous segment of raw data

    It is typically useful to estimate a noise covariance
    from empty room data or time intervals before starting
    the stimulation.

    Note: To speed up the computation you should consider preloading raw data
    by setting preload=True when reading the Raw data.

    Parameters
    ----------
    raw : instance of Raw
        Raw data
    tmin : float
        Beginning of time interval in seconds
    tmax : float
        End of time interval in seconds
    tstep : float
        Length of data chunks for artefact rejection in seconds.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )

    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    picks : array-like of int
        Indices of channels to include (if None, all channels
        except bad channels are used).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    cov : instance of Covariance
        Noise covariance matrix.
    """
    sfreq = raw.info['sfreq']

    # Convert to samples
    start = 0 if tmin is None else int(floor(tmin * sfreq))
    if tmax is None:
        stop = int(raw.last_samp - raw.first_samp)
    else:
        stop = int(ceil(tmax * sfreq))
    step = int(ceil(tstep * raw.info['sfreq']))

    # don't exclude any bad channels, inverses expect all channels present
    if picks is None:
        picks = pick_types(raw.info, meg=True, eeg=True, eog=False,
                           ref_meg=False, exclude=[])

    data = 0
    n_samples = 0
    mu = 0

    info = cp.copy(raw.info)
    info['chs'] = [info['chs'][k] for k in picks]
    info['ch_names'] = [info['ch_names'][k] for k in picks]
    info['nchan'] = len(picks)
    idx_by_type = channel_indices_by_type(info)

    # Read data in chuncks
    for first in range(start, stop, step):
        last = first + step
        if last >= stop:
            last = stop
        raw_segment, times = raw[picks, first:last]
        if _is_good(raw_segment, info['ch_names'], idx_by_type, reject, flat,
                    ignore_chs=info['bads']):
            mu += raw_segment.sum(axis=1)
            data += np.dot(raw_segment, raw_segment.T)
            n_samples += raw_segment.shape[1]
        else:
            logger.info("Artefact detected in [%d, %d]" % (first, last))

    _check_n_samples(n_samples, len(picks))
    mu /= n_samples
    data -= n_samples * mu[:, None] * mu[None, :]
    data /= (n_samples - 1.0)
    logger.info("Number of samples used : %d" % n_samples)
    logger.info('[done]')

    cov = Covariance(None)

    ch_names = [raw.info['ch_names'][k] for k in picks]
    # XXX : do not compute eig and eigvec now (think it's better...)
    eig = None
    eigvec = None

    #   Store structure for fif
    cov.update(kind=FIFF.FIFFV_MNE_NOISE_COV, diag=False, dim=len(data),
               names=ch_names, data=data,
               projs=cp.deepcopy(raw.info['projs']),
               bads=raw.info['bads'], nfree=n_samples, eig=eig,
               eigvec=eigvec)

    return cov


@verbose
def compute_covariance(epochs, keep_sample_mean=True, tmin=None, tmax=None,
                       projs=None, method='empirical', method_params=None,
                       cv=3, scalings=None, n_jobs=1, return_estimators=False,
                       verbose=None):
    """Estimate noise covariance matrix from epochs

    The noise covariance is typically estimated on pre-stim periods
    when the stim onset is defined from events.

    If the covariance is computed for multiple event types (events
    with different IDs), the following two options can be used and combined.
    A) either an Epochs object for each event type is created and
    a list of Epochs is passed to this function.
    B) an Epochs object is created for multiple events and passed
    to this function.

    Note: Baseline correction should be used when creating the Epochs.
          Otherwise the computed covariance matrix will be inaccurate.

    Note: For multiple event types, it is also possible to create a
          single Epochs object with events obtained using
          merge_events(). However, the resulting covariance matrix
          will only be correct if keep_sample_mean is True.

    Note: The covariance can be unstable if the number of samples is not
          sufficient. In that case it is common to regularize a covariance
          estimate. The ``method`` parameter of this function allows to
          regularize the covariance in an automated way. It also allows
          to select between different alternative estimation algorithms which
          themselves achieve regularization. Details are described in [1].

    Parameters
    ----------
    epochs : instance of Epochs, or a list of Epochs objects
        The epochs.
    keep_sample_mean : bool
        If False, the average response over epochs is computed for
        each event type and subtracted during the covariance
        computation. This is useful if the evoked response from a
        previous stimulus extends into the baseline period of the next.
        Note. This option is only implemented for method='empirical'.
    tmin : float | None
        Start time for baseline. If None start at first sample.
    tmax : float | None
        End time for baseline. If None end at last sample.
    projs : list of Projection | None
        List of projectors to use in covariance calculation, or None
        to indicate that the projectors from the epochs should be
        inherited. If None, then projectors from all epochs must match.
    method : str | list | None
        The method used for covariance estimation. If 'empirical' (default),
        the sample covariance will be computed. A list can be passed to run a
        set of the different methods.
        If 'auto' or a list of methods, the best estimator will be determined
        based on log-likelihood and cross-validation on unseen data as
        described in ref. [1]. Valid methods are:
        'empirical', the empirical or sample covariance,
        'diagonal_fixed', a diagonal regularization as in mne.cov.regularize
        (see MNE manual), 'ledoit_wolf', the Ledoit-Wolf estimator (see [2]),
        'shrunk' like 'ledoit_wolf' with cross-validation for optimal alpha
        (see scikit-learn documentation on covariance estimation), 'pca',
        probabilistic PCA with low rank
        (see [3]), and, 'factor_analysis', Factor Analysis with low rank
        (see [4]). If 'auto', expands to::

             ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']

        Note. 'ledoit_wolf' and 'pca' are similar to 'shrunk' and
        'factor_analysis', respectively. They are not included to avoid
        redundancy. In most cases 'shrunk' and 'factor_analysis' represent
        more appropriate default choices.

        .. versionadded:: 0.9.0

    method_params : dict
        Additional parameters to the estimation procedure. Only considered if
        method is not None. Keys must correspond to the value(s) of `method`.
        If None (default), expands to::

            'empirical': {'store_precision': False, 'assume_centered': True},
            'diagonal_fixed': {'grad': 0.01, 'mag': 0.01, 'eeg': 0.0,
                               'store_precision': False,
                               'assume_centered': True},
            'ledoit_wolf': {'store_precision': False, 'assume_centered': True},
            'shrunk': {'shrinkage': np.logspace(-4, 0, 30),
                       'store_precision': False, 'assume_centered': True},
            'pca': {'iter_n_components': None},
            'factor_analysis': {'iter_n_components': None}

    cv : int | sklearn cross_validation object
        The cross validation method. Defaults to 3, which will
        internally trigger a default 3-fold shuffle split.
    scalings : dict
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale magnetometers and gradiometers
        at the same unit.
    n_jobs : int
        Number of jobs to run in parallel.
    return_estimators : bool
        Whether to return all estimators or the best. Only considered if
        method equals 'auto' or is a list of str. Defaults to False
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    cov : instance of Covariance | list
        The computed covariance. If method equals 'auto' or is a list of str
        and return_estimators equals True, a list of covariance estimators is
        returned (sorted by log-likelihood, from high to low, i.e. from best
        to worst).

    References
    ----------
    [1] Engemann D. and Gramfort A. (2015) Automated model selection in
        covariance estimation and spatial whitening of MEG and EEG signals,
        vol. 108, 328-342, NeuroImage.
    [2] Ledoit, O., Wolf, M., (2004). A well-conditioned estimator for
        large-dimensional covariance matrices. Journal of Multivariate
        Analysis 88 (2), 365 - 411.
    [3] Tipping, M. E., Bishop, C. M., (1999). Probabilistic principal
        component analysis. Journal of the Royal Statistical Society: Series
        B (Statistical Methodology) 61 (3), 611 - 622.
    [4] Barber, D., (2012). Bayesian reasoning and machine learning.
        Cambridge University Press., Algorithm 21.1
    """
    accepted_methods = ('auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf',
                        'shrunk', 'pca', 'factor_analysis',)
    msg = ('Invalid method ({method}). Accepted values (individually or '
           'in a list) are "%s"' % '" or "'.join(accepted_methods + ('None',)))

    # scale to natural unit for best stability with MEG/EEG
    if isinstance(scalings, dict):
        for k, v in scalings.items():
            if k not in ('mag', 'grad', 'eeg'):
                raise ValueError('The keys in `scalings` must be "mag" or'
                                 '"grad" or "eeg". You gave me: %s' % k)
    scalings = _handle_default('scalings', scalings)

    _method_params = {
        'empirical': {'store_precision': False, 'assume_centered': True},
        'diagonal_fixed': {'grad': 0.01, 'mag': 0.01, 'eeg': 0.0,
                           'store_precision': False, 'assume_centered': True},
        'ledoit_wolf': {'store_precision': False, 'assume_centered': True},
        'shrunk': {'shrinkage': np.logspace(-4, 0, 30),
                   'store_precision': False, 'assume_centered': True},
        'pca': {'iter_n_components': None},
        'factor_analysis': {'iter_n_components': None}
    }
    if isinstance(method_params, dict):
        for key, values in method_params.items():
            if key not in _method_params:
                raise ValueError('key (%s) must be "%s"' %
                                 (key, '" or "'.join(_method_params)))

            _method_params[key].update(method_params[key])

    # for multi condition support epochs is required to refer to a list of
    # epochs objects

    def _unpack_epochs(epochs):
        if len(epochs.event_id) > 1:
            epochs = [epochs[k] for k in epochs.event_id]
        else:
            epochs = [epochs]
        return epochs

    if not isinstance(epochs, list):
        epochs = _unpack_epochs(epochs)
    else:
        epochs = sum([_unpack_epochs(epoch) for epoch in epochs], [])

    # check for baseline correction
    for epochs_t in epochs:
        if epochs_t.baseline is None and epochs_t.info['highpass'] < 0.5:
            warnings.warn('Epochs are not baseline corrected, covariance '
                          'matrix may be inaccurate')

    bads = epochs[0].info['bads']
    if projs is None:
        projs = cp.deepcopy(epochs[0].info['projs'])
        # make sure Epochs are compatible
        for epochs_t in epochs[1:]:
            if epochs_t.proj != epochs[0].proj:
                raise ValueError('Epochs must agree on the use of projections')
            for proj_a, proj_b in zip(epochs_t.info['projs'], projs):
                if not _proj_equal(proj_a, proj_b):
                    raise ValueError('Epochs must have same projectors')
    else:
        projs = cp.deepcopy(projs)
    ch_names = epochs[0].ch_names

    # make sure Epochs are compatible
    for epochs_t in epochs[1:]:
        if epochs_t.info['bads'] != bads:
            raise ValueError('Epochs must have same bad channels')
        if epochs_t.ch_names != ch_names:
            raise ValueError('Epochs must have same channel names')
    picks_list = _picks_by_type(epochs[0].info)
    picks_meeg = np.concatenate([b for _, b in picks_list])
    picks_meeg = np.sort(picks_meeg)
    ch_names = [epochs[0].ch_names[k] for k in picks_meeg]
    info = epochs[0].info  # we will overwrite 'epochs'

    if method == 'auto':
        method = ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']

    if not isinstance(method, (list, tuple)):
        method = [method]

    ok_sklearn = check_sklearn_version('0.15') is True
    if not ok_sklearn and (len(method) != 1 or method[0] != 'empirical'):
        raise ValueError('scikit-learn is not installed, `method` must be '
                         '`empirical`')

    if keep_sample_mean is False:
        if len(method) != 1 or 'empirical' not in method:
            raise ValueError('`keep_sample_mean=False` is only supported'
                             'with `method="empirical"`')
        for p, v in _method_params.items():
            if v.get('assume_centered', None) is False:
                raise ValueError('`assume_centered` must be True'
                                 ' if `keep_sample_mean` is False')
        # prepare mean covs
        n_epoch_types = len(epochs)
        data_mean = list(np.zeros(n_epoch_types))
        n_samples = np.zeros(n_epoch_types, dtype=np.int)
        n_epochs = np.zeros(n_epoch_types, dtype=np.int)

        for ii, epochs_t in enumerate(epochs):

            tslice = _get_tslice(epochs_t, tmin, tmax)
            for e in epochs_t:
                e = e[picks_meeg, tslice]
                if not keep_sample_mean:
                    data_mean[ii] += e
                n_samples[ii] += e.shape[1]
                n_epochs[ii] += 1

        n_samples_epoch = n_samples // n_epochs
        norm_const = np.sum(n_samples_epoch * (n_epochs - 1))
        data_mean = [1.0 / n_epoch * np.dot(mean, mean.T) for n_epoch, mean
                     in zip(n_epochs, data_mean)]

    if not all(k in accepted_methods for k in method):
        raise ValueError(msg.format(method=method))

    info = pick_info(info, picks_meeg)
    tslice = _get_tslice(epochs[0], tmin, tmax)
    epochs = [ee.get_data()[:, picks_meeg, tslice] for ee in epochs]
    picks_meeg = np.arange(len(picks_meeg))
    picks_list = _picks_by_type(info)

    if len(epochs) > 1:
        epochs = np.concatenate(epochs, 0)
    else:
        epochs = epochs[0]

    epochs = np.hstack(epochs)
    n_samples_tot = epochs.shape[-1]
    _check_n_samples(n_samples_tot, len(picks_meeg))

    epochs = epochs.T  # sklearn | C-order
    if ok_sklearn:
        cov_data = _compute_covariance_auto(epochs, method=method,
                                            method_params=_method_params,
                                            info=info,
                                            verbose=verbose,
                                            cv=cv,
                                            n_jobs=n_jobs,
                                            # XXX expose later
                                            stop_early=True,  # if needed.
                                            picks_list=picks_list,
                                            scalings=scalings)
    else:
        if _method_params['empirical']['assume_centered'] is True:
            cov = epochs.T.dot(epochs) / n_samples_tot
        else:
            cov = np.cov(epochs.T, bias=1)
        cov_data = {'empirical': {'data': cov}}

    if keep_sample_mean is False:
        cov = cov_data['empirical']['data']
        # undo scaling
        cov *= n_samples_tot
        # ... apply pre-computed class-wise normalization
        for mean_cov in data_mean:
            cov -= mean_cov
        cov /= norm_const

    covs = list()
    for this_method, data in cov_data.items():
        cov = Covariance(None)
        cov.update(kind=1, diag=False, dim=len(data['data']), names=ch_names,
                   data=data.pop('data'), projs=projs, bads=info['bads'],
                   nfree=n_samples_tot, eig=None, eigvec=None)

        logger.info('Number of samples used : %d' % n_samples_tot)
        logger.info('[done]')

        # add extra info
        cov.update(method=this_method, **data)
        covs.append(cov)

    if ok_sklearn:
        msg = ['log-likelihood on unseen data (descending order):']
        logliks = [(c['method'], c['loglik']) for c in covs]
        logliks.sort(reverse=True, key=lambda c: c[1])
        for k, v in logliks:
            msg.append('%s: %0.3f' % (k, v))
        logger.info('\n   '.join(msg))

    if ok_sklearn and not return_estimators:
        keys, scores = zip(*[(c['method'], c['loglik']) for c in covs])
        out = covs[np.argmax(scores)]
        logger.info('selecting best estimator: {0}'.format(out['method']))
    elif ok_sklearn:
        out = covs
        out.sort(key=lambda c: c['loglik'], reverse=True)
    else:
        out = covs[0]

    return out


def _compute_covariance_auto(data, method, info, method_params, cv,
                             scalings, n_jobs, stop_early, picks_list,
                             verbose):
    """docstring for _compute_covariance_auto"""
    from sklearn.grid_search import GridSearchCV
    from sklearn.covariance import (LedoitWolf, ShrunkCovariance,
                                    EmpiricalCovariance)

    # rescale to improve numerical stability
    _apply_scaling_array(data.T, picks_list=picks_list, scalings=scalings)
    estimator_cov_info = list()
    msg = 'Estimating covariance using %s'
    _RegCovariance, _ShrunkCovariance = _get_covariance_classes()
    for this_method in method:
        data_ = data.copy()
        name = this_method.__name__ if callable(this_method) else this_method
        logger.info(msg % name.upper())

        if this_method == 'empirical':
            est = EmpiricalCovariance(**method_params[this_method])
            est.fit(data_)
            _info = None
            estimator_cov_info.append((est, est.covariance_, _info))

        elif this_method == 'diagonal_fixed':
            est = _RegCovariance(info=info, **method_params[this_method])
            est.fit(data_)
            _info = None
            estimator_cov_info.append((est, est.covariance_, _info))

        elif this_method == 'ledoit_wolf':
            shrinkages = []
            lw = LedoitWolf(**method_params[this_method])

            for ch_type, picks in picks_list:
                lw.fit(data_[:, picks])
                shrinkages.append((
                    ch_type,
                    lw.shrinkage_,
                    picks
                ))
            sc = _ShrunkCovariance(shrinkage=shrinkages,
                                   **method_params[this_method])
            sc.fit(data_)
            _info = None
            estimator_cov_info.append((sc, sc.covariance_, _info))

        elif this_method == 'shrunk':
            shrinkage = method_params[this_method].pop('shrinkage')
            tuned_parameters = [{'shrinkage': shrinkage}]
            shrinkages = []
            gs = GridSearchCV(ShrunkCovariance(**method_params[this_method]),
                              tuned_parameters, cv=cv)
            for ch_type, picks in picks_list:
                gs.fit(data_[:, picks])
                shrinkages.append((
                    ch_type,
                    gs.best_estimator_.shrinkage,
                    picks
                ))
            shrinkages = [c[0] for c in zip(shrinkages)]
            sc = _ShrunkCovariance(shrinkage=shrinkages,
                                   **method_params[this_method])
            sc.fit(data_)
            _info = None
            estimator_cov_info.append((sc, sc.covariance_, _info))

        elif this_method == 'pca':
            mp = method_params[this_method]
            pca, _info = _auto_low_rank_model(data_, this_method,
                                              n_jobs=n_jobs,
                                              method_params=mp, cv=cv,
                                              stop_early=stop_early)
            pca.fit(data_)
            estimator_cov_info.append((pca, pca.get_covariance(), _info))

        elif this_method == 'factor_analysis':
            mp = method_params[this_method]
            fa, _info = _auto_low_rank_model(data_, this_method, n_jobs=n_jobs,
                                             method_params=mp, cv=cv,
                                             stop_early=stop_early)
            fa.fit(data_)
            estimator_cov_info.append((fa, fa.get_covariance(), _info))
        else:
            raise ValueError('Oh no! Your estimator does not have'
                             ' a .fit method')
        logger.info('Done.')

    logger.info('Using cross-validation to select the best estimator.')
    estimators, _, _ = zip(*estimator_cov_info)
    logliks = np.array([_cross_val(data, e, cv, n_jobs) for e in estimators])

    # undo scaling
    for c in estimator_cov_info:
        _undo_scaling_cov(c[1], picks_list, scalings)

    out = dict()
    estimators, covs, runtime_infos = zip(*estimator_cov_info)
    cov_methods = [c.__name__ if callable(c) else c for c in method]
    runtime_infos, covs = list(runtime_infos), list(covs)
    my_zip = zip(cov_methods, runtime_infos, logliks, covs, estimators)
    for this_method, runtime_info, loglik, data, est in my_zip:
        out[this_method] = {'loglik': loglik, 'data': data, 'estimator': est}
        if runtime_info is not None:
            out[this_method].update(runtime_info)

    return out


def _logdet(A):
    """Compute the log det of a symmetric matrix"""
    vals = linalg.eigh(A)[0]
    vals = np.abs(vals)  # avoid negative values (numerical errors)
    return np.sum(np.log(vals))


def _gaussian_loglik_scorer(est, X, y=None):
    """Compute the Gaussian log likelihood of X under the model in est"""
    # compute empirical covariance of the test set
    precision = est.get_precision()
    n_samples, n_features = X.shape
    log_like = np.zeros(n_samples)
    log_like = -.5 * (X * (np.dot(X, precision))).sum(axis=1)
    log_like -= .5 * (n_features * log(2. * np.pi) - _logdet(precision))
    out = np.mean(log_like)
    return out


def _cross_val(data, est, cv, n_jobs):
    """Helper to compute cross validation"""
    from sklearn.cross_validation import cross_val_score
    return np.mean(cross_val_score(est, data, cv=cv, n_jobs=n_jobs,
                                   scoring=_gaussian_loglik_scorer))


def _auto_low_rank_model(data, mode, n_jobs, method_params, cv,
                         stop_early=True, verbose=None):
    """compute latent variable models"""
    method_params = cp.deepcopy(method_params)
    iter_n_components = method_params.pop('iter_n_components')
    if iter_n_components is None:
        iter_n_components = np.arange(5, data.shape[1], 5)
    from sklearn.decomposition import PCA, FactorAnalysis
    if mode == 'factor_analysis':
        est = FactorAnalysis
    elif mode == 'pca':
        est = PCA
    else:
        raise ValueError('Come on, this is not a low rank estimator: %s' %
                         mode)
    est = est(**method_params)
    est.n_components = 1
    scores = np.empty_like(iter_n_components, dtype=np.float64)
    scores.fill(np.nan)

    # make sure we don't empty the thing if it's a generator
    max_n = max(list(cp.deepcopy(iter_n_components)))
    if max_n > data.shape[1]:
        warnings.warn('You are trying to estimate %i components on matrix '
                      'with %i features.' % (max_n, data.shape[1]))

    for ii, n in enumerate(iter_n_components):
        est.n_components = n
        try:  # this may fail depending on rank and split
            score = _cross_val(data=data, est=est, cv=cv, n_jobs=n_jobs)
        except ValueError:
            score = np.inf
        if np.isinf(score) or score > 0:
            logger.info('... infinite values encountered. stopping estimation')
            break
        logger.info('... rank: %i - loglik: %0.3f' % (n, score))
        if score != -np.inf:
            scores[ii] = score

        if (ii >= 3 and np.all(np.diff(scores[ii - 3:ii]) < 0.) and
           stop_early is True):
            # early stop search when loglik has been going down 3 times
            logger.info('early stopping parameter search.')
            break

    # happens if rank is too low right form the beginning
    if np.isnan(scores).all():
        raise RuntimeError('Oh no! Could not estimate covariance because all '
                           'scores were NaN. Please contact the MNE-Python '
                           'developers.')

    i_score = np.nanargmax(scores)
    best = est.n_components = iter_n_components[i_score]
    logger.info('... best model at rank = %i' % best)
    runtime_info = {'ranks': np.array(iter_n_components),
                    'scores': scores,
                    'best': best,
                    'cv': cv}
    return est, runtime_info


def _get_covariance_classes():
    """Prepare special cov estimators"""
    from sklearn.covariance import (EmpiricalCovariance, shrunk_covariance,
                                    ShrunkCovariance)

    class _RegCovariance(EmpiricalCovariance):
        """Aux class"""

        def __init__(self, info, grad=0.01, mag=0.01, eeg=0.0,
                     store_precision=False, assume_centered=False):
            self.info = info
            self.grad = grad
            self.mag = mag
            self.eeg = eeg
            self.store_precision = store_precision
            self.assume_centered = assume_centered

        def fit(self, X):
            EmpiricalCovariance.fit(self, X)
            self.covariance_ = 0.5 * (self.covariance_ + self.covariance_.T)
            cov_ = Covariance(None)
            cov_['data'] = self.covariance_
            cov_['names'] = self.info['ch_names']
            cov_['nfree'] = len(self.covariance_)
            cov_['bads'] = self.info['bads']
            cov_['projs'] = self.info['projs']
            cov_['diag'] = False
            cov_ = regularize(cov_, self.info, grad=self.grad, mag=self.mag,
                              eeg=self.eeg, proj=False,
                              exclude='bads')  # ~proj == important!!
            self.covariance_ = cov_.data
            return self

    class _ShrunkCovariance(ShrunkCovariance):
        """Aux class"""

        def __init__(self, store_precision, assume_centered, shrinkage=0.1):
            self.store_precision = store_precision
            self.assume_centered = assume_centered
            self.shrinkage = shrinkage

        def fit(self, X):
            EmpiricalCovariance.fit(self, X)
            cov = self.covariance_

            if not isinstance(self.shrinkage, (list, tuple)):
                shrinkage = [('all', self.shrinkage, np.arange(len(cov)))]
            else:
                shrinkage = self.shrinkage

            zero_cross_cov = np.zeros_like(cov, dtype=bool)
            for a, b in itt.combinations(shrinkage, 2):
                picks_i, picks_j = a[2], b[2]
                ch_ = a[0], b[0]
                if 'eeg' in ch_:
                    zero_cross_cov[np.ix_(picks_i, picks_j)] = True
                    zero_cross_cov[np.ix_(picks_j, picks_i)] = True

            self.zero_cross_cov_ = zero_cross_cov

            # Apply shrinkage to blocks
            for ch_type, c, picks in shrinkage:
                sub_cov = cov[np.ix_(picks, picks)]
                cov[np.ix_(picks, picks)] = shrunk_covariance(sub_cov,
                                                              shrinkage=c)

            # Apply shrinkage to cross-cov
            for a, b in itt.combinations(shrinkage, 2):
                shrinkage_i, shrinkage_j = a[1], b[1]
                picks_i, picks_j = a[2], b[2]
                c_ij = np.sqrt((1. - shrinkage_i) * (1. - shrinkage_j))
                cov[np.ix_(picks_i, picks_j)] *= c_ij
                cov[np.ix_(picks_j, picks_i)] *= c_ij

            # Set to zero the necessary cross-cov
            if np.any(zero_cross_cov):
                cov[zero_cross_cov] = 0.0

            self.covariance_ = cov
            return self

        def score(self, X_test, y=None):
            """Computes the log-likelihood of a Gaussian data set with
            `self.covariance_` as an estimator of its covariance matrix.

            Parameters
            ----------
            X_test : array-like, shape = [n_samples, n_features]
                Test data of which we compute the likelihood, where n_samples
                is the number of samples and n_features is the number of
                features. X_test is assumed to be drawn from the same
                distribution as the data used in fit (including centering).

            y : not used, present for API consistence purpose.

            Returns
            -------
            res : float
                The likelihood of the data set with `self.covariance_` as an
                estimator of its covariance matrix.
            """
            from sklearn.covariance import empirical_covariance, log_likelihood
            # compute empirical covariance of the test set
            test_cov = empirical_covariance(X_test - self.location_,
                                            assume_centered=True)
            if np.any(self.zero_cross_cov_):
                test_cov[self.zero_cross_cov_] = 0.
            res = log_likelihood(test_cov, self.get_precision())
            return res

    return _RegCovariance, _ShrunkCovariance


###############################################################################
# Writing

def write_cov(fname, cov):
    """Write a noise covariance matrix

    Parameters
    ----------
    fname : string
        The name of the file. It should end with -cov.fif or -cov.fif.gz.
    cov : Covariance
        The noise covariance matrix
    """
    cov.save(fname)


###############################################################################
# Prepare for inverse modeling

def _unpack_epochs(epochs):
    """ Aux Function """
    if len(epochs.event_id) > 1:
        epochs = [epochs[k] for k in epochs.event_id]
    else:
        epochs = [epochs]

    return epochs


def _get_ch_whitener(A, pca, ch_type, rank):
    """"Get whitener params for a set of channels"""
    # whitening operator
    eig, eigvec = linalg.eigh(A, overwrite_a=True)
    eigvec = eigvec.T
    eig[:-rank] = 0.0

    logger.info('Setting small %s eigenvalues to zero.' % ch_type)
    if not pca:  # No PCA case.
        logger.info('Not doing PCA for %s.' % ch_type)
    else:
        logger.info('Doing PCA for %s.' % ch_type)
        # This line will reduce the actual number of variables in data
        # and leadfield to the true rank.
        eigvec = eigvec[:-rank].copy()
    return eig, eigvec


@verbose
def prepare_noise_cov(noise_cov, info, ch_names, rank=None,
                      scalings=None, verbose=None):
    """Prepare noise covariance matrix

    Parameters
    ----------
    noise_cov : Covariance
        The noise covariance to process.
    info : dict
        The measurement info (used to get channel types and bad channels).
    ch_names : list
        The channel names to be considered.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    scalings : dict | None
        Data will be rescaled before rank estimation to improve accuracy.
        If dict, it will override the following dict (default if None):

            dict(mag=1e12, grad=1e11, eeg=1e5)

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    C_ch_idx = [noise_cov.ch_names.index(c) for c in ch_names]
    if noise_cov['diag'] is False:
        C = noise_cov.data[np.ix_(C_ch_idx, C_ch_idx)]
    else:
        C = np.diag(noise_cov.data[C_ch_idx])

    scalings = _handle_default('scalings_cov_rank', scalings)

    # Create the projection operator
    proj, ncomp, _ = make_projector(info['projs'], ch_names)
    if ncomp > 0:
        logger.info('    Created an SSP operator (subspace dimension = %d)'
                    % ncomp)
        C = np.dot(proj, np.dot(C, proj.T))

    pick_meg = pick_types(info, meg=True, eeg=False, ref_meg=False,
                          exclude='bads')
    pick_eeg = pick_types(info, meg=False, eeg=True, ref_meg=False,
                          exclude='bads')
    meg_names = [info['chs'][k]['ch_name'] for k in pick_meg]
    C_meg_idx = [k for k in range(len(C)) if ch_names[k] in meg_names]
    eeg_names = [info['chs'][k]['ch_name'] for k in pick_eeg]
    C_eeg_idx = [k for k in range(len(C)) if ch_names[k] in eeg_names]

    has_meg = len(C_meg_idx) > 0
    has_eeg = len(C_eeg_idx) > 0

    # Get the specified noise covariance rank
    if rank is not None:
        if isinstance(rank, dict):
            rank_meg = rank.get('meg', None)
            rank_eeg = rank.get('eeg', None)
        else:
            rank_meg = int(rank)
            rank_eeg = None
    else:
        rank_meg, rank_eeg = None, None

    if has_meg:
        C_meg = C[np.ix_(C_meg_idx, C_meg_idx)]
        this_info = pick_info(info, pick_meg)
        if rank_meg is None:
            if len(C_meg_idx) < len(pick_meg):
                this_info = pick_info(info, C_meg_idx)
            rank_meg = _estimate_rank_meeg_cov(C_meg, this_info, scalings)
        C_meg_eig, C_meg_eigvec = _get_ch_whitener(C_meg, False, 'MEG',
                                                   rank_meg)
    if has_eeg:
        C_eeg = C[np.ix_(C_eeg_idx, C_eeg_idx)]
        this_info = pick_info(info, pick_eeg)
        if rank_eeg is None:
            if len(C_meg_idx) < len(pick_meg):
                this_info = pick_info(info, C_eeg_idx)
            rank_eeg = _estimate_rank_meeg_cov(C_eeg, this_info, scalings)
        C_eeg_eig, C_eeg_eigvec = _get_ch_whitener(C_eeg, False, 'EEG',
                                                   rank_eeg)
        if not _has_eeg_average_ref_proj(info['projs']):
            warnings.warn('No average EEG reference present in info["projs"], '
                          'covariance may be adversely affected. Consider '
                          'recomputing covariance using a raw file with an '
                          'average eeg reference projector added.')

    n_chan = len(ch_names)
    eigvec = np.zeros((n_chan, n_chan), dtype=np.float)
    eig = np.zeros(n_chan, dtype=np.float)

    if has_meg:
        eigvec[np.ix_(C_meg_idx, C_meg_idx)] = C_meg_eigvec
        eig[C_meg_idx] = C_meg_eig
    if has_eeg:
        eigvec[np.ix_(C_eeg_idx, C_eeg_idx)] = C_eeg_eigvec
        eig[C_eeg_idx] = C_eeg_eig

    assert(len(C_meg_idx) + len(C_eeg_idx) == n_chan)

    noise_cov = cp.deepcopy(noise_cov)
    noise_cov.update(data=C, eig=eig, eigvec=eigvec, dim=len(ch_names),
                     diag=False, names=ch_names)

    return noise_cov


def regularize(cov, info, mag=0.1, grad=0.1, eeg=0.1, exclude='bads',
               proj=True, verbose=None):
    """Regularize noise covariance matrix

    This method works by adding a constant to the diagonal for each
    channel type separately. Special care is taken to keep the
    rank of the data constant.

    **Note:** This function is kept for reasons of backward-compatibility.
    Please consider explicitly using the ``method`` parameter in
    `compute_covariance` to directly combine estimation with regularization
    in a data-driven fashion see the
    `faq <http://martinos.org/mne/dev/faq.html#how-should-i-regularize-the-covariance-matrix>`_
    for more information.

    Parameters
    ----------
    cov : Covariance
        The noise covariance matrix.
    info : dict
        The measurement info (used to get channel types and bad channels).
    mag : float
        Regularization factor for MEG magnetometers.
    grad : float
        Regularization factor for MEG gradiometers.
    eeg : float
        Regularization factor for EEG.
    exclude : list | 'bads'
        List of channels to mark as bad. If 'bads', bads channels
        are extracted from both info['bads'] and cov['bads'].
    proj : bool
        Apply or not projections to keep rank of data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    reg_cov : Covariance
        The regularized covariance matrix.
    """  # noqa
    cov = cp.deepcopy(cov)

    if exclude is None:
        raise ValueError('exclude must be a list of strings or "bads"')

    if exclude == 'bads':
        exclude = info['bads'] + cov['bads']

    sel_eeg = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude=exclude)
    sel_mag = pick_types(info, meg='mag', eeg=False, ref_meg=False,
                         exclude=exclude)
    sel_grad = pick_types(info, meg='grad', eeg=False, ref_meg=False,
                          exclude=exclude)

    info_ch_names = info['ch_names']
    ch_names_eeg = [info_ch_names[i] for i in sel_eeg]
    ch_names_mag = [info_ch_names[i] for i in sel_mag]
    ch_names_grad = [info_ch_names[i] for i in sel_grad]

    # This actually removes bad channels from the cov, which is not backward
    # compatible, so let's leave all channels in
    cov_good = pick_channels_cov(cov, include=info_ch_names, exclude=exclude)
    ch_names = cov_good.ch_names

    idx_eeg, idx_mag, idx_grad = [], [], []
    for i, ch in enumerate(ch_names):
        if ch in ch_names_eeg:
            idx_eeg.append(i)
        elif ch in ch_names_mag:
            idx_mag.append(i)
        elif ch in ch_names_grad:
            idx_grad.append(i)
        else:
            raise Exception('channel is unknown type')

    C = cov_good['data']

    assert len(C) == (len(idx_eeg) + len(idx_mag) + len(idx_grad))

    if proj:
        projs = info['projs'] + cov_good['projs']
        projs = activate_proj(projs)

    for desc, idx, reg in [('EEG', idx_eeg, eeg), ('MAG', idx_mag, mag),
                           ('GRAD', idx_grad, grad)]:
        if len(idx) == 0 or reg == 0.0:
            logger.info("    %s regularization : None" % desc)
            continue

        logger.info("    %s regularization : %s" % (desc, reg))

        this_C = C[np.ix_(idx, idx)]
        if proj:
            this_ch_names = [ch_names[k] for k in idx]
            P, ncomp, _ = make_projector(projs, this_ch_names)
            U = linalg.svd(P)[0][:, :-ncomp]
            if ncomp > 0:
                logger.info('    Created an SSP operator for %s '
                            '(dimension = %d)' % (desc, ncomp))
                this_C = np.dot(U.T, np.dot(this_C, U))

        sigma = np.mean(np.diag(this_C))
        this_C.flat[::len(this_C) + 1] += reg * sigma  # modify diag inplace
        if proj and ncomp > 0:
            this_C = np.dot(U, np.dot(this_C, U.T))

        C[np.ix_(idx, idx)] = this_C

    # Put data back in correct locations
    idx = pick_channels(cov.ch_names, info_ch_names, exclude=exclude)
    cov['data'][np.ix_(idx, idx)] = C

    return cov


def compute_whitener(noise_cov, info, picks=None, rank=None,
                     scalings=None, verbose=None):
    """Compute whitening matrix

    Parameters
    ----------
    noise_cov : Covariance
        The noise covariance.
    info : dict
        The measurement info.
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    scalings : dict | None
        The rescaling method to be applied. See documentation of
        ``prepare_noise_cov`` for details.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    W : 2d array
        The whitening matrix.
    ch_names : list
        The channel names.
    """
    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, ref_meg=False,
                           exclude='bads')

    ch_names = [info['chs'][k]['ch_name'] for k in picks]

    noise_cov = cp.deepcopy(noise_cov)
    noise_cov = prepare_noise_cov(noise_cov, info, ch_names,
                                  rank=rank, scalings=scalings)
    n_chan = len(ch_names)

    W = np.zeros((n_chan, n_chan), dtype=np.float)
    #
    #   Omit the zeroes due to projection
    #
    eig = noise_cov['eig']
    nzero = (eig > 0)
    W[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
    #
    #   Rows of eigvec are the eigenvectors
    #
    W = np.dot(W, noise_cov['eigvec'])
    W = np.dot(noise_cov['eigvec'].T, W)
    return W, ch_names


@verbose
def whiten_evoked(evoked, noise_cov, picks=None, diag=False, rank=None,
                  scalings=None, verbose=None):
    """Whiten evoked data using given noise covariance

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    noise_cov : instance of Covariance
        The noise covariance
    picks : array-like of int | None
        The channel indices to whiten. Can be None to whiten MEG and EEG
        data.
    diag : bool
        If True, whiten using only the diagonal of the covariance.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    scalings : dict | None
        To achieve reliable rank estimation on multiple sensors,
        sensors have to be rescaled. This parameter controls the
        rescaling. If dict, it will override the
        following default dict (default if None):

            dict(mag=1e12, grad=1e11, eeg=1e5)

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    evoked_white : instance of Evoked
        The whitened evoked data.
    """
    evoked = cp.deepcopy(evoked)
    if picks is None:
        picks = pick_types(evoked.info, meg=True, eeg=True)
    W = _get_whitener_data(evoked.info, noise_cov, picks,
                           diag, rank, scalings, evoked.nave)
    evoked.data[picks] = np.sqrt(evoked.nave) * np.dot(W, evoked.data[picks])
    return evoked


@verbose
def _get_whitener_data(info, noise_cov, picks, diag=False, rank=None,
                       scalings=None, verbose=None):
    """Get whitening matrix for a set of data"""
    ch_names = [info['ch_names'][k] for k in picks]
    noise_cov = pick_channels_cov(noise_cov, include=ch_names, exclude=[])
    info = pick_info(info, picks)
    if diag:
        noise_cov = cp.deepcopy(noise_cov)
        noise_cov['data'] = np.diag(np.diag(noise_cov['data']))

    scalings = _handle_default('scalings_cov_rank', scalings)
    W = compute_whitener(noise_cov, info, rank=rank, scalings=scalings)[0]
    return W


@verbose
def _read_cov(fid, node, cov_kind, verbose=None):
    """Read a noise covariance matrix"""
    #   Find all covariance matrices
    covs = dir_tree_find(node, FIFF.FIFFB_MNE_COV)
    if len(covs) == 0:
        raise ValueError('No covariance matrices found')

    #   Is any of the covariance matrices a noise covariance
    for p in range(len(covs)):
        tag = find_tag(fid, covs[p], FIFF.FIFF_MNE_COV_KIND)

        if tag is not None and int(tag.data) == cov_kind:
            this = covs[p]

            #   Find all the necessary data
            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIM)
            if tag is None:
                raise ValueError('Covariance matrix dimension not found')
            dim = int(tag.data)

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_NFREE)
            if tag is None:
                nfree = -1
            else:
                nfree = int(tag.data)

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_METHOD)
            if tag is None:
                method = None
            else:
                method = tag.data

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_SCORE)
            if tag is None:
                score = None
            else:
                score = tag.data[0]

            tag = find_tag(fid, this, FIFF.FIFF_MNE_ROW_NAMES)
            if tag is None:
                names = []
            else:
                names = tag.data.split(':')
                if len(names) != dim:
                    raise ValueError('Number of names does not match '
                                     'covariance matrix dimension')

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV)
            if tag is None:
                tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIAG)
                if tag is None:
                    raise ValueError('No covariance matrix data found')
                else:
                    #   Diagonal is stored
                    data = tag.data
                    diagmat = True
                    logger.info('    %d x %d diagonal covariance (kind = '
                                '%d) found.' % (dim, dim, cov_kind))

            else:
                from scipy import sparse
                if not sparse.issparse(tag.data):
                    #   Lower diagonal is stored
                    vals = tag.data
                    data = np.zeros((dim, dim))
                    data[np.tril(np.ones((dim, dim))) > 0] = vals
                    data = data + data.T
                    data.flat[::dim + 1] /= 2.0
                    diagmat = False
                    logger.info('    %d x %d full covariance (kind = %d) '
                                'found.' % (dim, dim, cov_kind))
                else:
                    diagmat = False
                    data = tag.data
                    logger.info('    %d x %d sparse covariance (kind = %d)'
                                ' found.' % (dim, dim, cov_kind))

            #   Read the possibly precomputed decomposition
            tag1 = find_tag(fid, this, FIFF.FIFF_MNE_COV_EIGENVALUES)
            tag2 = find_tag(fid, this, FIFF.FIFF_MNE_COV_EIGENVECTORS)
            if tag1 is not None and tag2 is not None:
                eig = tag1.data
                eigvec = tag2.data
            else:
                eig = None
                eigvec = None

            #   Read the projection operator
            projs = _read_proj(fid, this)

            #   Read the bad channel list
            bads = read_bad_channels(fid, this)

            #   Put it together
            cov = dict(kind=cov_kind, diag=diagmat, dim=dim, names=names,
                       data=data, projs=projs, bads=bads, nfree=nfree, eig=eig,
                       eigvec=eigvec)
            if score is not None:
                cov['loglik'] = score
            if method is not None:
                cov['method'] = method

            return cov

    logger.info('    Did not find the desired covariance matrix (kind = %d)'
                % cov_kind)

    return None


def _write_cov(fid, cov):
    """Write a noise covariance matrix"""
    start_block(fid, FIFF.FIFFB_MNE_COV)

    #   Dimensions etc.
    write_int(fid, FIFF.FIFF_MNE_COV_KIND, cov['kind'])
    write_int(fid, FIFF.FIFF_MNE_COV_DIM, cov['dim'])
    if cov['nfree'] > 0:
        write_int(fid, FIFF.FIFF_MNE_COV_NFREE, cov['nfree'])

    #   Channel names
    if cov['names'] is not None and len(cov['names']) > 0:
        write_name_list(fid, FIFF.FIFF_MNE_ROW_NAMES, cov['names'])

    #   Data
    if cov['diag']:
        write_double(fid, FIFF.FIFF_MNE_COV_DIAG, cov['data'])
    else:
        # Store only lower part of covariance matrix
        dim = cov['dim']
        mask = np.tril(np.ones((dim, dim), dtype=np.bool)) > 0
        vals = cov['data'][mask].ravel()
        write_double(fid, FIFF.FIFF_MNE_COV, vals)

    #   Eigenvalues and vectors if present
    if cov['eig'] is not None and cov['eigvec'] is not None:
        write_float_matrix(fid, FIFF.FIFF_MNE_COV_EIGENVECTORS, cov['eigvec'])
        write_double(fid, FIFF.FIFF_MNE_COV_EIGENVALUES, cov['eig'])

    #   Projection operator
    if cov['projs'] is not None and len(cov['projs']) > 0:
        _write_proj(fid, cov['projs'])

    #   Bad channels
    if cov['bads'] is not None and len(cov['bads']) > 0:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, cov['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    # estimator method
    if 'method' in cov:
        write_string(fid, FIFF.FIFF_MNE_COV_METHOD, cov['method'])

    # negative log-likelihood score
    if 'loglik' in cov:
        write_double(
            fid, FIFF.FIFF_MNE_COV_SCORE, np.array(cov['loglik']))

    #   Done!
    end_block(fid, FIFF.FIFFB_MNE_COV)


def _apply_scaling_array(data, picks_list, scalings):
    """scale data type-dependently for estimation"""
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    if isinstance(scalings, dict):
        picks_dict = dict(picks_list)
        scalings = [(picks_dict[k], v) for k, v in scalings.items()
                    if k in picks_dict]
        for idx, scaling in scalings:
            data[idx, :] *= scaling  # F - order
    else:
        data *= scalings[:, np.newaxis]  # F - order


def _undo_scaling_array(data, picks_list, scalings):
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    if isinstance(scalings, dict):
        scalings = dict((k, 1. / v) for k, v in scalings.items())
    elif isinstance(scalings, np.ndarray):
        scalings = 1. / scalings
    return _apply_scaling_array(data, picks_list, scalings)


def _apply_scaling_cov(data, picks_list, scalings):
    """scalings resulting data after estimation"""
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    scales = None
    if isinstance(scalings, dict):
        n_channels = len(data)
        covinds = list(zip(*picks_list))[1]
        assert len(data) == sum(len(k) for k in covinds)
        assert list(sorted(np.concatenate(covinds))) == list(range(len(data)))
        scales = np.zeros(n_channels)
        for ch_t, idx in picks_list:
            scales[idx] = scalings[ch_t]
    elif isinstance(scalings, np.ndarray):
        if len(scalings) != len(data):
            raise ValueError('Scaling factors and data are of incompatible '
                             'shape')
        scales = scalings
    elif scalings is None:
        pass
    else:
        raise RuntimeError('Arff...')
    if scales is not None:
        assert np.sum(scales == 0.) == 0
        data *= (scales[None, :] * scales[:, None])


def _undo_scaling_cov(data, picks_list, scalings):
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    if isinstance(scalings, dict):
        scalings = dict((k, 1. / v) for k, v in scalings.items())
    elif isinstance(scalings, np.ndarray):
        scalings = 1. / scalings
    return _apply_scaling_cov(data, picks_list, scalings)


def _check_scaling_inputs(data, picks_list, scalings):
    """Aux function"""
    rescale_dict_ = dict(mag=1e15, grad=1e13, eeg=1e6)

    scalings_ = None
    if scalings == 'norm':
        scalings_ = 1. / _compute_row_norms(data)
    elif isinstance(scalings, dict):
        rescale_dict_.update(scalings)
        scalings_ = rescale_dict_
    elif isinstance(scalings, np.ndarray):
        scalings_ = scalings
    elif scalings is None:
        pass
    else:
        raise NotImplementedError("No way! That's not a rescaling "
                                  'option: %s' % scalings)
    return scalings_


def _estimate_rank_meeg_signals(data, info, scalings, tol=1e-4,
                                return_singular=False, copy=True):
    """Estimate rank for M/EEG data.

    Parameters
    ----------
    data : np.ndarray of float, shape(n_channels, n_samples)
        The M/EEG signals.
    info : mne.io.measurement_info.Info
        The measurment info.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e15, grad=1e13, eeg=1e6)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    copy : bool
        If False, values in data will be modified in-place during
        rank estimation (saves memory).

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    picks_list = _picks_by_type(info)
    _apply_scaling_array(data, picks_list, scalings)
    if data.shape[1] < data.shape[0]:
        ValueError("You've got fewer samples than channels, your "
                   "rank estimate might be inaccurate.")
    out = estimate_rank(data, tol=tol, norm=False,
                        return_singular=return_singular, copy=copy)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_array(data, picks_list, scalings)
    return out


def _estimate_rank_meeg_cov(data, info, scalings, tol=1e-4,
                            return_singular=False, copy=True):
    """Estimate rank for M/EEG data.

    Parameters
    ----------
    data : np.ndarray of float, shape (n_channels, n_channels)
        The M/EEG covariance.
    info : mne.io.measurement_info.Info
        The measurment info.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e12, grad=1e11, eeg=1e5)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.
    copy : bool
        If False, values in data will be modified in-place during
        rank estimation (saves memory).

    Returns
    -------
    rank : int
        Estimated rank of the data.
    s : array
        If return_singular is True, the singular values that were
        thresholded to determine the rank are also returned.
    """
    picks_list = _picks_by_type(info)
    scalings = _handle_default('scalings_cov_rank', scalings)
    _apply_scaling_cov(data, picks_list, scalings)
    if data.shape[1] < data.shape[0]:
        ValueError("You've got fewer samples than channels, your "
                   "rank estimate might be inaccurate.")
    out = estimate_rank(data, tol=tol, norm=False,
                        return_singular=return_singular, copy=copy)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_cov(data, picks_list, scalings)
    return out
