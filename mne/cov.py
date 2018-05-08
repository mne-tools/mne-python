# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import itertools as itt
from math import log
import os

import numpy as np
from scipy import linalg

from .io.write import start_file, end_file
from .io.proj import (make_projector, _proj_equal, activate_proj,
                      _needs_eeg_average_ref_proj, _check_projs,
                      _has_eeg_average_ref_proj)
from .io import fiff_open
from .io.pick import (pick_types, pick_channels_cov, pick_channels, pick_info,
                      _picks_by_type, _pick_data_channels,
                      _DATA_CH_TYPES_SPLIT)

from .io.constants import FIFF
from .io.meas_info import read_bad_channels, _simplify_info, create_info
from .io.proj import _read_proj, _write_proj
from .io.tag import find_tag
from .io.tree import dir_tree_find
from .io.write import (start_block, end_block, write_int, write_name_list,
                       write_double, write_float_matrix, write_string)
from .defaults import _handle_default
from .epochs import Epochs
from .event import make_fixed_length_events
from .utils import (check_fname, logger, verbose, estimate_rank,
                    _compute_row_norms, check_version, _time_mask, warn,
                    copy_function_doc_to_method_doc, _pl)
from . import viz

from .externals.six.moves import zip
from .externals.six import string_types
from .fixes import BaseEstimator, EmpiricalCovariance, _logdet


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
    """Get the slice."""
    mask = _time_mask(epochs.times, tmin, tmax, sfreq=epochs.info['sfreq'])
    tstart = np.where(mask)[0][0] if tmin is not None else None
    tend = np.where(mask)[0][-1] + 1 if tmax is not None else None
    tslice = slice(tstart, tend, None)
    return tslice


class Covariance(dict):
    """Noise covariance matrix.

    .. warning:: This class should not be instantiated directly, but
                 instead should be created using a covariance reading or
                 computation function.

    Parameters
    ----------
    data : array-like
        The data.
    names : list of str
        Channel names.
    bads : list of str
        Bad channels.
    projs : list
        Projection vectors.
    nfree : int
        Degrees of freedom.
    eig : array-like | None
        Eigenvalues.
    eigvec : array-like | None
        Eigenvectors.
    method : str | None
        The method used to compute the covariance.
    loglik : float
        The log likelihood.

    Attributes
    ----------
    data : array of shape (n_channels, n_channels)
        The covariance.
    ch_names : list of string
        List of channels' names.
    nfree : int
        Number of degrees of freedom i.e. number of time points used.
    dim : int
        The number of channels ``n_channels``.

    See Also
    --------
    compute_covariance
    compute_raw_covariance
    make_ad_hoc_cov
    read_cov
    """

    def __init__(self, data, names, bads, projs, nfree, eig=None, eigvec=None,
                 method=None, loglik=None):
        """Init of covariance."""
        diag = True if data.ndim == 1 else False
        projs = _check_projs(projs)
        self.update(data=data, dim=len(data), names=names, bads=bads,
                    nfree=nfree, eig=eig, eigvec=eigvec, diag=diag,
                    projs=projs, kind=FIFF.FIFFV_MNE_NOISE_COV)
        if method is not None:
            self['method'] = method
        if loglik is not None:
            self['loglik'] = loglik

    @property
    def data(self):
        """Numpy array of Noise covariance matrix."""
        return self['data']

    @property
    def ch_names(self):
        """Channel names."""
        return self['names']

    @property
    def nfree(self):
        """Number of degrees of freedom."""
        return self['nfree']

    def save(self, fname):
        """Save covariance matrix in a FIF file.

        Parameters
        ----------
        fname : str
            Output filename.
        """
        check_fname(fname, 'covariance', ('-cov.fif', '-cov.fif.gz',
                                          '_cov.fif', '_cov.fif.gz'))

        fid = start_file(fname)

        try:
            _write_cov(fid, self)
        except Exception:
            fid.close()
            os.remove(fname)
            raise

        end_file(fid)

    def copy(self):
        """Copy the Covariance object.

        Returns
        -------
        cov : instance of Covariance
            The copied object.
        """
        return deepcopy(self)

    def as_diag(self):
        """Set covariance to be processed as being diagonal.

        Returns
        -------
        cov : dict
            The covariance.

        Notes
        -----
        This function allows creation of inverse operators
        equivalent to using the old "--diagnoise" mne option.
        """
        if self['diag']:
            return self
        self['diag'] = True
        self['data'] = np.diag(self['data'])
        self['eig'] = None
        self['eigvec'] = None
        return self

    def __repr__(self):  # noqa: D105
        if self.data.ndim == 2:
            s = 'size : %s x %s' % self.data.shape
        else:  # ndim == 1
            s = 'diagonal : %s' % self.data.size
        s += ", n_samples : %s" % self.nfree
        s += ", data : %s" % self.data
        return "<Covariance  |  %s>" % s

    def __add__(self, cov):
        """Add Covariance taking into account number of degrees of freedom."""
        _check_covs_algebra(self, cov)
        this_cov = cov.copy()
        this_cov['data'] = (((this_cov['data'] * this_cov['nfree']) +
                             (self['data'] * self['nfree'])) /
                            (self['nfree'] + this_cov['nfree']))
        this_cov['nfree'] += self['nfree']

        this_cov['bads'] = list(set(this_cov['bads']).union(self['bads']))

        return this_cov

    def __iadd__(self, cov):
        """Add Covariance taking into account number of degrees of freedom."""
        _check_covs_algebra(self, cov)
        self['data'][:] = (((self['data'] * self['nfree']) +
                            (cov['data'] * cov['nfree'])) /
                           (self['nfree'] + cov['nfree']))
        self['nfree'] += cov['nfree']

        self['bads'] = list(set(self['bads']).union(cov['bads']))

        return self

    @verbose
    @copy_function_doc_to_method_doc(viz.misc.plot_cov)
    def plot(self, info, exclude=[], colorbar=True, proj=False, show_svd=True,
             show=True, verbose=None):
        return viz.misc.plot_cov(self, info, exclude, colorbar, proj, show_svd,
                                 show, verbose)


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
    verbose : bool, str, int, or None (default None)
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    cov : Covariance
        The noise covariance matrix.

    See Also
    --------
    write_cov, compute_covariance, compute_raw_covariance
    """
    check_fname(fname, 'covariance', ('-cov.fif', '-cov.fif.gz',
                                      '_cov.fif', '_cov.fif.gz'))
    f, tree = fiff_open(fname)[:2]
    with f as fid:
        return Covariance(**_read_cov(fid, tree, FIFF.FIFFV_MNE_NOISE_COV,
                                      limited=True))


###############################################################################
# Estimate from data

@verbose
def make_ad_hoc_cov(info, std=None, verbose=None):
    """Create an ad hoc noise covariance.

    Parameters
    ----------
    info : instance of Info
        Measurement info.
    std : dict of float | None
        Standard_deviation of the diagonal elements. If dict, keys should be
        `grad` for gradiometers, `mag` for magnetometers and `eeg` for EEG
        channels. If None, default values will be used (see Notes).
    verbose : bool, str, int, or None (default None)
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    cov : instance of Covariance
        The ad hoc diagonal noise covariance for the M/EEG data channels.

    Notes
    -----
    The default noise values are 5 fT/cm, 20 fT, and 0.2 uV for gradiometers,
    magnetometers, and EEG channels respectively.

    .. versionadded:: 0.9.0
    """
    picks = pick_types(info, meg=True, eeg=True, exclude=())
    std = _handle_default('noise_std', std)

    data = np.zeros(len(picks))
    for meg, eeg, val in zip(('grad', 'mag', False), (False, False, True),
                             (std['grad'], std['mag'], std['eeg'])):
        these_picks = pick_types(info, meg=meg, eeg=eeg)
        data[np.searchsorted(picks, these_picks)] = val * val
    ch_names = [info['ch_names'][pick] for pick in picks]
    return Covariance(data, ch_names, info['bads'], info['projs'], nfree=0)


def _check_n_samples(n_samples, n_chan):
    """Check to see if there are enough samples for reliable cov calc."""
    n_samples_min = 10 * (n_chan + 1) // 2
    if n_samples <= 0:
        raise ValueError('No samples found to compute the covariance matrix')
    if n_samples < n_samples_min:
        warn('Too few samples (required : %d got : %d), covariance '
             'estimate may be unreliable' % (n_samples_min, n_samples))


@verbose
def compute_raw_covariance(raw, tmin=0, tmax=None, tstep=0.2, reject=None,
                           flat=None, picks=None, method='empirical',
                           method_params=None, cv=3, scalings=None, n_jobs=1,
                           return_estimators=False, reject_by_annotation=True,
                           verbose=None):
    """Estimate noise covariance matrix from a continuous segment of raw data.

    It is typically useful to estimate a noise covariance from empty room
    data or time intervals before starting the stimulation.

    Parameters
    ----------
    raw : instance of Raw
        Raw data
    tmin : float
        Beginning of time interval in seconds. Defaults to 0.
    tmax : float | None (default None)
        End of time interval in seconds. If None (default), use the end of the
        recording.
    tstep : float (default 0.2)
        Length of data chunks for artefact rejection in seconds.
        Can also be None to use a single epoch of (tmax - tmin)
        duration. This can use a lot of memory for large ``Raw``
        instances.
    reject : dict | None (default None)
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

    flat : dict | None (default None)
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    picks : array-like of int | None (default None)
        Indices of channels to include (if None, data channels are used).
    method : str | list | None (default 'empirical')
        The method used for covariance estimation.
        See :func:`mne.compute_covariance`.

        .. versionadded:: 0.12
    method_params : dict | None (default None)
        Additional parameters to the estimation procedure.
        See :func:`mne.compute_covariance`.

        .. versionadded:: 0.12
    cv : int | sklearn model_selection object (default 3)
        The cross validation method. Defaults to 3, which will
        internally trigger by default :class:`sklearn.model_selection.KFold`
        with 3 splits.

        .. versionadded:: 0.12
    scalings : dict | None (default None)
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale magnetometers and gradiometers
        at the same unit.

        .. versionadded:: 0.12
    n_jobs : int (default 1)
        Number of jobs to run in parallel.

        .. versionadded:: 0.12
    return_estimators : bool (default False)
        Whether to return all estimators or the best. Only considered if
        method equals 'auto' or is a list of str. Defaults to False

        .. versionadded:: 0.12
    reject_by_annotation : bool
        Whether to reject based on annotations. If True (default), epochs
        overlapping with segments whose description begins with ``'bad'`` are
        rejected. If False, no rejection based on annotations is performed.

        .. versionadded:: 0.14.0
    verbose : bool | str | int | None (default None)
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    cov : instance of Covariance | list
        The computed covariance. If method equals 'auto' or is a list of str
        and return_estimators equals True, a list of covariance estimators is
        returned (sorted by log-likelihood, from high to low, i.e. from best
        to worst).

    See Also
    --------
    compute_covariance : Estimate noise covariance matrix from epochs

    Notes
    -----
    This function will:

    1. Partition the data into evenly spaced, equal-length epochs.
    2. Load them into memory.
    3. Subtract the mean across all time points and epochs for each channel.
    4. Process the :class:`Epochs` by :func:`compute_covariance`.

    This will produce a slightly different result compared to using
    :func:`make_fixed_length_events`, :class:`Epochs`, and
    :func:`compute_covariance` directly, since that would (with the recommended
    baseline correction) subtract the mean across time *for each epoch*
    (instead of across epochs) for each channel.
    """
    tmin = 0. if tmin is None else float(tmin)
    tmax = raw.times[-1] if tmax is None else float(tmax)
    tstep = tmax - tmin if tstep is None else float(tstep)
    tstep_m1 = tstep - 1. / raw.info['sfreq']  # inclusive!
    events = make_fixed_length_events(raw, 1, tmin, tmax, tstep)
    logger.info('Using up to %s segment%s' % (len(events), _pl(events)))

    # don't exclude any bad channels, inverses expect all channels present
    if picks is None:
        # Need to include all channels e.g. if eog rejection is to be used
        picks = np.arange(raw.info['nchan'])
        pick_mask = np.in1d(
            picks, _pick_data_channels(raw.info, with_ref_meg=False))
    else:
        pick_mask = slice(None)
    epochs = Epochs(raw, events, 1, 0, tstep_m1, baseline=None,
                    picks=picks, reject=reject, flat=flat, verbose=False,
                    preload=False, proj=False,
                    reject_by_annotation=reject_by_annotation)
    if method is None:
        method = 'empirical'
    if isinstance(method, string_types) and method == 'empirical':
        # potentially *much* more memory efficient to do it the iterative way
        picks = picks[pick_mask]
        data = 0
        n_samples = 0
        mu = 0
        # Read data in chunks
        for raw_segment in epochs:
            raw_segment = raw_segment[pick_mask]
            mu += raw_segment.sum(axis=1)
            data += np.dot(raw_segment, raw_segment.T)
            n_samples += raw_segment.shape[1]
        _check_n_samples(n_samples, len(picks))
        data -= mu[:, None] * (mu[None, :] / n_samples)
        data /= (n_samples - 1.0)
        logger.info("Number of samples used : %d" % n_samples)
        logger.info('[done]')
        ch_names = [raw.info['ch_names'][k] for k in picks]
        bads = [b for b in raw.info['bads'] if b in ch_names]
        return Covariance(data, ch_names, bads, raw.info['projs'],
                          nfree=n_samples)
    del picks, pick_mask

    # This makes it equivalent to what we used to do (and do above for
    # empirical mode), treating all epochs as if they were a single long one
    epochs.load_data()
    ch_means = epochs._data.mean(axis=0).mean(axis=1)
    epochs._data -= ch_means[np.newaxis, :, np.newaxis]
    # fake this value so there are no complaints from compute_covariance
    epochs.baseline = (None, None)
    return compute_covariance(epochs, keep_sample_mean=True, method=method,
                              method_params=method_params, cv=cv,
                              scalings=scalings, n_jobs=n_jobs,
                              return_estimators=return_estimators)


def _check_method_params(method, method_params, keep_sample_mean=True,
                         name='method', allow_auto=True):
    """Check that method and method_params are usable."""
    accepted_methods = ('auto', 'empirical', 'diagonal_fixed', 'ledoit_wolf',
                        'oas', 'shrunk', 'pca', 'factor_analysis', 'shrinkage')
    _method_params = {
        'empirical': {'store_precision': False, 'assume_centered': True},
        'diagonal_fixed': {'grad': 0.01, 'mag': 0.01, 'eeg': 0.,
                           'seeg': 0., 'ecog': 0.,
                           'hbo': 0., 'hbr': 0.,
                           'store_precision': False, 'assume_centered': True},
        'ledoit_wolf': {'store_precision': False, 'assume_centered': True},
        'oas': {'store_precision': False, 'assume_centered': True},
        'shrinkage': {'shrinkage': 0.1, 'store_precision': False,
                      'assume_centered': True},
        'shrunk': {'shrinkage': np.logspace(-4, 0, 30),
                   'store_precision': False, 'assume_centered': True},
        'pca': {'iter_n_components': None},
        'factor_analysis': {'iter_n_components': None}
    }

    for ch_type in _DATA_CH_TYPES_SPLIT:
        if ch_type not in _method_params['diagonal_fixed']:
            _method_params['diagonal_fixed'][ch_type] = 0.0

    if isinstance(method_params, dict):
        for key, values in method_params.items():
            if key not in _method_params:
                raise ValueError('key (%s) must be "%s"' %
                                 (key, '" or "'.join(_method_params)))

            _method_params[key].update(method_params[key])
        shrinkage = method_params.get('shrinkage', {}).get('shrinkage', 0.1)
        if not 0 <= shrinkage <= 1:
            raise ValueError('shrinkage must be between 0 and 1, got %s'
                             % (shrinkage,))

    if method is None:
        method = ['empirical']
    elif method == 'auto' and allow_auto:
        method = ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']

    if not isinstance(method, (list, tuple)):
        method = [method]

    if not all(k in accepted_methods for k in method):
        raise ValueError(
            'Invalid {name} ({method}). Accepted values (individually or '
            'in a list) are any of "{accepted_methods}" or None.'.format(
                name=name, method=method, accepted_methods=accepted_methods))

    if not keep_sample_mean:
        if len(method) != 1 or 'empirical' not in method:
            raise ValueError('`keep_sample_mean=False` is only supported'
                             'with %s="empirical"' % (name,))
        for p, v in _method_params.items():
            if v.get('assume_centered', None) is False:
                raise ValueError('`assume_centered` must be True'
                                 ' if `keep_sample_mean` is False')
    return method, _method_params


@verbose
def compute_covariance(epochs, keep_sample_mean=True, tmin=None, tmax=None,
                       projs=None, method='empirical', method_params=None,
                       cv=3, scalings=None, n_jobs=1, return_estimators=False,
                       on_mismatch='raise', verbose=None):
    """Estimate noise covariance matrix from epochs.

    The noise covariance is typically estimated on pre-stimulus periods
    when the stimulus onset is defined from events.

    If the covariance is computed for multiple event types (events
    with different IDs), the following two options can be used and combined:

        1. either an Epochs object for each event type is created and
           a list of Epochs is passed to this function.
        2. an Epochs object is created for multiple events and passed
           to this function.

    Parameters
    ----------
    epochs : instance of Epochs, or a list of Epochs objects
        The epochs.
    keep_sample_mean : bool (default True)
        If False, the average response over epochs is computed for
        each event type and subtracted during the covariance
        computation. This is useful if the evoked response from a
        previous stimulus extends into the baseline period of the next.
        Note. This option is only implemented for method='empirical'.
    tmin : float | None (default None)
        Start time for baseline. If None start at first sample.
    tmax : float | None (default None)
        End time for baseline. If None end at last sample.
    projs : list of Projection | None (default None)
        List of projectors to use in covariance calculation, or None
        to indicate that the projectors from the epochs should be
        inherited. If None, then projectors from all epochs must match.
    method : str | list | None (default 'empirical')
        The method used for covariance estimation. If 'empirical' (default),
        the sample covariance will be computed. A list can be passed to
        perform estimates using multiple methods.
        If 'auto' or a list of methods, the best estimator will be determined
        based on log-likelihood and cross-validation on unseen data as
        described in [1]_. Valid methods are:

        * ``'empirical'``: the empirical or sample covariance (default)
        * ``'diagonal_fixed'``: a diagonal regularization based on
          channel types as in :func:`mne.cov.regularize`.
        * ``'shrinkage'`` : covariance estimator with fixed shrinkage.

          .. versionadded:: 0.16
        * ``'ledoit_wolf'``: the Ledoit-Wolf estimator, which uses an
          empirical formula for the optimal shrinkage value [2]_.
        * ``'oas'``: the OAS estimator [5]_, which uses a different
          empricial formula for the optimal shrinkage value.

          .. versionadded:: 0.16
        * ``'shrunk'``: like 'ledoit_wolf', but with cross-validation
          for optimal alpha.
        * ``'pca'``: probabilistic PCA with low rank [3]_.
        * ``'factor_analysis'``: factor analysis with low rank [4]_.

        If ``'auto'``, this expands to::

             ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']

        The ``'auto'`` mode is not recommended if there are many
        segments of data, since computation can take a long time.

        .. versionadded:: 0.9.0
    method_params : dict | None (default None)
        Additional parameters to the estimation procedure. Only considered if
        method is not None. Keys must correspond to the value(s) of ``method``.
        If None (default), expands to the following (with the addition of
        ``{'store_precision': False, 'assume_centered': True} for all methods
        except ``'factor_analysis'`` and ``'pca'``)::

            {'diagonal_fixed': {'grad': 0.01, 'mag': 0.01, 'eeg': 0.0},
             'shrinkage': {'shrikage': 0.1},
             'shrunk': {'shrinkage': np.logspace(-4, 0, 30)},
             'pca': {'iter_n_components': None},
             'factor_analysis': {'iter_n_components': None}}

    cv : int | sklearn model_selection object (default 3)
        The cross validation method. Defaults to 3, which will
        internally trigger by default :class:`sklearn.model_selection.KFold`
        with 3 splits.
    scalings : dict | None (default None)
        Defaults to ``dict(mag=1e15, grad=1e13, eeg=1e6)``.
        These defaults will scale data to roughly the same order of
        magnitude.
    n_jobs : int (default 1)
        Number of jobs to run in parallel.
    return_estimators : bool (default False)
        Whether to return all estimators or the best. Only considered if
        method equals 'auto' or is a list of str. Defaults to False
    on_mismatch : str
        What to do when the MEG<->Head transformations do not match between
        epochs. If "raise" (default) an error is raised, if "warn" then a
        warning is emitted, if "ignore" then nothing is printed. Having
        mismatched transforms can in some cases lead to unexpected or
        unstable results in covariance calculation, e.g. when data
        have been processed with Maxwell filtering but not transformed
        to the same head position.
    verbose : bool | str | int | or None (default None)
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    cov : instance of Covariance | list
        The computed covariance. If method equals 'auto' or is a list of str
        and return_estimators equals True, a list of covariance estimators is
        returned (sorted by log-likelihood, from high to low, i.e. from best
        to worst).

    See Also
    --------
    compute_raw_covariance : Estimate noise covariance from raw data

    Notes
    -----
    Baseline correction or sufficient high-passing should be used
    when creating the :class:`Epochs` to ensure that the data are zero mean,
    otherwise the computed covariance matrix will be inaccurate.

    For multiple event types, it is also possible to create a
    single :class:`Epochs` object with events obtained using
    :func:`mne.merge_events`. However, the resulting covariance matrix
    will only be correct if ``keep_sample_mean is True``.

    The covariance can be unstable if the number of samples is small.
    In that case it is common to regularize the covariance estimate.
    The ``method`` parameter allows to regularize the covariance in an
    automated way. It also allows to select between different alternative
    estimation algorithms which themselves achieve regularization.
    Details are described in [1]_.

    ``'ledoit_wolf'`` and ``'pca'`` are similar to ``'shrunk'`` and
    ``'factor_analysis'``, respectively, except that they use
    cross validation (which is useful when samples are correlated, which
    is often the case for M/EEG data). The former two are not included in
    the ``'auto'`` mode to avoid redundancy.

    For more information on the advanced estimation methods, see
    :ref:`the sklearn manual <sklearn:covariance>`.

    References
    ----------
    .. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
           covariance estimation and spatial whitening of MEG and EEG
           signals, vol. 108, 328-342, NeuroImage.
    .. [2] Ledoit, O., Wolf, M., (2004). A well-conditioned estimator for
           large-dimensional covariance matrices. Journal of Multivariate
           Analysis 88 (2), 365 - 411.
    .. [3] Tipping, M. E., Bishop, C. M., (1999). Probabilistic principal
           component analysis. Journal of the Royal Statistical Society:
           Series B (Statistical Methodology) 61 (3), 611 - 622.
    .. [4] Barber, D., (2012). Bayesian reasoning and machine learning.
           Cambridge University Press., Algorithm 21.1
    .. [5] Chen et al. (2010). Shrinkage Algorithms for MMSE Covariance
           Estimation. IEEE Trans. on Sign. Proc., Volume 58, Issue 10,
           October 2010.
    """
    # scale to natural unit for best stability with MEG/EEG
    scalings = _check_scalings_user(scalings)
    method, _method_params = _check_method_params(
        method, method_params, keep_sample_mean)
    del method_params

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
    if any(epochs_t.baseline is None and epochs_t.info['highpass'] < 0.5 and
           keep_sample_mean for epochs_t in epochs):
        warn('Epochs are not baseline corrected, covariance '
             'matrix may be inaccurate')

    orig = epochs[0].info['dev_head_t']
    if not isinstance(on_mismatch, string_types) or \
            on_mismatch not in ['raise', 'warn', 'ignore']:
        raise ValueError('on_mismatch must be "raise", "warn", or "ignore", '
                         'got %s' % on_mismatch)
    for ei, epoch in enumerate(epochs):
        epoch.info._check_consistency()
        if (orig is None) != (epoch.info['dev_head_t'] is None) or \
                (orig is not None and not
                 np.allclose(orig['trans'],
                             epoch.info['dev_head_t']['trans'])):
            msg = ('MEG<->Head transform mismatch between epochs[0]:\n%s\n\n'
                   'and epochs[%s]:\n%s'
                   % (orig, ei, epoch.info['dev_head_t']))
            if on_mismatch == 'raise':
                raise ValueError(msg)
            elif on_mismatch == 'warn':
                warn(msg)

    bads = epochs[0].info['bads']
    if projs is None:
        projs = epochs[0].info['projs']
        # make sure Epochs are compatible
        for epochs_t in epochs[1:]:
            if epochs_t.proj != epochs[0].proj:
                raise ValueError('Epochs must agree on the use of projections')
            for proj_a, proj_b in zip(epochs_t.info['projs'], projs):
                if not _proj_equal(proj_a, proj_b):
                    raise ValueError('Epochs must have same projectors')
    projs = _check_projs(projs)
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

    if not keep_sample_mean:
        # prepare mean covs
        n_epoch_types = len(epochs)
        data_mean = [0] * n_epoch_types
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

    if info['comps']:
        info['comps'] = []
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
    cov_data = _compute_covariance_auto(
        epochs, method=method, method_params=_method_params, info=info,
        cv=cv, n_jobs=n_jobs, stop_early=True, picks_list=picks_list,
        scalings=scalings)

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
        cov = Covariance(data.pop('data'), ch_names, info['bads'], projs,
                         nfree=n_samples_tot)
        logger.info('Number of samples used : %d' % n_samples_tot)
        logger.info('[done]')

        # add extra info
        cov.update(method=this_method, **data)
        covs.append(cov)
    covs.sort(key=lambda c: c['loglik'], reverse=True)

    if len(covs) > 1:
        msg = ['log-likelihood on unseen data (descending order):']
        for c in covs:
            msg.append('%s: %0.3f' % (c['method'], c['loglik']))
        logger.info('\n   '.join(msg))
        if return_estimators:
            out = covs
        else:
            out = covs[0]
            logger.info('selecting best estimator: {0}'.format(out['method']))
    else:
        out = covs[0]

    return out


def _check_scalings_user(scalings):
    if isinstance(scalings, dict):
        for k, v in scalings.items():
            if k not in ('mag', 'grad', 'eeg'):
                raise ValueError('The keys in `scalings` must be "mag" or'
                                 '"grad" or "eeg". You gave me: %s' % k)
    elif scalings is not None and not isinstance(scalings, np.ndarray):
        raise TypeError('scalings must be a dict, ndarray, or None, got %s'
                        % type(scalings))
    scalings = _handle_default('scalings', scalings)
    return scalings


def _compute_covariance_auto(data, method, info, method_params, cv,
                             scalings, n_jobs, stop_early, picks_list):
    """Compute covariance auto mode."""
    # rescale to improve numerical stability
    _apply_scaling_array(data.T, picks_list=picks_list, scalings=scalings)
    estimator_cov_info = list()
    msg = 'Estimating covariance using %s'

    ok_sklearn = check_version('sklearn', '0.15')
    if not ok_sklearn and (len(method) != 1 or method[0] != 'empirical'):
        raise ValueError('scikit-learn is not installed, `method` must be '
                         '`empirical`')

    for this_method in method:
        data_ = data.copy()
        name = this_method.__name__ if callable(this_method) else this_method
        logger.info(msg % name.upper())
        mp = method_params[this_method]
        _info = None

        if this_method == 'empirical':
            est = EmpiricalCovariance(**mp)
            est.fit(data_)
            estimator_cov_info.append((est, est.covariance_, _info))
            del est

        elif this_method == 'diagonal_fixed':
            est = _RegCovariance(info=info, **mp)
            est.fit(data_)
            estimator_cov_info.append((est, est.covariance_, _info))
            del est

        elif this_method == 'ledoit_wolf':
            from sklearn.covariance import LedoitWolf
            shrinkages = []
            lw = LedoitWolf(**mp)

            for ch_type, picks in picks_list:
                lw.fit(data_[:, picks])
                shrinkages.append((ch_type, lw.shrinkage_, picks))
            sc = _ShrunkCovariance(shrinkage=shrinkages, **mp)
            sc.fit(data_)
            estimator_cov_info.append((sc, sc.covariance_, _info))
            del lw, sc

        elif this_method == 'oas':
            from sklearn.covariance import OAS
            shrinkages = []
            oas = OAS(**mp)

            for ch_type, picks in picks_list:
                oas.fit(data_[:, picks])
                shrinkages.append((ch_type, oas.shrinkage_, picks))
            sc = _ShrunkCovariance(shrinkage=shrinkages, **mp)
            sc.fit(data_)
            estimator_cov_info.append((sc, sc.covariance_, _info))
            del oas, sc

        elif this_method == 'shrinkage':
            sc = _ShrunkCovariance(**mp)
            sc.fit(data_)
            estimator_cov_info.append((sc, sc.covariance_, _info))
            del sc

        elif this_method == 'shrunk':
            try:
                from sklearn.model_selection import GridSearchCV
            except Exception:  # support sklearn < 0.18
                from sklearn.grid_search import GridSearchCV
            from sklearn.covariance import ShrunkCovariance
            shrinkage = mp.pop('shrinkage')
            tuned_parameters = [{'shrinkage': shrinkage}]
            shrinkages = []
            gs = GridSearchCV(ShrunkCovariance(**mp),
                              tuned_parameters, cv=cv)
            for ch_type, picks in picks_list:
                gs.fit(data_[:, picks])
                shrinkages.append((ch_type, gs.best_estimator_.shrinkage,
                                   picks))
            shrinkages = [c[0] for c in zip(shrinkages)]
            sc = _ShrunkCovariance(shrinkage=shrinkages, **mp)
            sc.fit(data_)
            estimator_cov_info.append((sc, sc.covariance_, _info))
            del shrinkage, sc

        elif this_method == 'pca':
            pca, _info = _auto_low_rank_model(
                data_, this_method, n_jobs=n_jobs, method_params=mp, cv=cv,
                stop_early=stop_early)
            pca.fit(data_)
            estimator_cov_info.append((pca, pca.get_covariance(), _info))
            del pca

        elif this_method == 'factor_analysis':
            fa, _info = _auto_low_rank_model(
                data_, this_method, n_jobs=n_jobs, method_params=mp, cv=cv,
                stop_early=stop_early)
            fa.fit(data_)
            estimator_cov_info.append((fa, fa.get_covariance(), _info))
            del fa
        else:
            raise ValueError('Oh no! Your estimator does not have'
                             ' a .fit method')
        logger.info('Done.')

    estimators, _, _ = zip(*estimator_cov_info)
    if len(method) > 1:
        logger.info('Using cross-validation to select the best estimator.')
        logliks = np.array(
            [_cross_val(data, e, cv, n_jobs) for e in estimators])
    else:
        logliks = [None]

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


def _gaussian_loglik_scorer(est, X, y=None):
    """Compute the Gaussian log likelihood of X under the model in est."""
    # compute empirical covariance of the test set
    precision = est.get_precision()
    n_samples, n_features = X.shape
    log_like = np.zeros(n_samples)
    log_like = -.5 * (X * (np.dot(X, precision))).sum(axis=1)
    log_like -= .5 * (n_features * log(2. * np.pi) - _logdet(precision))
    out = np.mean(log_like)
    return out


def _cross_val(data, est, cv, n_jobs):
    """Compute cross validation."""
    try:
        from sklearn.model_selection import cross_val_score
    except ImportError:
        # XXX support sklearn < 0.18
        from sklearn.cross_validation import cross_val_score
    return np.mean(cross_val_score(est, data, cv=cv, n_jobs=n_jobs,
                                   scoring=_gaussian_loglik_scorer))


def _auto_low_rank_model(data, mode, n_jobs, method_params, cv,
                         stop_early=True, verbose=None):
    """Compute latent variable models."""
    method_params = deepcopy(method_params)
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
    max_n = max(list(deepcopy(iter_n_components)))
    if max_n > data.shape[1]:
        warn('You are trying to estimate %i components on matrix '
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


###############################################################################
# Sklearn Estimators

class _RegCovariance(BaseEstimator):
    """Aux class."""

    def __init__(self, info, grad=0.01, mag=0.01, eeg=0., seeg=0., ecog=0.,
                 hbo=0., hbr=0., store_precision=False,
                 assume_centered=False):
        self.info = info
        # For sklearn compat, these cannot (easily?) be combined into
        # a single dictionary
        self.grad = grad
        self.mag = mag
        self.eeg = eeg
        self.seeg = seeg
        self.ecog = ecog
        self.hbo = hbo
        self.hbr = hbr
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def fit(self, X):
        """Fit covariance model with classical diagonal regularization."""
        from sklearn.covariance import EmpiricalCovariance
        self.estimator_ = EmpiricalCovariance(
            store_precision=self.store_precision,
            assume_centered=self.assume_centered)

        self.covariance_ = self.estimator_.fit(X).covariance_
        self.covariance_ = 0.5 * (self.covariance_ + self.covariance_.T)
        cov_ = Covariance(
            data=self.covariance_, names=self.info['ch_names'],
            bads=self.info['bads'], projs=self.info['projs'],
            nfree=len(self.covariance_))
        cov_ = regularize(
            cov_, self.info, proj=False, exclude='bads',
            grad=self.grad, mag=self.mag, eeg=self.eeg,
            ecog=self.ecog, seeg=self.seeg,
            hbo=self.hbo, hbr=self.hbr)  # ~proj == important!!
        self.estimator_.covariance_ = self.covariance_ = cov_.data
        return self

    def score(self, X_test, y=None):
        """Delegate call to modified EmpiricalCovariance instance."""
        return self.estimator_.score(X_test, y=y)

    def get_precision(self):
        """Delegate call to modified EmpiricalCovariance instance."""
        return self.estimator_.get_precision()


class _ShrunkCovariance(BaseEstimator):
    """Aux class."""

    def __init__(self, store_precision, assume_centered,
                 shrinkage=0.1):

        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.shrinkage = shrinkage

    def fit(self, X):
        """Fit covariance model with oracle shrinkage regularization."""
        from sklearn.covariance import shrunk_covariance
        from sklearn.covariance import EmpiricalCovariance
        self.estimator_ = EmpiricalCovariance(
            store_precision=self.store_precision,
            assume_centered=self.assume_centered)

        cov = self.estimator_.fit(X).covariance_

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

        self.estimator_.covariance_ = self.covariance_ = cov
        return self

    def score(self, X_test, y=None):
        """Delegate to modified EmpiricalCovariance instance."""
        from sklearn.covariance import empirical_covariance, log_likelihood
        # compute empirical covariance of the test set
        test_cov = empirical_covariance(X_test - self.estimator_.location_,
                                        assume_centered=True)
        if np.any(self.zero_cross_cov_):
            test_cov[self.zero_cross_cov_] = 0.
        res = log_likelihood(test_cov, self.estimator_.get_precision())
        return res

    def get_precision(self):
        """Delegate to modified EmpiricalCovariance instance."""
        return self.estimator_.get_precision()


###############################################################################
# Writing

def write_cov(fname, cov):
    """Write a noise covariance matrix.

    Parameters
    ----------
    fname : string
        The name of the file. It should end with -cov.fif or -cov.fif.gz.
    cov : Covariance
        The noise covariance matrix

    See Also
    --------
    read_cov
    """
    cov.save(fname)


###############################################################################
# Prepare for inverse modeling

def _unpack_epochs(epochs):
    """Aux Function."""
    if len(epochs.event_id) > 1:
        epochs = [epochs[k] for k in epochs.event_id]
    else:
        epochs = [epochs]

    return epochs


def _get_ch_whitener(A, pca, ch_type, rank):
    """Get whitener params for a set of channels."""
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


def _get_whitener(noise_cov, info, ch_names, rank, pca, scalings=None):
    #
    #   Handle noise cov
    #
    noise_cov = prepare_noise_cov(noise_cov, info, ch_names, rank)
    n_chan = len(ch_names)

    #   Omit the zeroes due to projection
    eig = noise_cov['eig']
    nzero = (eig > 0)
    n_nzero = np.sum(nzero)

    if pca:
        #   Rows of eigvec are the eigenvectors
        whitener = noise_cov['eigvec'][nzero] / np.sqrt(eig[nzero])[:, None]
        logger.info('Reducing data rank to %d' % n_nzero)
    else:
        whitener = np.zeros((n_chan, n_chan), dtype=np.float)
        whitener[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
        #   Rows of eigvec are the eigenvectors
        whitener = np.dot(whitener, noise_cov['eigvec'])

    logger.info('Total rank is %d' % n_nzero)

    return whitener, noise_cov, n_nzero


@verbose
def prepare_noise_cov(noise_cov, info, ch_names, rank=None,
                      scalings=None, verbose=None):
    """Prepare noise covariance matrix.

    Parameters
    ----------
    noise_cov : instance of Covariance
        The noise covariance to process.
    info : dict
        The measurement info (used to get channel types and bad channels).
    ch_names : list
        The channel names to be considered.
    rank : None | int | dict (default None)
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg', 'meg' or any other
        data channel type such as 'seeg' or 'ecog' can be used
        to specify the rank for each modality.
    scalings : dict | None
        Data will be rescaled before rank estimation to improve accuracy.
        If dict, it will override the following dict (default if None)::

            dict(mag=1e12, grad=1e11, eeg=1e5)

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    cov : instance of Covariance
        A copy of the covariance with the good channels subselected
        and parameters updated.
    """
    info = info.copy()
    noise_cov_idx = [noise_cov.ch_names.index(c) for c in ch_names]
    n_chan = len(ch_names)
    if not noise_cov['diag']:
        C = noise_cov.data[np.ix_(noise_cov_idx, noise_cov_idx)]
    else:
        C = np.diag(noise_cov.data[noise_cov_idx])

    scalings = _handle_default('scalings_cov_rank', scalings)

    # Create the projection operator
    proj, ncomp, _ = make_projector(info['projs'] + noise_cov['projs'],
                                    ch_names)
    if ncomp > 0:
        logger.info('    Created an SSP operator (subspace dimension = %d)'
                    % ncomp)
        C = np.dot(proj, np.dot(C, proj.T))

    picks_dict = dict(_picks_by_type(info, meg_combined=True,
                                     ref_meg=False, exclude='bads'))

    ch_names_dict = dict()
    out_idx_type = dict()
    for ch_type, ch_picks in picks_dict.items():
        ch_names_dict[ch_type] = [info['chs'][k]['ch_name'] for k in ch_picks]
        out_idx_type[ch_type] = [k for k in range(len(C))
                                 if ch_names[k] in ch_names_dict[ch_type]]

    del picks_dict

    # re-index based on ch_names order
    ch_names_type = dict()
    info_picks = dict()
    for ch_type, ch_picks in out_idx_type.items():
        ch_names_type[ch_type] = [ch_names[k] for k in ch_picks]
        info_picks[ch_type] = []
        if ch_names_type[ch_type]:
            # as pick_channels return full list if include is []
            info_picks[ch_type] = pick_channels(info['ch_names'],
                                                ch_names_type[ch_type])

        assert (len(info_picks[ch_type]) ==
                len(ch_names_type[ch_type]) ==
                len(out_idx_type[ch_type]))

    assert(sum((len(v) for k, v in out_idx_type.items())) == n_chan)

    eigvec = np.zeros((n_chan, n_chan))
    eig = np.zeros(n_chan)
    has_type = dict()
    for ch_type, ch_picks in out_idx_type.items():
        has_type[ch_type] = len(ch_picks) > 0

    # Get the specified noise covariance rank
    ranks_type = {ch_type: None for ch_type in out_idx_type}
    if rank is not None:
        if isinstance(rank, dict):
            for ch_type in out_idx_type:
                ranks_type[ch_type] = rank.get(ch_type, None)
        else:
            ranks_type['meg'] = int(rank)

    # XXX cov objects don't include compensation_grade
    # so we can't check that the supplied cov matches info
    if info['comps']:
        info['comps'] = []

    for ch_type, this_has in has_type.items():
        if not this_has:
            continue
        this_picks = out_idx_type[ch_type]
        this_C = C[np.ix_(this_picks, this_picks)]
        this_info = pick_info(_simplify_info(info), info_picks[ch_type],
                              copy=False)
        this_rank = ranks_type[ch_type]
        if ranks_type[ch_type] is None:
            this_rank = _estimate_rank_meeg_cov(this_C, this_info, scalings)
        eig[this_picks], eigvec[np.ix_(this_picks, this_picks)] = \
            _get_ch_whitener(this_C, False, ch_type.upper(), this_rank)

        # XXX : also handle ref for sEEG and ECoG
        if ch_type == 'eeg' and _needs_eeg_average_ref_proj(info) and not \
                _has_eeg_average_ref_proj(noise_cov['projs']):
            warn('No average EEG reference present in info["projs"], '
                 'covariance may be adversely affected. Consider recomputing '
                 'covariance using with an average eeg reference projector '
                 'added.')

    noise_cov = Covariance(
        data=C, names=ch_names, bads=list(noise_cov['bads']),
        projs=deepcopy(noise_cov['projs']),
        nfree=noise_cov['nfree'], eig=eig, eigvec=eigvec,
        method=noise_cov.get('method', None),
        loglik=noise_cov.get('loglik', None))
    return noise_cov


def regularize(cov, info, mag=0.1, grad=0.1, eeg=0.1, exclude='bads',
               proj=True, seeg=0.1, ecog=0.1, hbo=0.1, hbr=0.1,
               verbose=None):
    """Regularize noise covariance matrix.

    This method works by adding a constant to the diagonal for each
    channel type separately. Special care is taken to keep the
    rank of the data constant.

    .. note:: This function is kept for reasons of backward-compatibility.
              Please consider explicitly using the ``method`` parameter in
              :func:`mne.compute_covariance` to directly combine estimation
              with regularization in a data-driven fashion.
              See the `faq <http://martinos.org/mne/dev/faq.html#how-should-i-regularize-the-covariance-matrix>`_
              for more information.

    Parameters
    ----------
    cov : Covariance
        The noise covariance matrix.
    info : dict
        The measurement info (used to get channel types and bad channels).
    mag : float (default 0.1)
        Regularization factor for MEG magnetometers.
    grad : float (default 0.1)
        Regularization factor for MEG gradiometers.
    eeg : float (default 0.1)
        Regularization factor for EEG.
    exclude : list | 'bads' (default 'bads')
        List of channels to mark as bad. If 'bads', bads channels
        are extracted from both info['bads'] and cov['bads'].
    proj : bool (default true)
        Apply or not projections to keep rank of data.
    seeg : float (default 0.1)
        Regularization factor for sEEG signals.
    ecog : float (default 0.1)
        Regularization factor for ECoG signals.
    hbo : float (default 0.1)
        Regularization factor for HBO signals.
    hbr : float (default 0.1)
        Regularization factor for HBR signals.
    verbose : bool | str | int | None (default None)
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    reg_cov : Covariance
        The regularized covariance matrix.

    See Also
    --------
    mne.compute_covariance
    """  # noqa: E501
    cov = cov.copy()
    info._check_consistency()

    if exclude is None:
        raise ValueError('exclude must be a list of strings or "bads"')

    if exclude == 'bads':
        exclude = info['bads'] + cov['bads']

    picks_dict = {ch_type: [] for ch_type in _DATA_CH_TYPES_SPLIT}
    picks_dict.update(dict(_picks_by_type(info, exclude=exclude)))
    info_ch_names = info['ch_names']
    ch_names_by_type = dict()
    for ch_type, picks_type in picks_dict.items():
        ch_names_by_type[ch_type] = [info_ch_names[i] for i in picks_type]

    # This actually removes bad channels from the cov, which is not backward
    # compatible, so let's leave all channels in
    cov_good = pick_channels_cov(cov, include=info_ch_names, exclude=exclude)
    ch_names = cov_good.ch_names

    # Now get the indices for each channel type in the cov
    idx_cov = {ch_type: [] for ch_type in ch_names_by_type}
    for i, ch in enumerate(ch_names):
        for ch_type in ch_names_by_type:
            if ch in ch_names_by_type[ch_type]:
                idx_cov[ch_type].append(i)
                break
        else:
            raise Exception('channel %s is unknown type' % ch)

    C = cov_good['data']

    assert len(C) == sum(map(len, idx_cov.values()))

    if proj:
        projs = info['projs'] + cov_good['projs']
        projs = activate_proj(projs)

    regs = dict(mag=mag, grad=grad, eeg=eeg,
                seeg=seeg, ecog=ecog, hbo=hbo, hbr=hbr)

    for ch_type in idx_cov:
        desc = ch_type.upper()
        idx = idx_cov[ch_type]
        reg = regs[ch_type]

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


def _regularized_covariance(data, reg=None, method_params=None, info=None):
    """Compute a regularized covariance from data using sklearn.

    This is a convenience wrapper for mne.decoding functions, which
    adopted a slightly different covariance API.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        The covariance matrix.
    """
    if reg is None:
        reg = 'empirical'
    try:
        reg = float(reg)
    except ValueError:
        pass
    if isinstance(reg, float):
        if method_params is not None:
            raise ValueError('If reg is a float, method_params must be None '
                             '(got %s)' % (type(method_params),))
        method_params = dict(shrinkage=dict(
            shrinkage=reg, assume_centered=True, store_precision=False))
        reg = 'shrinkage'
    elif not isinstance(reg, string_types):
        raise ValueError('reg must be a float, str, or None, got %s (%s)'
                         % (reg, type(reg)))
    method, method_params = _check_method_params(reg, method_params,
                                                 name='reg', allow_auto=False)
    info = create_info(data.shape[-2], 1000., 'eeg') if info is None else info
    picks_list = _picks_by_type(info)
    scalings = _handle_default('scalings_cov_rank', None)
    cov = _compute_covariance_auto(
        data.T, method=method, method_params=method_params,
        info=info, cv=None, n_jobs=1, stop_early=True,
        picks_list=picks_list, scalings=scalings)[reg]['data']
    return cov


@verbose
def compute_whitener(noise_cov, info, picks=None, rank=None,
                     scalings=None, return_rank=False,
                     verbose=None):
    """Compute whitening matrix.

    Parameters
    ----------
    noise_cov : Covariance
        The noise covariance.
    info : dict
        The measurement info.
    picks : array-like of int | None
        The channels indices to include. If None the MEG and EEG
        channels in info, except bad channels, are used.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    scalings : dict | None
        The rescaling method to be applied. See documentation of
        ``prepare_noise_cov`` for details.
    return_rank : bool
        If True, return the rank used to compute the whitener.

        .. versionadded:: 0.15
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    W : 2d array
        The whitening matrix.
    ch_names : list
        The channel names.
    rank : int
        Rank reduction of the whitener. Returned only if return_rank is True.
    """
    if picks is None:
        # If this changes, we will need to change _setup_plot_projector, too:
        picks = _pick_data_channels(info, with_ref_meg=False, exclude='bads')

    ch_names = [info['ch_names'][k] for k in picks]

    # XXX this relies on pick_channels, which does not respect order,
    # so this could create problems if users have reordered their data
    noise_cov = pick_channels_cov(noise_cov, include=ch_names, exclude=[])
    if len(noise_cov['data']) != len(ch_names):
        missing = list(set(ch_names) - set(noise_cov['names']))
        raise RuntimeError('Not all channels present in noise covariance:\n%s'
                           % missing)

    scalings = _handle_default('scalings_cov_rank', scalings)
    W, noise_cov, n_nzero = _get_whitener(noise_cov, info, ch_names, rank,
                                          pca=False, scalings=scalings)

    # Do the back projection
    W = np.dot(noise_cov['eigvec'].T, W)
    out = W, ch_names
    if return_rank:
        out += (n_nzero,)
    return out


@verbose
def whiten_evoked(evoked, noise_cov, picks=None, diag=None, rank=None,
                  scalings=None, verbose=None):
    """Whiten evoked data using given noise covariance.

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    noise_cov : instance of Covariance
        The noise covariance
    picks : array-like of int | None
        The channel indices to whiten. Can be None to whiten any data channel
        such as MEG and EEG data.
    diag : bool (default False)
        If True, whiten using only the diagonal of the covariance.
    rank : None | int | dict (default None)
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg', 'meg' or any other
        data channel type such as 'seeg' or 'ecog' can be used
        to specify the rank for each modality.
    scalings : dict | None (default None)
        To achieve reliable rank estimation on multiple sensors,
        sensors have to be rescaled. This parameter controls the
        rescaling. If dict, it will override the
        following default dict (default if None):

            dict(mag=1e12, grad=1e11, eeg=1e5)

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    evoked_white : instance of Evoked
        The whitened evoked data.
    """
    evoked = evoked.copy()
    if picks is None:
        picks = pick_types(evoked.info, meg=True, eeg=True, seeg=True,
                           ecog=True)

    if diag:
        noise_cov = noise_cov.as_diag()

    W, rank = compute_whitener(noise_cov, evoked.info, picks=picks,
                               rank=rank, scalings=scalings)

    evoked.data[picks] = np.sqrt(evoked.nave) * np.dot(W, evoked.data[picks])
    return evoked


@verbose
def _read_cov(fid, node, cov_kind, limited=False, verbose=None):
    """Read a noise covariance matrix."""
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
                    diag = True
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
                    diag = False
                    logger.info('    %d x %d full covariance (kind = %d) '
                                'found.' % (dim, dim, cov_kind))
                else:
                    diag = False
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
            assert dim == len(data)
            assert data.ndim == (1 if diag else 2)
            cov = dict(kind=cov_kind, diag=diag, dim=dim, names=names,
                       data=data, projs=projs, bads=bads, nfree=nfree, eig=eig,
                       eigvec=eigvec)
            if score is not None:
                cov['loglik'] = score
            if method is not None:
                cov['method'] = method
            if limited:
                del cov['kind'], cov['dim'], cov['diag']

            return cov

    logger.info('    Did not find the desired covariance matrix (kind = %d)'
                % cov_kind)

    return None


def _write_cov(fid, cov):
    """Write a noise covariance matrix."""
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
    """Scale data type-dependently for estimation."""
    scalings = _check_scaling_inputs(data, picks_list, scalings)
    if isinstance(scalings, dict):
        picks_dict = dict(picks_list)
        scalings = [(picks_dict[k], v) for k, v in scalings.items()
                    if k in picks_dict]
        for idx, scaling in scalings:
            data[idx, :] *= scaling  # F - order
    else:
        data *= scalings[:, np.newaxis]  # F - order


def _invert_scalings(scalings):
    if isinstance(scalings, dict):
        scalings = dict((k, 1. / v) for k, v in scalings.items())
    elif isinstance(scalings, np.ndarray):
        scalings = 1. / scalings
    return scalings


def _undo_scaling_array(data, picks_list, scalings):
    scalings = _invert_scalings(_check_scaling_inputs(data, picks_list,
                                                      scalings))
    return _apply_scaling_array(data, picks_list, scalings)


def _apply_scaling_cov(data, picks_list, scalings):
    """Scale resulting data after estimation."""
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
    scalings = _invert_scalings(_check_scaling_inputs(data, picks_list,
                                                      scalings))
    return _apply_scaling_cov(data, picks_list, scalings)


def _check_scaling_inputs(data, picks_list, scalings):
    """Aux function."""
    rescale_dict_ = dict(mag=1e15, grad=1e13, eeg=1e6)

    scalings_ = None
    if isinstance(scalings, string_types) and scalings == 'norm':
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


def _estimate_rank_meeg_signals(data, info, scalings, tol='auto',
                                return_singular=False):
    """Estimate rank for M/EEG data.

    Parameters
    ----------
    data : np.ndarray of float, shape(n_channels, n_samples)
        The M/EEG signals.
    info : Info
        The measurment info.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e15, grad=1e13, eeg=1e6)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    tol : float | str
        Tolerance. See ``estimate_rank``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.

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
                        return_singular=return_singular)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_array(data, picks_list, scalings)
    return out


def _estimate_rank_meeg_cov(data, info, scalings, tol='auto',
                            return_singular=False):
    """Estimate rank of M/EEG covariance data, given the covariance.

    Parameters
    ----------
    data : np.ndarray of float, shape (n_channels, n_channels)
        The M/EEG covariance.
    info : Info
        The measurment info.
    scalings : dict | 'norm' | np.ndarray | None
        The rescaling method to be applied. If dict, it will override the
        following default dict:

            dict(mag=1e12, grad=1e11, eeg=1e5)

        If 'norm' data will be scaled by channel-wise norms. If array,
        pre-specified norms will be used. If None, no scaling will be applied.
    tol : float | str
        Tolerance. See ``estimate_rank``.
    return_singular : bool
        If True, also return the singular values that were used
        to determine the rank.

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
                        return_singular=return_singular)
    rank = out[0] if isinstance(out, tuple) else out
    ch_type = ' + '.join(list(zip(*picks_list))[0])
    logger.info('estimated rank (%s): %d' % (ch_type, rank))
    _undo_scaling_cov(data, picks_list, scalings)
    return out
