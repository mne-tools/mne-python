# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy as cp
import os
from math import floor, ceil
import warnings

import numpy as np
from scipy import linalg

from .io.write import start_file, end_file
from .io.proj import (make_projector, proj_equal, activate_proj,
                      _has_eeg_average_ref_proj)
from .io import fiff_open
from .io.pick import (pick_types, channel_indices_by_type, pick_channels_cov,
                      pick_channels)
from .io.constants import FIFF
from .io.meas_info import read_bad_channels
from .io.proj import _read_proj, _write_proj
from .io.tag import find_tag
from .io.tree import dir_tree_find
from .io.write import (start_block, end_block, write_int, write_name_list,
                       write_double, write_float_matrix)
from .epochs import _is_good
from .utils import check_fname, logger, verbose
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
        return self['data']

    @property
    def ch_names(self):
        return self['names']

    @property
    def nfree(self):
        return self['nfree']

    def save(self, fname):
        """save covariance matrix in a FIF file"""
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


###############################################################################
# IO

def read_cov(fname):
    """Read a noise covariance from a FIF file.

    Parameters
    ----------
    fname : string
        The name of file containing the covariance matrix. It should end with
        -cov.fif or -cov.fif.gz.

    Returns
    -------
    cov : Covariance
        The noise covariance matrix.
    """
    check_fname(fname, 'covariance', ('-cov.fif', '-cov.fif.gz'))

    return Covariance(fname)


###############################################################################
# Estimate from data

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
    reject : dict
        Rejection parameters based on peak to peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )

    flat : dict
        Rejection parameters based on flatness of signal
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'
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
                       projs=None, verbose=None):
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

    Parameters
    ----------
    epochs : instance of Epochs, or a list of Epochs objects
        The epochs
    keep_sample_mean : bool
        If False, the average response over epochs is computed for
        each event type and subtracted during the covariance
        computation. This is useful if the evoked response from a
        previous stimulus extends into the baseline period of the next.
    tmin : float | None
        Start time for baseline. If None start at first sample.
    tmax : float | None
        End time for baseline. If None end at last sample.
    projs : list of Projection | None
        List of projectors to use in covariance calculation, or None
        to indicate that the projectors from the epochs should be
        inherited. If None, then projectors from all epochs must match.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    cov : instance of Covariance
        The computed covariance.
    """

    if not isinstance(epochs, list):
        epochs = _unpack_epochs(epochs)
    else:
        epochs = [ep for li in [_unpack_epochs(epoch) for epoch in epochs]
                  for ep in li]

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
                if not proj_equal(proj_a, proj_b):
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

    n_epoch_types = len(epochs)
    data = 0.0
    data_mean = list(np.zeros(n_epoch_types))
    n_samples = np.zeros(n_epoch_types, dtype=np.int)
    n_epochs = np.zeros(n_epoch_types, dtype=np.int)

    picks_meeg = pick_types(epochs[0].info, meg=True, eeg=True, eog=False,
                            ref_meg=False, exclude=[])
    ch_names = [epochs[0].ch_names[k] for k in picks_meeg]

    for i, epochs_t in enumerate(epochs):

        tstart, tend = None, None
        if tmin is not None:
            tstart = np.where(epochs_t.times >= tmin)[0][0]
        if tmax is not None:
            tend = np.where(epochs_t.times <= tmax)[0][-1] + 1
        tslice = slice(tstart, tend, None)

        for e in epochs_t:
            e = e[picks_meeg][:, tslice]
            if not keep_sample_mean:
                data_mean[i] += e
            data += np.dot(e, e.T)
            n_samples[i] += e.shape[1]
            n_epochs[i] += 1

    n_samples_tot = int(np.sum(n_samples))

    _check_n_samples(n_samples_tot, len(picks_meeg))

    if keep_sample_mean:
        data /= n_samples_tot
    else:
        n_samples_epoch = n_samples // n_epochs
        norm_const = np.sum(n_samples_epoch * (n_epochs - 1))
        for i, mean in enumerate(data_mean):
            data -= 1.0 / n_epochs[i] * np.dot(mean, mean.T)
        data /= norm_const

    cov = Covariance(None)

    # XXX : do not compute eig and eigvec now (think it's better...)
    eig = None
    eigvec = None

    cov.update(kind=1, diag=False, dim=len(data), names=ch_names,
               data=data, projs=projs, bads=epochs[0].info['bads'],
               nfree=n_samples_tot, eig=eig, eigvec=eigvec)

    logger.info("Number of samples used : %d" % n_samples_tot)
    logger.info('[done]')

    return cov


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

def rank(A, tol=1e-8):
    s = linalg.svd(A, compute_uv=0)
    return np.sum(np.where(s > s[0] * tol, 1, 0))


def _unpack_epochs(epochs):
    """ Aux Function """
    if len(epochs.event_id) > 1:
        epochs = [epochs[k] for k in epochs.event_id]
    else:
        epochs = [epochs]

    return epochs


@verbose
def _get_whitener(A, pca, ch_type, verbose=None):
    # whitening operator
    rnk = rank(A)
    eig, eigvec = linalg.eigh(A, overwrite_a=True)
    eigvec = eigvec.T
    eig[:-rnk] = 0.0
    logger.info('Setting small %s eigenvalues to zero.' % ch_type)
    if not pca:  # No PCA case.
        logger.info('Not doing PCA for %s.' % ch_type)
    else:
        logger.info('Doing PCA for %s.' % ch_type)
        # This line will reduce the actual number of variables in data
        # and leadfield to the true rank.
        eigvec = eigvec[:-rnk].copy()
    return eig, eigvec


@verbose
def prepare_noise_cov(noise_cov, info, ch_names, verbose=None):
    """Prepare noise covariance matrix

    Parameters
    ----------
    noise_cov : Covariance
        The noise covariance to process.
    info : dict
        The measurement info (used to get channel types and bad channels).
    ch_names : list
        The channel names to be considered.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    C_ch_idx = [noise_cov.ch_names.index(c) for c in ch_names]
    if noise_cov['diag'] is False:
        C = noise_cov.data[C_ch_idx][:, C_ch_idx]
    else:
        C = np.diag(noise_cov.data[C_ch_idx])

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

    if has_meg:
        C_meg = C[C_meg_idx][:, C_meg_idx]
        C_meg_eig, C_meg_eigvec = _get_whitener(C_meg, False, 'MEG')

    if has_eeg:
        C_eeg = C[C_eeg_idx][:, C_eeg_idx]
        C_eeg_eig, C_eeg_eigvec = _get_whitener(C_eeg, False, 'EEG')
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
    """
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

        this_C = C[idx][:, idx]
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


def compute_whitener(noise_cov, info, picks=None, verbose=None):
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
    noise_cov = prepare_noise_cov(noise_cov, info, ch_names)
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


def whiten_evoked(evoked, noise_cov, picks, diag=False):
    """Whiten evoked data using given noise covariance

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data
    noise_cov : instance of Covariance
        The noise covariance
    picks : array-like of int
        The channel indices to whiten
    diag : bool
        If True, whiten using only the diagonal of the covariance

    Returns
    -------
    evoked_white : instance of Evoked
        The whitened evoked data.
    """
    ch_names = [evoked.ch_names[k] for k in picks]
    n_chan = len(ch_names)
    evoked = cp.deepcopy(evoked)

    if diag:
        noise_cov = cp.deepcopy(noise_cov)
        noise_cov['data'] = np.diag(np.diag(noise_cov['data']))

    noise_cov = prepare_noise_cov(noise_cov, evoked.info, ch_names)

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
    evoked.data[picks] = np.sqrt(evoked.nave) * np.dot(W, evoked.data[picks])
    return evoked


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

    #   Done!
    end_block(fid, FIFF.FIFFB_MNE_COV)
