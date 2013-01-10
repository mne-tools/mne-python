# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy as cp
import os
from math import floor, ceil
import warnings

import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from . import fiff, verbose
from .fiff.write import start_file, end_file
from .fiff.proj import make_projector, proj_equal, activate_proj
from .fiff import fiff_open
from .fiff.pick import pick_types, channel_indices_by_type, pick_channels_cov
from .fiff.constants import FIFF
from .epochs import _is_good


def _check_covs_algebra(cov1, cov2):
    if cov1.ch_names != cov2.ch_names:
        raise ValueError('Both Covariance do not have the same list of '
                         'channels.')
    if map(str, cov1['projs']) != map(str, cov2['projs']):
        raise ValueError('Both Covariance do not have the same list of '
                         'SSP projections.')
    if cov1['bads'] != cov2['bads']:
        raise ValueError('Both Covariance do not have the same list of '
                         'bad channels.')


class Covariance(dict):
    """Noise covariance matrix

    Parameters
    ----------
    fname : string
        The name of the raw file

    Attributes
    ----------
    data : 2D array of shape [n_channels x n_channels]
        The covariance

    ch_names: list of string
        List of channels' names

    nfree : int
        Number of degrees of freedom i.e. number of time points used
    """

    def __init__(self, fname):
        if fname is None:
            return

        # Reading
        fid, tree, _ = fiff_open(fname)
        self.update(fiff.read_cov(fid, tree, FIFF.FIFFV_MNE_NOISE_COV))
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
        fid = start_file(fname)

        try:
            fiff.write_cov(fid, self)
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
            raise ValueError('Covariance is already diagonal.')
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
        this_cov['data'] = ((this_cov['data'] * this_cov['nfree']) +
                            (self['data'] * self['nfree'])) / \
                                (self['nfree'] + this_cov['nfree'])
        this_cov['nfree'] += self['nfree']
        return this_cov

    def __iadd__(self, cov):
        """Add Covariance taking into account number of degrees of freedom"""
        _check_covs_algebra(self, cov)
        self['data'][:] = ((self['data'] * self['nfree']) + \
                            (cov['data'] * cov['nfree'])) / \
                                (self['nfree'] + cov['nfree'])
        self['nfree'] += cov['nfree']
        return self


###############################################################################
# IO

def read_cov(fname):
    """Read a noise covariance from a FIF file.

    Parameters
    ----------
    fname : string
        The name of file containing the covariance matrix.

    Returns
    -------
    cov : Covariance
        The noise covariance matrix.
    """
    return Covariance(fname)


###############################################################################
# Estimate from data

def _check_n_samples(n_samples, n_chan):
    """Check to see if there are enough samples for reliable cov calc"""
    n_samples_min = 10 * (n_chan + 1) / 2
    if n_samples <= 0:
        raise ValueError('No samples found to compute the covariance matrix')
    if n_samples < n_samples_min:
        text = ('Too few samples (required : %d got : %d), covariance '
                'estimate may be unreliable' % (n_samples_min, n_samples))
        warnings.warn(text)
        logger.warn(text)


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
        Size of data chunks for artefact rejection.
    reject : dict
        Rejection parameters based on peak to peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done.
        Values are float. Example:
        reject = dict(grad=4000e-13, # T / m (gradiometers)
                      mag=4e-12, # T (magnetometers)
                      eeg=40e-6, # uV (EEG channels)
                      eog=250e-6 # uV (EOG channels)
                      )
    flat : dict
        Rejection parameters based on flatness of signal
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'
        If flat is None then no rejection is done.
    picks : array of int
        Indices of channels to include (if None, all channels
        are used).
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
        stop = raw.last_samp - raw.first_samp
    else:
        stop = int(ceil(tmax * sfreq))
    step = int(ceil(tstep * raw.info['sfreq']))

    if picks is None:
        picks_data = pick_types(raw.info, meg=True, eeg=True, eog=False)
    else:
        picks_data = picks

    picks = pick_types(raw.info, meg=True, eeg=True, eog=True)
    idx = [list(picks).index(k) for k in picks_data]

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
        if _is_good(raw_segment, info['ch_names'], idx_by_type, reject, flat):
            mu += raw_segment[idx].sum(axis=1)
            data += np.dot(raw_segment[idx], raw_segment[idx].T)
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

    ch_names = [raw.info['ch_names'][k] for k in picks_data]
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
    with different IDs), an Epochs object for each event type has to
    be created and a list of Epochs has to be passed to this function.

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
        epochs = [epochs]

    # check for baseline correction
    for epochs_t in epochs:
        if epochs_t.baseline is None:
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

    picks_meeg = pick_types(epochs[0].info, meg=True, eeg=True, eog=False)
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
        n_samples_epoch = n_samples / n_epochs
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
        The name of the file

    cov : Covariance
        The noise covariance matrix
    """
    cov.save(fname)


###############################################################################
# Prepare for inverse modeling

def rank(A, tol=1e-8):
    s = linalg.svd(A, compute_uv=0)
    return np.sum(np.where(s > s[0] * tol, 1, 0))


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

    pick_meg = pick_types(info, meg=True, eeg=False, exclude=info['bads'])
    pick_eeg = pick_types(info, meg=False, eeg=True, exclude=info['bads'])
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


def regularize(cov, info, mag=0.1, grad=0.1, eeg=0.1, exclude=None,
               proj=True, verbose=None):
    """Regularize noise covariance matrix

    This method works by adding a constant to the diagonal for each
    channel type separatly. Special care is taken to keep the
    rank of the data constant.

    Parameters
    ----------
    cov : Covariance
        The noise covariance matrix.
    info : dict
        The measurement info (used to get channel types and bad channels)
    mag : float
        Regularization factor for MEG magnetometers
    grad : float
        Regularization factor for MEG gradiometers
    eeg : float
        Regularization factor for EEG
    exclude : list
        List of channels to mark as bad. If None, bads channels
        are extracted from info and cov['bads'].
    proj : bool
        Apply or not projections to keep rank of data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    reg_cov : Covariance
        The regularized covariance matrix.
    """
    if exclude is None:
        exclude = info['bads'] + cov['bads']

    sel_eeg = pick_types(info, meg=False, eeg=True, exclude=exclude)
    sel_mag = pick_types(info, meg='mag', eeg=False, exclude=exclude)
    sel_grad = pick_types(info, meg='grad', eeg=False, exclude=exclude)

    info_ch_names = info['ch_names']
    ch_names_eeg = [info_ch_names[i] for i in sel_eeg]
    ch_names_mag = [info_ch_names[i] for i in sel_mag]
    ch_names_grad = [info_ch_names[i] for i in sel_grad]

    cov = pick_channels_cov(cov, include=info_ch_names, exclude=exclude)
    ch_names = cov.ch_names

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

    C = cov['data']

    assert len(C) == (len(idx_eeg) + len(idx_mag) + len(idx_grad))

    if proj:
        projs = info['projs'] + cov['projs']
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

    cov['data'] = C

    return cov


def compute_whitener(noise_cov, info, picks=None, verbose=None):
    """Compute whitening matrix

    Parameters
    ----------
    noise_cov : Covariance
        The noise covariance
    info : dict
        The measurement info
    picks : array of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    W : 2d array
        The whitening matrix
    ch_names : list
        The channel names
    """
    if picks is None:
        picks = pick_types(info, meg=True, eeg=True, exclude=info['bads'])

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
    picks : array of ints
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
