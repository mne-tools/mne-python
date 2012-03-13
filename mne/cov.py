# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy
import os
from math import floor, ceil
import warnings

import numpy as np
from scipy import linalg

from . import fiff
from .fiff.write import start_file, end_file
from .fiff.proj import make_projector, proj_equal
from .fiff import fiff_open
from .fiff.pick import pick_types, channel_indices_by_type
from .fiff.constants import FIFF
from .epochs import _is_good


def _check_covs_algebra(cov1, cov2):
    if cov1.ch_names != cov2.ch_names:
        raise ValueError('Both Covariance do not have the same list of '
                         'channels.')
    if map(str, cov1._cov['projs']) != map(str, cov2._cov['projs']):
        raise ValueError('Both Covariance do not have the same list of '
                         'SSP projections.')
    if cov1._cov['bads'] != cov2._cov['bads']:
        raise ValueError('Both Covariance do not have the same list of '
                         'bad channels.')


class Covariance(object):
    """Noise covariance matrix

    Parameters
    ----------
    fname: string
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
        cov = fiff.read_cov(fid, tree, FIFF.FIFFV_MNE_NOISE_COV)
        fid.close()

        self._cov = cov
        self.data = cov['data']
        self.ch_names = cov['names']
        self.nfree = cov['nfree']

    def save(self, fname):
        """save covariance matrix in a FIF file"""
        fid = start_file(fname)

        try:
            fiff.write_cov(fid, self._cov)
        except Exception as inst:
            os.remove(fname)
            raise '%s', inst

        end_file(fid)

    def __repr__(self):
        s = "size : %s x %s" % self.data.shape
        s += ", data : %s" % self.data
        return "Covariance (%s)" % s

    def __add__(self, cov):
        """Add Covariance taking into account number of degrees of freedom"""
        _check_covs_algebra(self, cov)
        this_cov = copy.deepcopy(cov)
        this_cov.data[:] = ((this_cov.data * this_cov.nfree) + \
                            (self.data * self.nfree)) / \
                                (self.nfree + this_cov.nfree)
        this_cov._cov['nfree'] += self._cov['nfree']
        this_cov.nfree = this_cov._cov['nfree']
        return this_cov

    def __iadd__(self, cov):
        """Add Covariance taking into account number of degrees of freedom"""
        _check_covs_algebra(self, cov)
        self.data[:] = ((self.data * self.nfree) + \
                            (cov.data * cov.nfree)) / \
                                (self.nfree + cov.nfree)
        self._cov['nfree'] += cov._cov['nfree']
        self.nfree = cov._cov['nfree']
        return self


###############################################################################
# IO

def read_cov(fname):
    """Read a noise covariance from a FIF file.

    Parameters
    ----------
    fname: string
        The name of file containing the covariance matrix.

    Returns
    -------
    cov: Covariance
        The noise covariance matrix.
    """
    return Covariance(fname)


###############################################################################
# Estimate from data

def compute_raw_data_covariance(raw, tmin=None, tmax=None, tstep=0.2,
                                reject=None, flat=None):
    """Estimate noise covariance matrix from a continuous segment of raw data

    It is typically useful to estimate a noise covariance
    from empty room data or time intervals before starting
    the stimulation.

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

    picks = pick_types(raw.info, meg=True, eeg=True, eog=True)
    picks_data = pick_types(raw.info, meg=True, eeg=True, eog=False)
    idx = [list(picks).index(k) for k in picks_data]

    data = 0
    n_samples = 0
    mu = 0

    info = copy.copy(raw.info)
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
            print "Artefact detected in [%d, %d]" % (first, last)

    mu /= n_samples
    data -= n_samples * mu[:, None] * mu[None, :]
    data /= (n_samples - 1.0)
    print "Number of samples used : %d" % n_samples
    print '[done]'

    cov = Covariance(None)
    cov.data = data
    cov.ch_names = [raw.info['ch_names'][k] for k in picks_data]
    cov.nfree = n_samples

    # XXX : do not compute eig and eigvec now (think it's better...)
    eig = None
    eigvec = None

    #   Store structure for fif
    cov._cov = dict(kind=FIFF.FIFFV_MNE_NOISE_COV, diag=False, dim=len(data),
                    names=cov.ch_names, data=data,
                    projs=copy.deepcopy(raw.info['projs']),
                    bads=raw.info['bads'], nfree=n_samples, eig=eig,
                    eigvec=eigvec)

    return cov


def compute_covariance(epochs, keep_sample_mean=True):
    """Estimate noise covariance matrix from epochs

    The noise covariance is typically estimated on pre-stim periods
    when the stim onset if defined from events.

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
    projs = epochs[0].info['projs']
    ch_names = epochs[0].ch_names

    # make sure Epochs are compatible
    for epochs_t in epochs[1:]:
        if epochs_t.info['bads'] != bads:
            raise ValueError('Epochs must have same bad channels')
        if epochs_t.ch_names != ch_names:
            raise ValueError('Epochs must have same channel names')
        for proj_a, proj_b in zip(epochs_t.info['projs'], projs):
            if not proj_equal(proj_a, proj_b):
                raise ValueError('Epochs must have same projectors')

    n_epoch_types = len(epochs)
    data = 0.0
    data_mean = list(np.zeros(n_epoch_types))
    n_samples = np.zeros(n_epoch_types, dtype=np.int)
    n_epochs = np.zeros(n_epoch_types, dtype=np.int)

    picks_meeg = pick_types(epochs[0].info, meg=True, eeg=True, eog=False)
    ch_names = [epochs[0].ch_names[k] for k in picks_meeg]
    for i, epochs_t in enumerate(epochs):
        for e in epochs_t:
            e = e[picks_meeg]
            if not keep_sample_mean:
                data_mean[i] += e
            data += np.dot(e, e.T)
            n_samples[i] += e.shape[1]
            n_epochs[i] += 1

    n_samples_tot = int(np.sum(n_samples))

    if n_samples_tot == 0:
        raise ValueError('Not enough samples to compute the noise covariance'
                         ' matrix : %d samples' % n_samples_tot)

    if keep_sample_mean:
        data /= n_samples_tot
    else:
        n_samples_epoch = n_samples / n_epochs
        norm_const = np.sum(n_samples_epoch * (n_epochs - 1))
        for i, mean in enumerate(data_mean):
            data -= 1.0 / n_epochs[i] * np.dot(mean, mean.T)
        data /= norm_const

    cov = Covariance(None)
    cov.data = data
    cov.ch_names = ch_names
    cov.nfree = n_samples_tot

    # XXX : do not compute eig and eigvec now (think it's better...)
    eig = None
    eigvec = None

    #   Store structure for fif
    cov._cov = dict(kind=1, diag=False, dim=len(data), names=ch_names,
                    data=data, projs=copy.deepcopy(epochs[0].info['projs']),
                    bads=epochs[0].info['bads'], nfree=n_samples_tot, eig=eig,
                    eigvec=eigvec)

    print "Number of samples used : %d" % n_samples_tot
    print '[done]'

    return cov


###############################################################################
# Writing

def write_cov(fname, cov):
    """Write a noise covariance matrix

    Parameters
    ----------
    fname: string
        The name of the file

    cov: Covariance
        The noise covariance matrix
    """
    cov.save(fname)


###############################################################################
# Prepare for inverse modeling

def rank(A, tol=1e-8):
    s = linalg.svd(A, compute_uv=0)
    return np.sum(np.where(s > s[0] * tol, 1, 0))


def _get_whitener(A, pca, ch_type):
    # whitening operator
    rnk = rank(A)
    eig, eigvec = linalg.eigh(A, overwrite_a=True)
    eigvec = eigvec.T
    eig[:-rnk] = 0.0
    print 'Setting small %s eigenvalues to zero.' % ch_type
    if not pca:  # No PCA case.
        print 'Not doing PCA for %s.' % ch_type
    else:
        print 'Doing PCA for %s.' % ch_type
        # This line will reduce the actual number of variables in data
        # and leadfield to the true rank.
        eigvec = eigvec[:-rnk].copy()
    return eig, eigvec


def prepare_noise_cov(noise_cov, info, ch_names):
    C_ch_idx = [noise_cov.ch_names.index(c) for c in ch_names]
    C = noise_cov.data[C_ch_idx][:, C_ch_idx]

    # Create the projection operator
    proj, ncomp, _ = make_projector(info['projs'], ch_names)
    if ncomp > 0:
        print '    Created an SSP operator (subspace dimension = %d)' % ncomp
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

    noise_cov = dict(data=C, eig=eig, eigvec=eigvec, dim=len(ch_names),
                     diag=False, names=ch_names)

    return noise_cov
