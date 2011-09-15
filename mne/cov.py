# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy
import os
from math import floor, ceil
import numpy as np
from scipy import linalg

from .fiff.constants import FIFF
from .fiff.tag import find_tag
from .fiff.tree import dir_tree_find
from .fiff.proj import read_proj
from .fiff.channels import _read_bad_channels

from .fiff.write import start_block, end_block, write_int, write_name_list, \
                       write_double, write_float_matrix, start_file, end_file
from .fiff.proj import write_proj, make_projector
from .fiff import fiff_open
from .fiff.pick import pick_types, channel_indices_by_type
from .epochs import _is_good


class Covariance(object):
    """Noise covariance matrix"""

    _kind_to_id = dict(full=1, sparse=2, diagonal=3)  # XXX : check
    _id_to_kind = {1: 'full', 2: 'sparse', 3: 'diagonal'}  # XXX : check

    def __init__(self, fname, kind='full'):
        self.kind = kind

        if fname is None:
            return

        if self.kind in Covariance._kind_to_id:
            cov_kind = Covariance._kind_to_id[self.kind]
        else:
            raise ValueError('Unknown type of covariance. '
                             'Choose between full, sparse or diagonal.')

        # Reading
        fid, tree, _ = fiff_open(fname)
        cov = read_cov(fid, tree, cov_kind)
        fid.close()

        self._cov = cov
        self.data = cov['data']
        self.ch_names = cov['names']

    def save(self, fname):
        """save covariance matrix in a FIF file"""
        write_cov_file(fname, self._cov)

    def __repr__(self):
        s = "kind : %s" % self.kind
        s += ", size : %s x %s" % self.data.shape
        s += ", data : %s" % self.data
        return "Covariance (%s)" % s


###############################################################################
# IO

def read_cov(fid, node, cov_kind):
    """Read a noise covariance matrix

    Parameters
    ----------
    fid: file
        The file descriptor

    node: dict
        The node in the FIF tree

    cov_kind: int
        The type of covariance. XXX : clarify

    Returns
    -------
    data: dict
        The noise covariance
    """
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

            dim = tag.data
            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_NFREE)
            if tag is None:
                nfree = -1
            else:
                nfree = tag.data

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
                    print '\t%d x %d diagonal covariance (kind = %d) found.' \
                                                        % (dim, dim, cov_kind)

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
                    print '\t%d x %d full covariance (kind = %d) found.' \
                                                        % (dim, dim, cov_kind)
                else:
                    diagmat = False
                    data = tag.data
                    print '\t%d x %d sparse covariance (kind = %d) found.' \
                                                        % (dim, dim, cov_kind)

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
            projs = read_proj(fid, this)

            #   Read the bad channel list
            bads = _read_bad_channels(fid, this)

            #   Put it together
            cov = dict(kind=cov_kind, diag=diagmat, dim=dim, names=names,
                       data=data, projs=projs, bads=bads, nfree=nfree, eig=eig,
                       eigvec=eigvec)
            return cov

    print 'Did not find the desired covariance matrix'

    return None


###############################################################################
# Estimate from data

def compute_raw_data_covariance(raw, tmin=None, tmax=None, tstep=0.2,
                             reject=None, flat=None, keep_sample_mean=True):
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
    keep_sample_mean : bool
        If False data are centered at each instant before computing
        the covariance.

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
    return cov


def compute_covariance(epochs, keep_sample_mean=True):
    """Estimate noise covariance matrix from epochs

    The noise covariance is typically estimated on pre-stim periods
    when the stim onset if defined from events.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    keep_sample_mean : bool
        If False data are centered at each instant before computing
        the covariance.
    Returns
    -------
    cov : instance of Covariance
        The computed covariance.
    """
    data = 0.0
    data_mean = 0.0
    n_samples = 0
    n_epochs = 0
    picks_meeg = pick_types(epochs.info, meg=True, eeg=True, eog=False)
    ch_names = [epochs.ch_names[k] for k in picks_meeg]
    for e in epochs:
        e = e[picks_meeg]
        if not keep_sample_mean:
            data_mean += np.sum(e, axis=0)
        data += np.dot(e, e.T)
        n_samples += e.shape[1]
        n_epochs += 1

    if n_samples == 0:
        raise ValueError('Not enough samples to compute the noise covariance'
                         ' matrix : %d samples' % n_samples)

    if keep_sample_mean:
        data /= n_samples
    else:
        data /= n_samples - 1
        data -= n_samples / (1.0 - n_samples) * np.dot(data_mean, data_mean.T)
    cov = Covariance(None)
    cov.data = data
    cov.ch_names = ch_names

    print "Number of samples used : %d" % n_samples
    print '[done]'

    return cov


###############################################################################
# Writing

def write_cov(fid, cov):
    """Write a noise covariance matrix

    Parameters
    ----------
    fid: file
        The file descriptor

    cov: dict
        The noise covariance matrix to write
    """
    start_block(fid, FIFF.FIFFB_MNE_COV)

    #   Dimensions etc.
    write_int(fid, FIFF.FIFF_MNE_COV_KIND, cov['kind'])
    write_int(fid, FIFF.FIFF_MNE_COV_DIM, cov['dim'])
    if cov['nfree'] > 0:
        write_int(fid, FIFF.FIFF_MNE_COV_NFREE, cov['nfree'])

    #   Channel names
    if cov['names'] is not None:
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
    write_proj(fid, cov['projs'])

    #   Bad channels
    if cov['bads'] is not None:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, cov['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    #   Done!
    end_block(fid, FIFF.FIFFB_MNE_COV)


def write_cov_file(fname, cov):
    """Write a noise covariance matrix

    Parameters
    ----------
    fname: string
        The name of the file

    cov: dict
        The noise covariance
    """
    fid = start_file(fname)

    try:
        write_cov(fid, cov)
    except Exception as inst:
        os.remove(fname)
        raise '%s', inst

    end_file(fid)


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
        print '\tCreated an SSP operator (subspace dimension = %d)' % ncomp
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
