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
from .epochs import Epochs, _is_good


def rank(A, tol=1e-8):
    s = linalg.svd(A, compute_uv=0)
    return np.sum(np.where(s > s[0] * tol, 1, 0))


def _get_whitener(A, rnk, pca, ch_type):
    # whitening operator
    D, V = linalg.eigh(A, overwrite_a=True)
    I = np.argsort(D)[::-1]
    D = D[I]
    V = V[:, I]
    D = 1.0 / D
    if not pca:  # No PCA case.
        print 'Not doing PCA for %s.' % ch_type
        W = np.sqrt(D)[:, None] * V.T
    else:  # Rey's approach. MNE has been changed to implement this.
        print 'Setting small %s eigenvalues to zero.' % ch_type
        D[rnk:] = 0.0
        W = np.sqrt(D)[:, None] * V.T
        # This line will reduce the actual number of variables in data
        # and leadfield to the true rank.
        W = W[:rnk]
    return W


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

    def get_whitener(self, info, mag_reg=0.1, grad_reg=0.1, eeg_reg=0.1,
                     pca=True):
        """Compute whitener based on a list of channels

        Parameters
        ----------
        info : dict
            Measurement info of data to apply the whitener.
            Defines data channels and which are the bad channels
            to be ignored.
        mag_reg : float
            Regularization of the magnetometers.
            Recommended between 0.05 and 0.2
        grad_reg : float
            Regularization of the gradiometers.
            Recommended between 0.05 and 0.2
        eeg_reg : float
            Regularization of the EGG channels.
            Recommended between 0.05 and 0.2
        pca : bool
            If True, whitening is restricted to the space of
            the data. It makes sense when data have a low rank
            due to SSP or maxfilter.

        Returns
        -------
        W : instance of Whitener
        """

        if not 0 <= grad_reg <= 1:
            raise ValueError('grad_reg should be a scalar between 0 and 1')
        if not 0 <= mag_reg <= 1:
            raise ValueError('mag_reg should be a scalar between 0 and 1')
        if not 0 <= eeg_reg <= 1:
            raise ValueError('eeg_reg should be a scalar between 0 and 1')

        if pca and self.kind == 'diagonal':
            print "Setting pca to False with a diagonal covariance matrix."
            pca = False

        bads = info['bads']
        C_idx = [k for k, name in enumerate(self.ch_names)
                 if name in info['ch_names'] and name not in bads]
        ch_names = [self.ch_names[k] for k in C_idx]
        C_noise = self.data[np.ix_(C_idx, C_idx)]  # take covariance submatrix

        # Create the projection operator
        proj, ncomp, _ = make_projector(info['projs'], ch_names)
        if ncomp > 0:
            print '\tCreated an SSP operator (subspace dimension = %d)' % ncomp
            C_noise = np.dot(proj, np.dot(C_noise, proj.T))

        # Regularize Noise Covariance Matrix.
        variances = np.diag(C_noise)
        ind_meg = pick_types(info, meg=True, eeg=False, exclude=bads)
        names_meg = [info['ch_names'][k] for k in ind_meg]
        C_ind_meg = [ch_names.index(name) for name in names_meg]

        ind_grad = pick_types(info, meg='grad', eeg=False, exclude=bads)
        names_grad = [info['ch_names'][k] for k in ind_grad]
        C_ind_grad = [ch_names.index(name) for name in names_grad]

        ind_mag = pick_types(info, meg='mag', eeg=False, exclude=bads)
        names_mag = [info['ch_names'][k] for k in ind_mag]
        C_ind_mag = [ch_names.index(name) for name in names_mag]

        ind_eeg = pick_types(info, meg=False, eeg=True, exclude=bads)
        names_eeg = [info['ch_names'][k] for k in ind_eeg]
        C_ind_eeg = [ch_names.index(name) for name in names_eeg]

        has_meg = len(ind_meg) > 0
        has_eeg = len(ind_eeg) > 0

        if self.kind == 'diagonal':
            C_noise = np.diag(variances)
            rnkC_noise = len(variances)
            print 'Rank of noise covariance is %d' % rnkC_noise
        else:
            # estimate noise covariance matrix rank
            # Loop on all the required data types (MEG MAG, MEG GRAD, EEG)

            if has_meg:  # Separate rank of MEG
                rank_meg = rank(C_noise[C_ind_meg][:, C_ind_meg])
                print 'Rank of MEG part of noise covariance is %d' % rank_meg
            if has_eeg:  # Separate rank of EEG
                rank_eeg = rank(C_noise[C_ind_eeg][:, C_ind_eeg])
                print 'Rank of EEG part of noise covariance is %d' % rank_eeg

            for ind, reg in zip([C_ind_grad, C_ind_mag, C_ind_eeg],
                                [grad_reg, mag_reg, eeg_reg]):
                if len(ind) > 0:
                    # add constant on diagonal
                    C_noise[ind, ind] += reg * np.mean(variances[ind])

            if has_meg and has_eeg:  # Sets cross terms to zero
                C_noise[np.ix_(C_ind_meg, C_ind_eeg)] = 0.0
                C_noise[np.ix_(C_ind_eeg, C_ind_meg)] = 0.0

        # whitening operator
        if has_meg:
            W_meg = _get_whitener(C_noise[C_ind_meg][:, C_ind_meg], rank_meg,
                                  pca, 'MEG')

        if has_eeg:
            W_eeg = _get_whitener(C_noise[C_ind_eeg][:, C_ind_eeg], rank_eeg,
                                  pca, 'EEG')

        if has_meg and not has_eeg:  # Only MEG case.
            W = W_meg
        elif has_eeg and not has_meg:  # Only EEG case.
            W = W_eeg
        elif has_eeg and has_meg:  # Bimodal MEG and EEG case.
            # Whitening of MEG and EEG separately, which assumes zero
            # covariance between MEG and EEG (i.e., a block diagonal noise
            # covariance). This was recommended by Matti as EEG does not
            # measure all the signals from the same environmental noise sources
            # as MEG.
            W = np.r_[np.c_[W_meg, np.zeros((W_meg.shape[0], W_eeg.shape[1]))],
                      np.c_[np.zeros((W_eeg.shape[0], W_meg.shape[1])), W_eeg]]

        whitener = Whitener(W, names_meg + names_eeg)
        return whitener

    def __repr__(self):
        s = "kind : %s" % self.kind
        s += ", size : %s x %s" % self.data.shape
        s += ", data : %s" % self.data
        return "Covariance (%s)" % s


class Whitener(object):
    """Whitener

    Attributes
    ----------
    W : array
        Whiten matrix
    ch_names : list of strings
        Channel names (columns of W)
    """

    def __init__(self, W, ch_names):
        self.W = W
        self.ch_names = ch_names

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

    raise ValueError('Did not find the desired covariance matrix')

    return None

###############################################################################
# Estimate from data


def _estimate_compute_covariance_from_epochs(epochs, bmin, bmax, reject, flat,
                                           keep_sample_mean):
    """Estimate noise covariance matrix from epochs
    """
    picks_no_eog = pick_types(epochs.info, meg=True, eeg=True, eog=False)
    n_channels = len(picks_no_eog)
    ch_names = [epochs.ch_names[k] for k in picks_no_eog]
    data = np.zeros((n_channels, n_channels))
    n_samples = 0
    if bmin is None:
        bmin = epochs.times[0]
    if bmax is None:
        bmax = epochs.times[-1]
    bmask = (epochs.times >= bmin) & (epochs.times <= bmax)

    idx_by_type = channel_indices_by_type(epochs.info)

    for e in epochs:

        if not _is_good(e, epochs.ch_names, idx_by_type, reject, flat):
            print "Artefact detected in epoch"
            continue

        e = e[picks_no_eog]
        mu = e[:, bmask].mean(axis=1)
        e -= mu[:, None]
        if not keep_sample_mean:
            e -= np.mean(e, axis=0)
        data += np.dot(e, e.T)
        n_samples += e.shape[1]

    print "Number of samples used : %d" % n_samples
    print '[done]'
    return data, n_samples, ch_names


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


def compute_covariance(raw, events, event_ids, tmin, tmax,
                     bmin=None, bmax=None, reject=None, flat=None,
                     keep_sample_mean=True):
    """Estimate noise covariance matrix from raw file and events.

    The noise covariance is typically estimated on pre-stim periods
    when the stim onset if defined from events.

    Parameters
    ----------
    raw : Raw instance
        The raw data
    events : array
        The events a.k.a. the triggers
    event_ids : array-like of int
        The valid events to consider
    tmin : float
        Initial time in (s) around trigger. Ex: if tmin=0.2
        then the beginning is 200ms before trigger onset.
    tmax : float
        Final time in (s) around trigger. Ex: if tmax=0.5
        then the end is 500ms after trigger onset.
    bmin : float
        Used to specify a specific baseline for the offset.
        bmin is the init of baseline.
    bmax : float
        bmax is the end of baseline.
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
        The computed covariance.
    """
    # Pick all channels
    picks = pick_types(raw.info, meg=True, eeg=True, eog=True)
    if isinstance(event_ids, int):
        event_ids = list(event_ids)
    data = 0.0
    n_samples = 0

    for event_id in event_ids:
        print "Processing event : %d" % event_id
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            baseline=(None, 0))
        d, n, ch_names = _estimate_compute_covariance_from_epochs(epochs,
                      bmin=bmin, bmax=bmax, reject=reject, flat=flat,
                      keep_sample_mean=keep_sample_mean)
        data += d
        n_samples += n

    if n_samples == 0:
        raise ValueError('Not enough samples to compute the noise covariance'
                         ' matrix : %d samples' % n_samples)

    data /= n_samples - 1
    cov = Covariance(None)
    cov.data = data
    cov.ch_names = ch_names
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
