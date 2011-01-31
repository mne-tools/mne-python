# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os
import copy
import numpy as np
from scipy import linalg

from .fiff.constants import FIFF
from .fiff.tag import find_tag
from .fiff.tree import dir_tree_find
from .fiff.proj import read_proj
from .fiff.channels import _read_bad_channels

from .fiff.write import start_block, end_block, write_int, write_name_list, \
                       write_double, write_float_matrix, start_file, end_file
from .fiff.proj import write_proj
from .fiff import fiff_open
from .fiff.pick import pick_types, pick_channels_forward


class Covariance(object):
    """Noise covariance matrix"""

    _kind_to_id = dict(full=1, sparse=2, diagonal=3) # XXX : check
    _id_to_kind = {1: 'full', 2: 'sparse', 3: 'diagonal'} # XXX : check

    def __init__(self, kind='full'):
        self.kind = kind

    def load(self, fname):
        """load covariance matrix from FIF file"""

        if self.kind in Covariance._kind_to_id:
            cov_kind = Covariance._kind_to_id[self.kind]
        else:
            raise ValueError, ('Unknown type of covariance. '
                               'Choose between full, sparse or diagonal.')

        # Reading
        fid, tree, _ = fiff_open(fname)
        cov = read_cov(fid, tree, cov_kind)
        fid.close()

        self._cov = cov
        self.data = cov['data']

    def save(self, fname):
        """save covariance matrix in a FIF file"""
        write_cov_file(fname, self._cov)

    def estimate_from_raw(self, raw, picks=None, quantum_sec=10):
        """Estimate noise covariance matrix from a raw FIF file
        """
        #   Set up the reading parameters
        start = raw['first_samp']
        stop = raw['last_samp'] + 1
        quantum = int(quantum_sec * raw['info']['sfreq'])

        cov = 0
        n_samples = 0

        # Read data
        for first in range(start, stop, quantum):
            last = first + quantum
            if last >= stop:
                last = stop

            data, times = raw[picks, first:last]

            if self.kind is 'full':
                cov += np.dot(data, data.T)
            elif self.kind is 'diagonal':
                cov += np.diag(np.sum(data ** 2, axis=1))
            else:
                raise ValueError, "Unsupported covariance kind"

            n_samples += data.shape[1]

        self.data = cov / n_samples # XXX : check
        print '[done]'

    def _regularize(self, data, variances, ch_names, eps):
        """Operates inplace in data
        """
        if len(ch_names) > 0:
            ind = [self._cov['names'].index(name) for name in ch_names]
            reg = eps * np.mean(variances[ind])
            for ii in ind:
                data[ind,ind] += reg

    def whiten_evoked(self, ave, eps=0.2):
        """Whiten an evoked data file

        The whitening matrix is estimated and then multiplied to data.
        It makes the additive white noise assumption of MNE
        realistic.

        Parameters
        ----------
        ave : evoked data
            A evoked data set read with fiff.read_evoked
        eps : float
            The regularization factor used.

        Returns
        -------
        ave : evoked data
            Evoked data set after whitening.
        W : array of shape [n_channels, n_channels]
            The whitening matrix
        """

        data = self.data.copy() # will be the regularized covariance
        variances = np.diag(data)

        # Add (eps x identity matrix) to magnetometers only.
        # This is based on the mean magnetometer variance like MNE C-code does it.
        mag_ind = pick_types(ave['info'], meg='mag', eeg=False, stim=False)
        mag_names = [ave['info']['chs'][k]['ch_name'] for k in mag_ind]
        self._regularize(data, variances, mag_names, eps)

        # Add (eps x identity matrix) to gradiometers only.
        grad_ind = pick_types(ave['info'], meg='grad', eeg=False, stim=False)
        grad_names = [ave['info']['chs'][k]['ch_name'] for k in grad_ind]
        self._regularize(data, variances, grad_names, eps)

        # Add (eps x identity matrix) to eeg only.
        eeg_ind = pick_types(ave['info'], meg=False, eeg=True, stim=False)
        eeg_names = [ave['info']['chs'][k]['ch_name'] for k in eeg_ind]
        self._regularize(data, variances, eeg_names, eps)

        d, V = linalg.eigh(data) # Compute eigen value decomposition.

        # Compute the unique square root inverse, which is a whitening matrix.
        # This matrix can be multiplied with data and leadfield matrix to get
        # whitened inverse solutions.
        d = 1.0 / np.sqrt(d)
        W = np.dot(V, d[:,None] * V.T)

        # Get all channel indices
        n_channels = len(ave['info']['chs'])
        ave_ch_names = [ave['info']['chs'][k]['ch_name']
                                            for k in range(n_channels)]
        ind = [ave_ch_names.index(name) for name in self._cov['names']]

        ave_whiten = copy.copy(ave)
        ave_whiten['evoked']['epochs'][ind] = np.dot(W,
                                                ave['evoked']['epochs'][ind])

        return ave_whiten, W

    def whiten_evoked_and_forward(self, ave, fwd, eps=0.2):
        """Whiten an evoked data set and a forward solution

        The whitening matrix is estimated and then multiplied to
        forward solution a.k.a. the leadfield matrix.
        It makes the additive white noise assumption of MNE
        realistic.

        Parameters
        ----------
        ave : evoked data
            A evoked data set read with fiff.read_evoked
        fwd : forward data
            A forward solution read with mne.read_forward
        eps : float
            The regularization factor used.

        Returns
        -------
        ave : evoked data
            A evoked data set read with fiff.read_evoked
        fwd : evoked data
            Forward solution after whitening.
        W : array of shape [n_channels, n_channels]
            The whitening matrix
        """
        # handle evoked
        ave_whiten, W = self.whiten_evoked(ave, eps=eps)

        ave_ch_names = [ch['ch_name'] for ch in ave_whiten['info']['chs']]

        # handle forward (keep channels in covariance matrix)
        fwd_whiten = copy.copy(fwd)
        ind = [fwd_whiten['sol']['row_names'].index(name)
                                                for name in self._cov['names']]
        fwd_whiten['sol']['data'][ind] = np.dot(W,
                                                fwd_whiten['sol']['data'][ind])
        fwd_whiten['sol']['row_names'] = [fwd_whiten['sol']['row_names'][k]
                                                                  for k in ind]
        fwd_whiten['chs'] = [fwd_whiten['chs'][k] for k in ind]

        # keep in forward the channels in the evoked dataset
        fwd_whiten = pick_channels_forward(fwd, include=ave_ch_names,
                                                exclude=ave['info']['bads'])

        return ave_whiten, fwd_whiten, W

    def __repr__(self):
        s = "kind : %s" % self.kind
        s += ", size : %s x %s" % self.data.shape
        s += ", data : %s" % self.data
        return "Covariance (%s)" % s


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
        raise ValueError, 'No covariance matrices found'

    #   Is any of the covariance matrices a noise covariance
    for p in range(len(covs)):
        tag = find_tag(fid, covs[p], FIFF.FIFF_MNE_COV_KIND)
        if tag is not None and tag.data == cov_kind:
            this = covs[p]

            #   Find all the necessary data
            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIM)
            if tag is None:
                raise ValueError, 'Covariance matrix dimension not found'

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
                    raise ValueError, ('Number of names does not match '
                                       'covariance matrix dimension')

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV)
            if tag is None:
                tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIAG)
                if tag is None:
                    raise ValueError, 'No covariance matrix data found'
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
                    data.flat[::dim+1] /= 2.0
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

    raise ValueError, 'Did not find the desired covariance matrix'

    return None

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
