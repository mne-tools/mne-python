# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

import logging
logger = logging.getLogger('mne')

from .constants import FIFF
from .write import start_block, end_block, write_int, write_name_list, \
                       write_double, write_float_matrix
from .tag import find_tag
from .tree import dir_tree_find
from .proj import read_proj, write_proj
from .channels import read_bad_channels
from .. import verbose


@verbose
def read_cov(fid, node, cov_kind, verbose=None):
    """Read a noise covariance matrix

    This is a low-level function. Consider using mne.cov.read_cov()
    for most user-level purposes.

    Parameters
    ----------
    fid : file
        The file descriptor

    node : dict
        The node in the FIF tree

    cov_kind : int
        The type of covariance. This is typically
        FIFF.FIFFV_MNE_NOISE_COV, although it could also be
        FIFF.FIFFV_MNE_SOURCE_COV or FIFF.FIFFV_MNE_FMRI_PRIOR_COV.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    data : dict
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
            projs = read_proj(fid, this)

            #   Read the bad channel list
            bads = read_bad_channels(fid, this)

            #   Put it together
            cov = dict(kind=cov_kind, diag=diagmat, dim=dim, names=names,
                       data=data, projs=projs, bads=bads, nfree=nfree, eig=eig,
                       eigvec=eigvec)
            return cov

    logger.info('Did not find the desired covariance matrix')

    return None


def write_cov(fid, cov):
    """Write a noise covariance matrix

    Parameters
    ----------
    fid : file
        The file descriptor

    cov : dict
        The noise covariance matrix to write
    """
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
    write_proj(fid, cov['projs'])

    #   Bad channels
    if cov['bads'] is not None:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, cov['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    #   Done!
    end_block(fid, FIFF.FIFFB_MNE_COV)
