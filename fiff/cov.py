import os
import numpy as np

from .constants import FIFF
from .tag import find_tag
from .tree import dir_tree_find
from .proj import read_proj
from .channels import _read_bad_channels


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
                    raise ValueError, 'Number of names does not match covariance matrix dimension'

            tag = find_tag(fid, this, FIFF.FIFF_MNE_COV)
            if tag is None:
                tag = find_tag(fid, this, FIFF.FIFF_MNE_COV_DIAG)
                if tag is None:
                    raise ValueError, 'No covariance matrix data found'
                else:
                    #   Diagonal is stored
                    data = tag.data
                    diagmat = True
                    print '\t%d x %d diagonal covariance (kind = %d) found.' % (dim, dim, cov_kind)

            else:
                from scipy import sparse
                if not sparse.issparse(tag.data):
                    #   Lower diagonal is stored
                    vals = tag.data
                    data = np.zeros((dim, dim))
                    q = 0
                    for j in range(dim):
                        for k in range(j):
                            data[j, k] = vals[q];
                            q += 1
                    data = data + data.T
                    data.flat[::dim+1] /= 2.0

                    diagmat = False;
                    print '\t%d x %d full covariance (kind = %d) found.' % (dim, dim, cov_kind)
                else:
                    diagmat = False
                    data = tag.data
                    print '\t%d x %d sparse covariance (kind = %d) found.' % (dim, dim, cov_kind)

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

from .write import start_block, end_block, write_int, write_name_list, \
                   write_double, write_float_matrix, start_file, end_file
from .proj import write_proj


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
        write_int(fid, FIFF.FIFF_MNE_COV_NFREE, cov['nfree']);

    #   Channel names
    if cov['names'] is not None:
        write_name_list(fid, FIFF.FIFF_MNE_ROW_NAMES, cov['names'])

    #   Data
    if cov['diag']:
        write_double(fid, FIFF.FIFF_MNE_COV_DIAG, cov['data']);
    else:
        dim = cov['dim']
        vals = np.empty(dim*(dim + 1)/2)
        # XXX : should be improved later
        q = 0
        for j in range(dim):
            for k in range(j):
                vals[q] = cov['data'][j,k]
                q = q + 1

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
