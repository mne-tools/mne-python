# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np

from .constants import FIFF
from .tag import find_tag, has_tag, read_tag
from .tree import dir_tree_find

from ..utils import logger, verbose


def hex2dec(s):
    return int(s, 16)


def _read_named_matrix(fid, node, matkind):
    """read_named_matrix(fid,node)

    Read named matrix from the given node

    Parameters
    ----------
    fid : file
        The file descriptor
    node : dict
        Node
    matkind : mat kind
        XXX
    Returns
    -------
    mat : dict
        The matrix with row and col names.
    """

    #   Descend one level if necessary
    if node['block'] != FIFF.FIFFB_MNE_NAMED_MATRIX:
        for k in range(node['nchild']):
            if node['children'][k]['block'] == FIFF.FIFFB_MNE_NAMED_MATRIX:
                if has_tag(node['children'][k], matkind):
                    node = node['children'][k]
                    break
        else:
            raise ValueError('Desired named matrix (kind = %d) not'
                             ' available' % matkind)

    else:
        if not has_tag(node, matkind):
            raise ValueError('Desired named matrix (kind = %d) not available'
                                                                    % matkind)

    #   Read everything we need
    tag = find_tag(fid, node, matkind)
    if tag is None:
        raise ValueError('Matrix data missing')
    else:
        data = tag.data

    nrow, ncol = data.shape
    tag = find_tag(fid, node, FIFF.FIFF_MNE_NROW)
    if tag is not None:
        if tag.data != nrow:
            raise ValueError('Number of rows in matrix data and '
                             'FIFF_MNE_NROW tag do not match')

    tag = find_tag(fid, node, FIFF.FIFF_MNE_NCOL)
    if tag is not None:
        if tag.data != ncol:
            raise ValueError('Number of columns in matrix data and '
                             'FIFF_MNE_NCOL tag do not match')

    tag = find_tag(fid, node, FIFF.FIFF_MNE_ROW_NAMES)
    if tag is not None:
        row_names = tag.data
    else:
        row_names = None

    tag = find_tag(fid, node, FIFF.FIFF_MNE_COL_NAMES)
    if tag is not None:
        col_names = tag.data
    else:
        col_names = None

    #   Put it together
    mat = dict(nrow=nrow, ncol=ncol)
    if row_names is not None:
        mat['row_names'] = row_names.split(':')
    else:
        mat['row_names'] = None

    if col_names is not None:
        mat['col_names'] = col_names.split(':')
    else:
        mat['col_names'] = None

    mat['data'] = data.astype(np.float)
    return mat


@verbose
def read_ctf_comp(fid, node, chs, verbose=None):
    """Read the CTF software compensation data from the given node

    Parameters
    ----------
    fid : file
        The file descriptor.
    node : dict
        The node in the FIF tree.
    chs : list
        The list of channels # XXX unclear.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    compdata : list
        The compensation data
    """
    compdata = []
    comps = dir_tree_find(node, FIFF.FIFFB_MNE_CTF_COMP_DATA)

    for node in comps:
        #   Read the data we need
        mat = _read_named_matrix(fid, node, FIFF.FIFF_MNE_CTF_COMP_DATA)
        for p in range(node['nent']):
            kind = node['directory'][p].kind
            pos = node['directory'][p].pos
            if kind == FIFF.FIFF_MNE_CTF_COMP_KIND:
                tag = read_tag(fid, pos)
                break
        else:
            raise Exception('Compensation type not found')

        #   Get the compensation kind and map it to a simple number
        one = dict(ctfkind=tag.data)
        del tag

        if one['ctfkind'] == int('47314252', 16):  # hex2dec('47314252'):
            one['kind'] = 1
        elif one['ctfkind'] == int('47324252', 16):  # hex2dec('47324252'):
            one['kind'] = 2
        elif one['ctfkind'] == int('47334252', 16):  # hex2dec('47334252'):
            one['kind'] = 3
        else:
            one['kind'] = int(one['ctfkind'])

        for p in range(node['nent']):
            kind = node['directory'][p].kind
            pos = node['directory'][p].pos
            if kind == FIFF.FIFF_MNE_CTF_COMP_CALIBRATED:
                tag = read_tag(fid, pos)
                calibrated = tag.data
                break
        else:
            calibrated = False

        one['save_calibrated'] = calibrated
        one['rowcals'] = np.ones(mat['data'].shape[0], dtype=np.float)
        one['colcals'] = np.ones(mat['data'].shape[1], dtype=np.float)

        row_cals, col_cals = None, None  # initialize cals

        if not calibrated:
            #
            #   Calibrate...
            #
            #   Do the columns first
            #
            ch_names = [c['ch_name'] for c in chs]

            col_cals = np.zeros(mat['data'].shape[1], dtype=np.float)
            for col in range(mat['data'].shape[1]):
                p = ch_names.count(mat['col_names'][col])
                if p == 0:
                    raise Exception('Channel %s is not available in data'
                                                % mat['col_names'][col])
                elif p > 1:
                    raise Exception('Ambiguous channel %s' %
                                                        mat['col_names'][col])
                idx = ch_names.index(mat['col_names'][col])
                col_cals[col] = 1.0 / (chs[idx]['range'] * chs[idx]['cal'])

            #    Then the rows
            row_cals = np.zeros(mat['data'].shape[0])
            for row in range(mat['data'].shape[0]):
                p = ch_names.count(mat['row_names'][row])
                if p == 0:
                    raise Exception('Channel %s is not available in data'
                                               % mat['row_names'][row])
                elif p > 1:
                    raise Exception('Ambiguous channel %s' %
                                                mat['row_names'][row])
                idx = ch_names.index(mat['row_names'][row])
                row_cals[row] = chs[idx]['range'] * chs[idx]['cal']

            mat['data'] = row_cals[:, None] * mat['data'] * col_cals[None, :]
            one['rowcals'] = row_cals
            one['colcals'] = col_cals

        one['data'] = mat
        compdata.append(one)
        if row_cals is not None:
            del row_cals
        if col_cals is not None:
            del col_cals

    if len(compdata) > 0:
        logger.info('    Read %d compensation matrices' % len(compdata))

    return compdata


###############################################################################
# Writing

from .write import start_block, end_block, write_int
from .matrix import write_named_matrix


def write_ctf_comp(fid, comps):
    """Write the CTF compensation data into a fif file

    Parameters
    ----------
    fid : file
        The open FIF file descriptor

    comps : list
        The compensation data to write
    """
    if len(comps) <= 0:
        return

    #  This is very simple in fact
    start_block(fid, FIFF.FIFFB_MNE_CTF_COMP)
    for comp in comps:
        start_block(fid, FIFF.FIFFB_MNE_CTF_COMP_DATA)
        #    Write the compensation kind
        write_int(fid, FIFF.FIFF_MNE_CTF_COMP_KIND, comp['ctfkind'])
        write_int(fid, FIFF.FIFF_MNE_CTF_COMP_CALIBRATED,
                  comp['save_calibrated'])

        if not comp['save_calibrated']:
            # Undo calibration
            comp = deepcopy(comp)
            data = ((1. / comp['rowcals'][:, None]) * comp['data']['data']
                    * (1. / comp['colcals'][None, :]))
            comp['data']['data'] = data
        write_named_matrix(fid, FIFF.FIFF_MNE_CTF_COMP_DATA, comp['data'])
        end_block(fid, FIFF.FIFFB_MNE_CTF_COMP_DATA)

    end_block(fid, FIFF.FIFFB_MNE_CTF_COMP)
