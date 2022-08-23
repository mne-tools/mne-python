# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD-3-Clause

from copy import deepcopy

import numpy as np

from .constants import FIFF
from .tag import read_tag
from .tree import dir_tree_find
from .write import start_block, end_block, write_int
from .matrix import write_named_matrix, _read_named_matrix

from ..utils import logger, verbose, _pl


def _add_kind(one):
    """Convert CTF kind to MNE kind."""
    if one['ctfkind'] == int('47314252', 16):
        one['kind'] = 1
    elif one['ctfkind'] == int('47324252', 16):
        one['kind'] = 2
    elif one['ctfkind'] == int('47334252', 16):
        one['kind'] = 3
    else:
        one['kind'] = int(one['ctfkind'])


def _calibrate_comp(comp, chs, row_names, col_names,
                    mult_keys=('range', 'cal'), flip=False):
    """Get row and column cals."""
    ch_names = [c['ch_name'] for c in chs]
    row_cals = np.zeros(len(row_names))
    col_cals = np.zeros(len(col_names))
    for names, cals, inv in zip((row_names, col_names), (row_cals, col_cals),
                                (False, True)):
        for ii in range(len(cals)):
            p = ch_names.count(names[ii])
            if p != 1:
                raise RuntimeError('Channel %s does not appear exactly once '
                                   'in data, found %d instance%s'
                                   % (names[ii], p, _pl(p)))
            idx = ch_names.index(names[ii])
            val = chs[idx][mult_keys[0]] * chs[idx][mult_keys[1]]
            val = float(1. / val) if inv else float(val)
            val = 1. / val if flip else val
            cals[ii] = val
    comp['rowcals'] = row_cals
    comp['colcals'] = col_cals
    comp['data']['data'] = (row_cals[:, None] *
                            comp['data']['data'] * col_cals[None, :])


@verbose
def read_ctf_comp(fid, node, chs, verbose=None):
    """Read the CTF software compensation data from the given node.

    Parameters
    ----------
    fid : file
        The file descriptor.
    node : dict
        The node in the FIF tree.
    chs : list
        The list of channels from info['chs'] to match with
        compensators that are read.
    %(verbose)s

    Returns
    -------
    compdata : list
        The compensation data
    """
    return _read_ctf_comp(fid, node, chs, None)


def _read_ctf_comp(fid, node, chs, ch_names_mapping):
    """Read the CTF software compensation data from the given node.

    Parameters
    ----------
    fid : file
        The file descriptor.
    node : dict
        The node in the FIF tree.
    chs : list
        The list of channels from info['chs'] to match with
        compensators that are read.
    ch_names_mapping : dict | None
        The channel renaming to use.
    %(verbose)s

    Returns
    -------
    compdata : list
        The compensation data
    """
    from .meas_info import _rename_comps
    ch_names_mapping = dict() if ch_names_mapping is None else ch_names_mapping
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
        _add_kind(one)
        for p in range(node['nent']):
            kind = node['directory'][p].kind
            pos = node['directory'][p].pos
            if kind == FIFF.FIFF_MNE_CTF_COMP_CALIBRATED:
                tag = read_tag(fid, pos)
                calibrated = tag.data
                break
        else:
            calibrated = False

        one['save_calibrated'] = bool(calibrated)
        one['data'] = mat
        _rename_comps([one], ch_names_mapping)
        if not calibrated:
            #   Calibrate...
            _calibrate_comp(one, chs, mat['row_names'], mat['col_names'])
        else:
            one['rowcals'] = np.ones(mat['data'].shape[0], dtype=np.float64)
            one['colcals'] = np.ones(mat['data'].shape[1], dtype=np.float64)

        compdata.append(one)

    if len(compdata) > 0:
        logger.info('    Read %d compensation matrices' % len(compdata))

    return compdata


###############################################################################
# Writing

def write_ctf_comp(fid, comps):
    """Write the CTF compensation data into a fif file.

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
        if comp.get('save_calibrated', False):
            write_int(fid, FIFF.FIFF_MNE_CTF_COMP_CALIBRATED,
                      comp['save_calibrated'])

        if not comp.get('save_calibrated', True):
            # Undo calibration
            comp = deepcopy(comp)
            data = ((1. / comp['rowcals'][:, None]) * comp['data']['data'] *
                    (1. / comp['colcals'][None, :]))
            comp['data']['data'] = data
        write_named_matrix(fid, FIFF.FIFF_MNE_CTF_COMP_DATA, comp['data'])
        end_block(fid, FIFF.FIFFB_MNE_CTF_COMP_DATA)

    end_block(fid, FIFF.FIFFB_MNE_CTF_COMP)
