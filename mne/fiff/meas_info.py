# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from warnings import warn
from copy import deepcopy
import os.path as op
import numpy as np
from scipy import linalg
from StringIO import StringIO
from datetime import datetime as dt

from .open import fiff_open
from .tree import dir_tree_find, copy_tree
from .constants import FIFF
from .tag import read_tag
from .proj import read_proj, write_proj
from .ctf import read_ctf_comp, write_ctf_comp
from .channels import read_bad_channels
from .write import (start_file, end_file, start_block, end_block,
                    write_string, write_dig_point, write_float, write_int,
                    write_coord_trans, write_ch_info, write_name_list)
from ..utils import logger, verbose


def _summarize_str(st):
    """Aux function"""
    return st[:56][::-1].split(',', 1)[-1][::-1] + ', ...'


class Info(dict):
    """ Info class to nicely represent info dicts
    """

    def __repr__(self):
        """Summarize info instead of printing all"""
        strs = ['<Info | %s non-empty fields']
        non_empty = 0
        for k, v in self.items():
            if k in ['bads', 'ch_names']:
                entr = (', '.join(b for ii, b in enumerate(v) if ii < 10)
                           if v else '0 items')
                if len(entr) >= 56:
                    # get rid of of half printed ch names
                    entr = _summarize_str(entr)
            elif k == 'filename' and v:
                path, fname = op.split(v)
                entr = path[:10] + '.../' + fname
            elif k == 'projs' and v:
                entr = ', '.join(p['desc'] + ': o%s' %
                                 {0: 'ff', 1: 'n'}[p['active']] for p in v)
                if len(entr) >= 56:
                    entr = _summarize_str(entr)
            elif k == 'meas_date' and np.iterable(v):
                # first entire in meas_date is meaningful
                entr = dt.fromtimestamp(v[0]).strftime('%Y-%m-%d %H:%M:%S')
            else:
                this_len = (len(v) if hasattr(v, '__len__') else
                           ('%s' % v if v is not None else None))
                entr = (('%d items' % this_len) if isinstance(this_len, int)
                           else ('%s' % this_len if this_len else ''))
            if entr:
                non_empty += 1
                entr = ' | ' + entr
            strs.append('%s : %s%s' % (k, str(type(v))[7:-2], entr))
        strs_non_empty = sorted(s for s in strs if '|' in s)
        strs_empty = sorted(s for s in strs if '|' not in s)
        st = '\n    '.join(strs_non_empty + strs_empty)
        st += '\n>'
        st %= non_empty
        return st


def read_fiducials(fname):
    """Read fiducials from a fiff file

    Returns
    -------
    pts : list of dicts
        List of digitizer points (each point in a dict).
    coord_frame : int
        The coordinate frame of the points (one of
        mne.fiff.FIFF.FIFFV_COORD_...)
    """
    fid, tree, _ = fiff_open(fname)
    with fid:
        isotrak = dir_tree_find(tree, FIFF.FIFFB_ISOTRAK)
        isotrak = isotrak[0]
        pts = []
        coord_frame = FIFF.FIFFV_COORD_UNKNOWN
        for k in range(isotrak['nent']):
            kind = isotrak['directory'][k].kind
            pos = isotrak['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                pts.append(tag.data)
            elif kind == FIFF.FIFF_MNE_COORD_FRAME:
                tag = read_tag(fid, pos)
                coord_frame = tag.data[0]

    if coord_frame == FIFF.FIFFV_COORD_UNKNOWN:
        err = ("No coordinate frame was found in the file %r, it is probably "
               "not a valid fiducials file." % fname)
        raise ValueError(err)

    # coord_frame is not stored in the tag
    for pt in pts:
        pt['coord_frame'] = coord_frame

    return pts, coord_frame


def write_fiducials(fname, pts, coord_frame=0):
    """Write fiducials to a fiff file

    Parameters
    ----------
    fname : str
        Destination file name.
    pts : iterator of dict
        Iterator through digitizer points. Each point is a dictionary with
        the keys 'kind', 'ident' and 'r'.
    coord_frame : int
        The coordinate frame of the points (one of
        mne.fiff.FIFF.FIFFV_COORD_...)
    """
    pts_frames = set((pt.get('coord_frame', coord_frame) for pt in pts))
    bad_frames = pts_frames - set((coord_frame,))
    if len(bad_frames) > 0:
        err = ("Points have coord_frame entries that are incompatible with "
               "coord_frame=%i: %s." % (coord_frame, str(tuple(bad_frames))))
        raise ValueError(err)

    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_ISOTRAK)
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, coord_frame)
    for pt in pts:
        write_dig_point(fid, pt)

    end_block(fid, FIFF.FIFFB_ISOTRAK)
    end_file(fid)


@verbose
def read_info(fname, verbose=None):
    """Read measurement info from a file

    Parameters
    ----------
    fname : str
        File name.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    info : instance of mne.fiff.meas_info.Info
       Info on dataset.
    """
    f, tree, _ = fiff_open(fname)
    with f as fid:
        info = read_meas_info(fid, tree)[0]
    return info


@verbose
def read_meas_info(fid, tree, verbose=None):
    """Read the measurement info

    Parameters
    ----------
    fid : file
        Open file descriptor.
    tree : tree
        FIF tree structure.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    info : instance of mne.fiff.meas_info.Info
       Info on dataset.
    meas : dict
        Node in tree that contains the info.
    """
    #   Find the desired blocks
    meas = dir_tree_find(tree, FIFF.FIFFB_MEAS)
    if len(meas) == 0:
        raise ValueError('Could not find measurement data')
    if len(meas) > 1:
        raise ValueError('Cannot read more that 1 measurement data')
    meas = meas[0]

    meas_info = dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    if len(meas_info) == 0:
        raise ValueError('Could not find measurement info')
    if len(meas_info) > 1:
        raise ValueError('Cannot read more that 1 measurement info')
    meas_info = meas_info[0]

    #   Read measurement info
    dev_head_t = None
    ctf_head_t = None
    meas_date = None
    highpass = None
    lowpass = None
    nchan = None
    sfreq = None
    chs = []
    experimenter = None
    description = None
    proj_id = None
    proj_name = None
    p = 0
    for k in range(meas_info['nent']):
        kind = meas_info['directory'][k].kind
        pos = meas_info['directory'][k].pos
        if kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = int(tag.data)
        elif kind == FIFF.FIFF_SFREQ:
            tag = read_tag(fid, pos)
            sfreq = float(tag.data)
        elif kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
            p += 1
        elif kind == FIFF.FIFF_LOWPASS:
            tag = read_tag(fid, pos)
            lowpass = float(tag.data)
        elif kind == FIFF.FIFF_HIGHPASS:
            tag = read_tag(fid, pos)
            highpass = float(tag.data)
        elif kind == FIFF.FIFF_MEAS_DATE:
            tag = read_tag(fid, pos)
            meas_date = tag.data
        elif kind == FIFF.FIFF_COORD_TRANS:
            tag = read_tag(fid, pos)
            cand = tag.data
            if cand['from'] == FIFF.FIFFV_COORD_DEVICE and \
                                cand['to'] == FIFF.FIFFV_COORD_HEAD:
                dev_head_t = cand
            elif cand['from'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD and \
                                cand['to'] == FIFF.FIFFV_COORD_HEAD:
                ctf_head_t = cand
        elif kind == FIFF.FIFF_EXPERIMENTER:
            tag = read_tag(fid, pos)
            experimenter = tag.data
        elif kind == FIFF.FIFF_DESCRIPTION:
            tag = read_tag(fid, pos)
            description = tag.data
        elif kind == FIFF.FIFF_PROJ_ID:
            tag = read_tag(fid, pos)
            proj_id = tag.data
        elif kind == FIFF.FIFF_PROJ_NAME:
            tag = read_tag(fid, pos)
            proj_name = tag.data

    # Check that we have everything we need
    if nchan is None:
        raise ValueError('Number of channels in not defined')

    if sfreq is None:
        raise ValueError('Sampling frequency is not defined')

    if len(chs) == 0:
        raise ValueError('Channel information not defined')

    if len(chs) != nchan:
        raise ValueError('Incorrect number of channel definitions found')

    if dev_head_t is None or ctf_head_t is None:
        hpi_result = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
        if len(hpi_result) == 1:
            hpi_result = hpi_result[0]
            for k in range(hpi_result['nent']):
                kind = hpi_result['directory'][k].kind
                pos = hpi_result['directory'][k].pos
                if kind == FIFF.FIFF_COORD_TRANS:
                    tag = read_tag(fid, pos)
                    cand = tag.data
                    if cand['from'] == FIFF.FIFFV_COORD_DEVICE and \
                                cand['to'] == FIFF.FIFFV_COORD_HEAD:
                        dev_head_t = cand
                    elif cand['from'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD and \
                                cand['to'] == FIFF.FIFFV_COORD_HEAD:
                        ctf_head_t = cand

    #   Locate the Polhemus data
    isotrak = dir_tree_find(meas_info, FIFF.FIFFB_ISOTRAK)
    dig = None
    if len(isotrak) == 0:
        logger.info('Isotrak not found')
    elif len(isotrak) > 1:
        warn('Multiple Isotrak found')
    else:
        isotrak = isotrak[0]
        dig = []
        for k in range(isotrak['nent']):
            kind = isotrak['directory'][k].kind
            pos = isotrak['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                dig.append(tag.data)
                dig[-1]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    #   Locate the acquisition information
    acqpars = dir_tree_find(meas_info, FIFF.FIFFB_DACQ_PARS)
    acq_pars = None
    acq_stim = None
    if len(acqpars) == 1:
        acqpars = acqpars[0]
        for k in range(acqpars['nent']):
            kind = acqpars['directory'][k].kind
            pos = acqpars['directory'][k].pos
            if kind == FIFF.FIFF_DACQ_PARS:
                tag = read_tag(fid, pos)
                acq_pars = tag.data
            elif kind == FIFF.FIFF_DACQ_STIM:
                tag = read_tag(fid, pos)
                acq_stim = tag.data

    #   Load the SSP data
    projs = read_proj(fid, meas_info)

    #   Load the CTF compensation data
    comps = read_ctf_comp(fid, meas_info, chs)

    #   Load the bad channel list
    bads = read_bad_channels(fid, meas_info)

    #
    #   Put the data together
    #
    if tree['id'] is not None:
        info = Info(file_id=tree['id'])
    else:
        info = Info(file_id=None)

    #   Load extra information blocks
    read_extra_meas_info(fid, tree, info)

    #  Make the most appropriate selection for the measurement id
    if meas_info['parent_id'] is None:
        if meas_info['id'] is None:
            if meas['id'] is None:
                if meas['parent_id'] is None:
                    info['meas_id'] = info['file_id']
                else:
                    info['meas_id'] = meas['parent_id']
            else:
                info['meas_id'] = meas['id']
        else:
            info['meas_id'] = meas_info['id']
    else:
        info['meas_id'] = meas_info['parent_id']

    info['experimenter'] = experimenter
    info['description'] = description
    info['proj_id'] = proj_id
    info['proj_name'] = proj_name

    if meas_date is None:
        info['meas_date'] = [info['meas_id']['secs'], info['meas_id']['usecs']]
    else:
        info['meas_date'] = meas_date

    info['nchan'] = nchan
    info['sfreq'] = sfreq
    info['highpass'] = highpass if highpass is not None else 0
    info['lowpass'] = lowpass if lowpass is not None else info['sfreq'] / 2.0

    #   Add the channel information and make a list of channel names
    #   for convenience
    info['chs'] = chs
    info['ch_names'] = [ch['ch_name'] for ch in chs]

    #
    #  Add the coordinate transformations
    #
    info['dev_head_t'] = dev_head_t
    info['ctf_head_t'] = ctf_head_t
    if dev_head_t is not None and ctf_head_t is not None:
        head_ctf_trans = linalg.inv(ctf_head_t['trans'])
        dev_ctf_trans = np.dot(head_ctf_trans, info['dev_head_t']['trans'])
        info['dev_ctf_t'] = {'from': FIFF.FIFFV_COORD_DEVICE,
                             'to': FIFF.FIFFV_MNE_COORD_CTF_HEAD,
                             'trans': dev_ctf_trans}
    else:
        info['dev_ctf_t'] = None

    #   All kinds of auxliary stuff
    info['dig'] = dig
    info['bads'] = bads
    info['projs'] = projs
    info['comps'] = comps
    info['acq_pars'] = acq_pars
    info['acq_stim'] = acq_stim

    return info, meas


def read_extra_meas_info(fid, tree, info):
    """Read extra blocks from fid"""
    # current method saves them into a cStringIO file instance for simplicity
    # this and its partner, write_extra_meas_info, could be made more
    # comprehensive (i.e.., actually parse and read the data instead of
    # just storing it for later)
    blocks = [FIFF.FIFFB_SUBJECT, FIFF.FIFFB_EVENTS,
              FIFF.FIFFB_HPI_RESULT, FIFF.FIFFB_HPI_MEAS,
              FIFF.FIFFB_PROCESSING_HISTORY]
    info['orig_blocks'] = blocks

    fid_str = StringIO()
    fid_str = start_file(fid_str)
    start_block(fid_str, FIFF.FIFFB_MEAS_INFO)
    for block in blocks:
        nodes = dir_tree_find(tree, block)
        copy_tree(fid, tree['id'], nodes, fid_str)
    info['orig_fid_str'] = fid_str


def write_extra_meas_info(fid, info):
    """Write otherwise left out blocks of data"""
    # uses cStringIO fake file to read the appropriate blocks
    if 'orig_blocks' in info and info['orig_blocks'] is not None:
        # Blocks from the original
        blocks = info['orig_blocks']
        fid_str, tree, _ = fiff_open(info['orig_fid_str'])
        for block in blocks:
            nodes = dir_tree_find(tree, block)
            copy_tree(fid_str, tree['id'], nodes, fid)


def write_meas_info(fid, info, data_type=None, reset_range=True):
    """Write measurement info in fif file.

    Parameters
    ----------
    fid : file
        Open file descriptor
    info : instance of mne.fiff.meas_info.Info
        The measurement info structure
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (mne.fiff.FIFF.FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.

    Note
    ----
    Tags are written in a particular order for compatibility with maxfilter
    """

    # Measurement info
    start_block(fid, FIFF.FIFFB_MEAS_INFO)

    #   Extra measurement info
    write_extra_meas_info(fid, info)

    #   Polhemus data
    if info['dig'] is not None:
        start_block(fid, FIFF.FIFFB_ISOTRAK)
        for d in info['dig']:
            write_dig_point(fid, d)

        end_block(fid, FIFF.FIFFB_ISOTRAK)

    #   megacq parameters
    if info['acq_pars'] is not None or info['acq_stim'] is not None:
        start_block(fid, FIFF.FIFFB_DACQ_PARS)
        if info['acq_pars'] is not None:
            write_string(fid, FIFF.FIFF_DACQ_PARS, info['acq_pars'])

        if info['acq_stim'] is not None:
            write_string(fid, FIFF.FIFF_DACQ_STIM, info['acq_stim'])

        end_block(fid, FIFF.FIFFB_DACQ_PARS)

    #   Coordinate transformations if the HPI result block was not there
    if info['dev_head_t'] is not None:
        write_coord_trans(fid, info['dev_head_t'])

    if info['ctf_head_t'] is not None:
        write_coord_trans(fid, info['ctf_head_t'])

    #   Projectors
    write_proj(fid, info['projs'])

    #   CTF compensation info
    write_ctf_comp(fid, info['comps'])

    #   Bad channels
    if len(info['bads']) > 0:
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, info['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    #   General
    if info.get('experimenter') is not None:
        write_string(fid, FIFF.FIFF_EXPERIMENTER, info['experimenter'])
    if info.get('description') is not None:
        write_string(fid, FIFF.FIFF_DESCRIPTION, info['description'])
    if info.get('proj_id') is not None:
        write_int(fid, FIFF.FIFF_PROJ_ID, info['proj_id'])
    if info.get('proj_name') is not None:
        write_string(fid, FIFF.FIFF_PROJ_NAME, info['proj_name'])
    if info.get('meas_date') is not None:
        write_int(fid, FIFF.FIFF_MEAS_DATE, info['meas_date'])
    write_int(fid, FIFF.FIFF_NCHAN, info['nchan'])
    write_float(fid, FIFF.FIFF_SFREQ, info['sfreq'])
    write_float(fid, FIFF.FIFF_LOWPASS, info['lowpass'])
    write_float(fid, FIFF.FIFF_HIGHPASS, info['highpass'])
    if data_type is not None:
        write_int(fid, FIFF.FIFF_DATA_PACK, data_type)

    #  Channel information
    for k, c in enumerate(info['chs']):
        #   Scan numbers may have been messed up
        c = deepcopy(c)
        c['scanno'] = k + 1
        # for float/double, the "range" param is unnecessary
        if reset_range is True:
            c['range'] = 1.0
        write_ch_info(fid, c)

    end_block(fid, FIFF.FIFFB_MEAS_INFO)
