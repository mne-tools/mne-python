# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from warnings import warn
from copy import deepcopy
from datetime import datetime as dt
import os.path as op

import numpy as np
from scipy import linalg

from .pick import channel_type
from .constants import FIFF
from .open import fiff_open
from .tree import dir_tree_find
from .tag import read_tag, find_tag
from .proj import _read_proj, _write_proj, _uniquify_projs
from .ctf import read_ctf_comp, write_ctf_comp
from .write import (start_file, end_file, start_block, end_block,
                    write_string, write_dig_point, write_float, write_int,
                    write_coord_trans, write_ch_info, write_name_list,
                    write_julian, write_float_matrix)
from .proc_history import _read_proc_history, _write_proc_history
from ..utils import logger, verbose
from ..fixes import Counter
from .. import __version__
from ..externals.six import b, BytesIO, string_types, text_type


_kind_dict = dict(
    eeg=(FIFF.FIFFV_EEG_CH, FIFF.FIFFV_COIL_EEG, FIFF.FIFF_UNIT_V),
    mag=(FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_VV_MAG_T3, FIFF.FIFF_UNIT_T),
    grad=(FIFF.FIFFV_MEG_CH, FIFF.FIFFV_COIL_VV_PLANAR_T1, FIFF.FIFF_UNIT_T_M),
    misc=(FIFF.FIFFV_MISC_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_NONE),
    stim=(FIFF.FIFFV_STIM_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    eog=(FIFF.FIFFV_EOG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    ecg=(FIFF.FIFFV_ECG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
    seeg=(FIFF.FIFFV_SEEG_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V),
)


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
            if k == 'chs':
                ch_types = [channel_type(self, idx) for idx in range(len(v))]
                ch_counts = Counter(ch_types)
                entr += " (%s)" % ', '.join("%s: %d" % (ch_type.upper(), count)
                                            for ch_type, count
                                            in ch_counts.items())
            strs.append('%s : %s%s' % (k, str(type(v))[7:-2], entr))
        strs_non_empty = sorted(s for s in strs if '|' in s)
        strs_empty = sorted(s for s in strs if '|' not in s)
        st = '\n    '.join(strs_non_empty + strs_empty)
        st += '\n>'
        st %= non_empty
        return st

    def _anonymize(self):
        if self.get('subject_info') is not None:
            del self['subject_info']


def read_fiducials(fname):
    """Read fiducials from a fiff file

    Parameters
    ----------
    fname : str
        The filename to read.

    Returns
    -------
    pts : list of dicts
        List of digitizer points (each point in a dict).
    coord_frame : int
        The coordinate frame of the points (one of
        mne.io.constants.FIFF.FIFFV_COORD_...)
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
        mne.io.constants.FIFF.FIFFV_COORD_...)
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


def _read_dig_points(fname, comments='%'):
    """Read digitizer data from file.

    This function can read space-delimited text files of digitizer data.

    Parameters
    ----------
    fname : str
        The filepath of space delimited file with points.
    comments : str
        The character used to indicate the start of a comment;
        Default: '%'.

    Returns
    -------
    dig_points : np.ndarray, shape (n_points, 3)
        Array of dig points.
    """
    dig_points = np.loadtxt(fname, comments=comments, ndmin=2)
    if dig_points.shape[-1] != 3:
        err = 'Data must be (n, 3) instead of %s' % (dig_points.shape,)
        raise ValueError(err)

    return dig_points


def _write_dig_points(fname, dig_points):
    """Write points to file

    Parameters
    ----------
    fname : str
        Path to the file to write. The kind of file to write is determined
        based on the extension: '.txt' for tab separated text file.
    dig_points : numpy.ndarray, shape (n_points, 3)
        Points.
    """
    _, ext = op.splitext(fname)
    dig_points = np.asarray(dig_points)
    if (dig_points.ndim != 2) or (dig_points.shape[1] != 3):
        err = ("Points must be of shape (n_points, 3), "
               "not %s" % (dig_points.shape,))
        raise ValueError(err)

    if ext == '.txt':
        with open(fname, 'wb') as fid:
            version = __version__
            now = dt.now().strftime("%I:%M%p on %B %d, %Y")
            fid.write(b("% Ascii 3D points file created by mne-python version "
                        "{version} at {now}\n".format(version=version,
                                                      now=now)))
            fid.write(b("% {N} 3D points, "
                        "x y z per line\n".format(N=len(dig_points))))
            np.savetxt(fid, dig_points, delimiter='\t', newline='\n')
    else:
        msg = "Unrecognized extension: %r. Need '.txt'." % ext
        raise ValueError(msg)


def _make_dig_points(nasion=None, lpa=None, rpa=None, hpi=None,
                     dig_points=None):
    """Constructs digitizer info for the info.

    Parameters
    ----------
    nasion : array-like | numpy.ndarray, shape (3,) | None
        Point designated as the nasion point.
    lpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the left auricular point.
    rpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the right auricular point.
    hpi : array-like | numpy.ndarray, shape (n_points, 3) | None
        Points designated as head position indicator points.
    dig_points : array-like | numpy.ndarray, shape (n_points, 3)
        Points designed as the headshape points.

    Returns
    -------
    dig : list
        List of digitizer points to be added to the info['dig'].
    """
    dig = []
    if nasion is not None:
        nasion = np.asarray(nasion)
        if nasion.shape == (3,):
            dig.append({'r': nasion, 'ident': FIFF.FIFFV_POINT_NASION,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame':  FIFF.FIFFV_COORD_HEAD})
        else:
            msg = ('Nasion should have the shape (3,) instead of %s'
                   % (nasion.shape,))
            raise ValueError(msg)
    if lpa is not None:
        lpa = np.asarray(lpa)
        if lpa.shape == (3,):
            dig.append({'r': lpa, 'ident': FIFF.FIFFV_POINT_LPA,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame':  FIFF.FIFFV_COORD_HEAD})
        else:
            msg = ('LPA should have the shape (3,) instead of %s'
                   % (lpa.shape,))
            raise ValueError(msg)
    if rpa is not None:
        rpa = np.asarray(rpa)
        if rpa.shape == (3,):
            dig.append({'r': rpa, 'ident': FIFF.FIFFV_POINT_RPA,
                        'kind': FIFF.FIFFV_POINT_CARDINAL,
                        'coord_frame':  FIFF.FIFFV_COORD_HEAD})
        else:
            msg = ('RPA should have the shape (3,) instead of %s'
                   % (rpa.shape,))
            raise ValueError(msg)
    if hpi is not None:
        hpi = np.asarray(hpi)
        if hpi.shape[1] == 3:
            for idx, point in enumerate(hpi):
                dig.append({'r': point, 'ident': idx,
                            'kind': FIFF.FIFFV_POINT_HPI,
                            'coord_frame': FIFF.FIFFV_COORD_HEAD})
        else:
            msg = ('HPI should have the shape (n_points, 3) instead of '
                   '%s' % (hpi.shape,))
            raise ValueError(msg)
    if dig_points is not None:
        dig_points = np.asarray(dig_points)
        if dig_points.shape[1] == 3:
            for idx, point in enumerate(dig_points):
                dig.append({'r': point, 'ident': idx,
                            'kind': FIFF.FIFFV_POINT_EXTRA,
                            'coord_frame': FIFF.FIFFV_COORD_HEAD})
        else:
            msg = ('Points should have the shape (n_points, 3) instead of '
                   '%s' % (dig_points.shape,))
            raise ValueError(msg)

    return dig


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
    info : instance of mne.io.meas_info.Info
       Info on dataset.
    """
    f, tree, _ = fiff_open(fname)
    with f as fid:
        info = read_meas_info(fid, tree)[0]
    return info


def read_bad_channels(fid, node):
    """Read bad channels

    Parameters
    ----------
    fid : file
        The file descriptor.

    node : dict
        The node of the FIF tree that contains info on the bad channels.

    Returns
    -------
    bads : list
        A list of bad channel's names.
    """
    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = []
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None and tag.data is not None:
                bads = tag.data.split(':')
    return bads


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
    info : instance of mne.io.meas_info.Info
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
    line_freq = None
    custom_ref_applied = False
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
        elif kind == FIFF.FIFF_LINE_FREQ:
            tag = read_tag(fid, pos)
            line_freq = float(tag.data)
        elif kind == FIFF.FIFF_CUSTOM_REF:
            tag = read_tag(fid, pos)
            custom_ref_applied = bool(tag.data)

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
                    if (cand['from'] == FIFF.FIFFV_COORD_DEVICE and
                            cand['to'] == FIFF.FIFFV_COORD_HEAD and
                            dev_head_t is None):
                        dev_head_t = cand
                    elif (cand['from'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD and
                          cand['to'] == FIFF.FIFFV_COORD_HEAD and
                          ctf_head_t is None):
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
    projs = _read_proj(fid, meas_info)

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

    #   Locate events list
    events = dir_tree_find(meas_info, FIFF.FIFFB_EVENTS)
    evs = list()
    for event in events:
        ev = dict()
        for k in range(event['nent']):
            kind = event['directory'][k].kind
            pos = event['directory'][k].pos
            if kind == FIFF.FIFF_EVENT_CHANNELS:
                ev['channels'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_EVENT_LIST:
                ev['list'] = read_tag(fid, pos).data
        evs.append(ev)
    info['events'] = evs

    #   Locate HPI result
    hpi_results = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
    hrs = list()
    for hpi_result in hpi_results:
        hr = dict()
        hr['dig_points'] = []
        for k in range(hpi_result['nent']):
            kind = hpi_result['directory'][k].kind
            pos = hpi_result['directory'][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                hr['dig_points'].append(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_DIGITIZATION_ORDER:
                hr['order'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_COILS_USED:
                hr['used'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_COIL_MOMENTS:
                hr['moments'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_FIT_GOODNESS:
                hr['goodness'] = read_tag(fid, pos).data
            elif kind == FIFF.FIFF_HPI_FIT_GOOD_LIMIT:
                hr['good_limit'] = float(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_FIT_DIST_LIMIT:
                hr['dist_limit'] = float(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_FIT_ACCEPT:
                hr['accept'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_COORD_TRANS:
                hr['coord_trans'] = read_tag(fid, pos).data
        hrs.append(hr)
    info['hpi_results'] = hrs

    #   Locate HPI Measurement
    hpi_meass = dir_tree_find(meas_info, FIFF.FIFFB_HPI_MEAS)
    hms = list()
    for hpi_meas in hpi_meass:
        hm = dict()
        for k in range(hpi_meas['nent']):
            kind = hpi_meas['directory'][k].kind
            pos = hpi_meas['directory'][k].pos
            if kind == FIFF.FIFF_CREATOR:
                hm['creator'] = text_type(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_SFREQ:
                hm['sfreq'] = float(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_NCHAN:
                hm['nchan'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_NAVE:
                hm['nave'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_HPI_NCOIL:
                hm['ncoil'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_FIRST_SAMPLE:
                hm['first_samp'] = int(read_tag(fid, pos).data)
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                hm['last_samp'] = int(read_tag(fid, pos).data)
        hpi_coils = dir_tree_find(hpi_meas, FIFF.FIFFB_HPI_COIL)
        hcs = []
        for hpi_coil in hpi_coils:
            hc = dict()
            for k in range(hpi_coil['nent']):
                kind = hpi_coil['directory'][k].kind
                pos = hpi_coil['directory'][k].pos
                if kind == FIFF.FIFF_HPI_COIL_NO:
                    hc['number'] = int(read_tag(fid, pos).data)
                elif kind == FIFF.FIFF_EPOCH:
                    hc['epoch'] = read_tag(fid, pos).data
                elif kind == FIFF.FIFF_HPI_SLOPES:
                    hc['slopes'] = read_tag(fid, pos).data
                elif kind == FIFF.FIFF_HPI_CORR_COEFF:
                    hc['corr_coeff'] = read_tag(fid, pos).data
                elif kind == FIFF.FIFF_CUSTOM_REF:
                    hc['custom_ref'] = read_tag(fid, pos).data
            hcs.append(hc)
        hm['hpi_coils'] = hcs
        hms.append(hm)
    info['hpi_meas'] = hms

    subject_info = dir_tree_find(meas_info, FIFF.FIFFB_SUBJECT)
    si = None
    if len(subject_info) == 1:
        subject_info = subject_info[0]
        si = dict()
        for k in range(subject_info['nent']):
            kind = subject_info['directory'][k].kind
            pos = subject_info['directory'][k].pos
            if kind == FIFF.FIFF_SUBJ_ID:
                tag = read_tag(fid, pos)
                si['id'] = int(tag.data)
            elif kind == FIFF.FIFF_SUBJ_HIS_ID:
                tag = read_tag(fid, pos)
                si['his_id'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_LAST_NAME:
                tag = read_tag(fid, pos)
                si['last_name'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_FIRST_NAME:
                tag = read_tag(fid, pos)
                si['first_name'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_MIDDLE_NAME:
                tag = read_tag(fid, pos)
                si['middle_name'] = text_type(tag.data)
            elif kind == FIFF.FIFF_SUBJ_BIRTH_DAY:
                tag = read_tag(fid, pos)
                si['birthday'] = tag.data
            elif kind == FIFF.FIFF_SUBJ_SEX:
                tag = read_tag(fid, pos)
                si['sex'] = int(tag.data)
            elif kind == FIFF.FIFF_SUBJ_HAND:
                tag = read_tag(fid, pos)
                si['hand'] = int(tag.data)
    info['subject_info'] = si

    hpi_subsystem = dir_tree_find(meas_info, FIFF.FIFFB_HPI_SUBSYSTEM)
    hs = None
    if len(hpi_subsystem) == 1:
        hpi_subsystem = hpi_subsystem[0]
        hs = dict()
        for k in range(hpi_subsystem['nent']):
            kind = hpi_subsystem['directory'][k].kind
            pos = hpi_subsystem['directory'][k].pos
            if kind == FIFF.FIFF_HPI_NCOIL:
                tag = read_tag(fid, pos)
                hs['ncoil'] = int(tag.data)
            elif kind == FIFF.FIFF_EVENT_CHANNEL:
                tag = read_tag(fid, pos)
                hs['event_channel'] = text_type(tag.data)
            hpi_coils = dir_tree_find(hpi_subsystem, FIFF.FIFFB_HPI_COIL)
            hc = []
            for coil in hpi_coils:
                this_coil = dict()
                for j in range(coil['nent']):
                    kind = coil['directory'][j].kind
                    pos = coil['directory'][j].pos
                    if kind == FIFF.FIFF_EVENT_BITS:
                        tag = read_tag(fid, pos)
                        this_coil['event_bits'] = np.array(tag.data)
                hc.append(this_coil)
            hs['hpi_coils'] = hc
    info['hpi_subsystem'] = hs

    #   Read processing history
    _read_proc_history(fid, tree, info)

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
    info['line_freq'] = line_freq

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
    info['custom_ref_applied'] = custom_ref_applied

    return info, meas


def write_meas_info(fid, info, data_type=None, reset_range=True):
    """Write measurement info into a file id (from a fif file)

    Parameters
    ----------
    fid : file
        Open file descriptor.
    info : instance of mne.io.meas_info.Info
        The measurement info structure.
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.

    Notes
    -----
    Tags are written in a particular order for compatibility with maxfilter.
    """

    # Measurement info
    start_block(fid, FIFF.FIFFB_MEAS_INFO)

    for event in info['events']:
        start_block(fid, FIFF.FIFFB_EVENTS)
        if event.get('channels') is not None:
            write_int(fid, FIFF.FIFF_EVENT_CHANNELS, event['channels'])
        if event.get('list') is not None:
            write_int(fid, FIFF.FIFF_EVENT_LIST, event['list'])
        end_block(fid, FIFF.FIFFB_EVENTS)

    #   HPI Result
    for hpi_result in info['hpi_results']:
        start_block(fid, FIFF.FIFFB_HPI_RESULT)
        for d in hpi_result['dig_points']:
            write_dig_point(fid, d)
        if 'order' in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_DIGITIZATION_ORDER,
                      hpi_result['order'])
        if 'used' in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_COILS_USED, hpi_result['used'])
        if 'moments' in hpi_result:
            write_float_matrix(fid, FIFF.FIFF_HPI_COIL_MOMENTS,
                               hpi_result['moments'])
        if 'goodness' in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_GOODNESS,
                        hpi_result['goodness'])
        if 'good_limit' in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_GOOD_LIMIT,
                        hpi_result['good_limit'])
        if 'dist_limit' in hpi_result:
            write_float(fid, FIFF.FIFF_HPI_FIT_DIST_LIMIT,
                        hpi_result['dist_limit'])
        if 'accept' in hpi_result:
            write_int(fid, FIFF.FIFF_HPI_FIT_ACCEPT, hpi_result['accept'])
        if 'coord_trans' in hpi_result:
            write_coord_trans(fid, hpi_result['coord_trans'])
        end_block(fid, FIFF.FIFFB_HPI_RESULT)

    #   HPI Measurement
    for hpi_meas in info['hpi_meas']:
        start_block(fid, FIFF.FIFFB_HPI_MEAS)
        if hpi_meas.get('creator') is not None:
            write_string(fid, FIFF.FIFF_CREATOR, hpi_meas['creator'])
        if hpi_meas.get('sfreq') is not None:
            write_float(fid, FIFF.FIFF_SFREQ, hpi_meas['sfreq'])
        if hpi_meas.get('nchan') is not None:
            write_int(fid, FIFF.FIFF_NCHAN, hpi_meas['nchan'])
        if hpi_meas.get('nave') is not None:
            write_int(fid, FIFF.FIFF_NAVE, hpi_meas['nave'])
        if hpi_meas.get('ncoil') is not None:
            write_int(fid, FIFF.FIFF_HPI_NCOIL, hpi_meas['ncoil'])
        if hpi_meas.get('first_samp') is not None:
            write_int(fid, FIFF.FIFF_FIRST_SAMPLE, hpi_meas['first_samp'])
        if hpi_meas.get('last_samp') is not None:
            write_int(fid, FIFF.FIFF_LAST_SAMPLE, hpi_meas['last_samp'])
        for hpi_coil in hpi_meas['hpi_coils']:
            start_block(fid, FIFF.FIFFB_HPI_COIL)
            if hpi_coil.get('number') is not None:
                write_int(fid, FIFF.FIFF_HPI_COIL_NO, hpi_coil['number'])
            if hpi_coil.get('epoch') is not None:
                write_float_matrix(fid, FIFF.FIFF_EPOCH, hpi_coil['epoch'])
            if hpi_coil.get('slopes') is not None:
                write_float(fid, FIFF.FIFF_HPI_SLOPES, hpi_coil['slopes'])
            if hpi_coil.get('corr_coeff') is not None:
                write_float(fid, FIFF.FIFF_HPI_CORR_COEFF,
                            hpi_coil['corr_coeff'])
            if hpi_coil.get('custom_ref') is not None:
                write_float(fid, FIFF.FIFF_CUSTOM_REF,
                            hpi_coil['custom_ref'])
            end_block(fid, FIFF.FIFFB_HPI_COIL)
        end_block(fid, FIFF.FIFFB_HPI_MEAS)

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
    _write_proj(fid, info['projs'])

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
    if info.get('line_freq') is not None:
        write_float(fid, FIFF.FIFF_LINE_FREQ, info['line_freq'])
    if data_type is not None:
        write_int(fid, FIFF.FIFF_DATA_PACK, data_type)
    if info.get('custom_ref_applied'):
        write_int(fid, FIFF.FIFF_CUSTOM_REF, info['custom_ref_applied'])

    #  Channel information
    for k, c in enumerate(info['chs']):
        #   Scan numbers may have been messed up
        c = deepcopy(c)
        c['scanno'] = k + 1
        # for float/double, the "range" param is unnecessary
        if reset_range is True:
            c['range'] = 1.0
        write_ch_info(fid, c)

    # Subject information
    if info.get('subject_info') is not None:
        start_block(fid, FIFF.FIFFB_SUBJECT)
        si = info['subject_info']
        if si.get('id') is not None:
            write_int(fid, FIFF.FIFF_SUBJ_ID, si['id'])
        if si.get('his_id') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_HIS_ID, si['his_id'])
        if si.get('last_name') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_LAST_NAME, si['last_name'])
        if si.get('first_name') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_FIRST_NAME, si['first_name'])
        if si.get('middle_name') is not None:
            write_string(fid, FIFF.FIFF_SUBJ_MIDDLE_NAME, si['middle_name'])
        if si.get('birthday') is not None:
            write_julian(fid, FIFF.FIFF_SUBJ_BIRTH_DAY, si['birthday'])
        if si.get('sex') is not None:
            write_int(fid, FIFF.FIFF_SUBJ_SEX, si['sex'])
        if si.get('hand') is not None:
            write_int(fid, FIFF.FIFF_SUBJ_HAND, si['hand'])
        end_block(fid, FIFF.FIFFB_SUBJECT)

    if info.get('hpi_subsystem') is not None:
        hs = info['hpi_subsystem']
        start_block(fid, FIFF.FIFFB_HPI_SUBSYSTEM)
        if hs.get('ncoil') is not None:
            write_int(fid, FIFF.FIFF_HPI_NCOIL, hs['ncoil'])
        if hs.get('event_channel') is not None:
            write_string(fid, FIFF.FIFF_EVENT_CHANNEL, hs['event_channel'])
        if hs.get('hpi_coils') is not None:
            for coil in hs['hpi_coils']:
                start_block(fid, FIFF.FIFFB_HPI_COIL)
                if coil.get('event_bits') is not None:
                    write_int(fid, FIFF.FIFF_EVENT_BITS,
                              coil['event_bits'])
                end_block(fid, FIFF.FIFFB_HPI_COIL)
        end_block(fid, FIFF.FIFFB_HPI_SUBSYSTEM)

    end_block(fid, FIFF.FIFFB_MEAS_INFO)

    #   Processing history
    _write_proc_history(fid, info)


def write_info(fname, info, data_type=None, reset_range=True):
    """Write measurement info in fif file.

    Parameters
    ----------
    fname : str
        The name of the file. Should end by -info.fif.
    info : instance of mne.io.meas_info.Info
        The measurement info structure
    data_type : int
        The data_type in case it is necessary. Should be 4 (FIFFT_FLOAT),
        5 (FIFFT_DOUBLE), or 16 (FIFFT_DAU_PACK16) for
        raw data.
    reset_range : bool
        If True, info['chs'][k]['range'] will be set to unity.
    """
    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MEAS)
    write_meas_info(fid, info, data_type, reset_range)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)


def _is_equal_dict(dicts):
    """Aux function"""
    tests = zip(*[d.items() for d in dicts])
    is_equal = []
    for d in tests:
        k0, v0 = d[0]
        is_equal.append(all(np.all(k == k0) and
                        np.all(v == v0) for k, v in d))
    return all(is_equal)


@verbose
def _merge_dict_values(dicts, key, verbose=None):
    """Merge things together

    Fork for {'dict', 'list', 'array', 'other'}
    and consider cases where one or all are of the same type.
    """
    values = [d[key] for d in dicts]
    msg = ("Don't know how to merge '%s'. Make sure values are "
           "compatible." % key)

    def _flatten(lists):
        return [item for sublist in lists for item in sublist]

    def _check_isinstance(values, kind, func):
        return func([isinstance(v, kind) for v in values])

    def _where_isinstance(values, kind):
        """Aux function"""
        return np.where([isinstance(v, type) for v in values])[0]

    # list
    if _check_isinstance(values, list, all):
        lists = (d[key] for d in dicts)
        return (_uniquify_projs(_flatten(lists)) if key == 'projs'
                else _flatten(lists))
    elif _check_isinstance(values, list, any):
        idx = _where_isinstance(values, list)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            lists = (d[key] for d in dicts if isinstance(d[key], list))
            return _flatten(lists)
    # dict
    elif _check_isinstance(values, dict, all):
        is_qual = _is_equal_dict(values)
        if is_qual:
            return values[0]
        else:
            RuntimeError(msg)
    elif _check_isinstance(values, dict, any):
        idx = _where_isinstance(values, dict)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            raise RuntimeError(msg)
    # ndarray
    elif _check_isinstance(values, np.ndarray, all):
        is_qual = all(np.all(values[0] == x) for x in values[1:])
        if is_qual:
            return values[0]
        elif key == 'meas_date':
            logger.info('Found multiple entries for %s. '
                        'Setting value to `None`' % key)
            return None
        else:
            raise RuntimeError(msg)
    elif _check_isinstance(values, np.ndarray, any):
        idx = _where_isinstance(values, np.ndarray)
        if len(idx) == 1:
            return values[int(idx)]
        elif len(idx) > 1:
            raise RuntimeError(msg)
    # other
    else:
        unique_values = set(values)
        if len(unique_values) == 1:
            return list(values)[0]
        elif isinstance(list(unique_values)[0], BytesIO):
            logger.info('Found multiple StringIO instances. '
                        'Setting value to `None`')
            return None
        elif isinstance(list(unique_values)[0], string_types):
            logger.info('Found multiple filenames. '
                        'Setting value to `None`')
            return None
        else:
            raise RuntimeError(msg)


@verbose
def _merge_info(infos, verbose=None):
    """Merge two measurement info dictionaries"""

    info = Info()
    ch_names = _merge_dict_values(infos, 'ch_names')
    duplicates = set([ch for ch in ch_names if ch_names.count(ch) > 1])
    if len(duplicates) > 0:
        msg = ("The following channels are present in more than one input "
               "measurement info objects: %s" % list(duplicates))
        raise ValueError(msg)
    info['nchan'] = len(ch_names)
    info['ch_names'] = ch_names
    info['chs'] = []
    for this_info in infos:
        info['chs'].extend(this_info['chs'])

    transforms = ['ctf_head_t', 'dev_head_t', 'dev_ctf_t']
    for trans_name in transforms:
        trans = [i[trans_name] for i in infos if i[trans_name]]
        if len(trans) == 0:
            info[trans_name] = None
        elif len(trans) == 1:
            info[trans_name] = trans[0]
        elif all(np.all(trans[0]['trans'] == x['trans']) and
                 trans[0]['from'] == x['from'] and
                 trans[0]['to'] == x['to']
                 for x in trans[1:]):
            info[trans_name] = trans[0]
        else:
            msg = ("Measurement infos provide mutually inconsistent %s" %
                   trans_name)
            raise ValueError(msg)
    other_fields = ['acq_pars', 'acq_stim', 'bads', 'buffer_size_sec',
                    'comps', 'custom_ref_applied', 'description', 'dig',
                    'experimenter', 'file_id', 'filename', 'highpass',
                    'hpi_results', 'hpi_meas', 'hpi_subsystem', 'events',
                    'line_freq', 'lowpass', 'meas_date', 'meas_id',
                    'proj_id', 'proj_name', 'projs', 'sfreq',
                    'subject_info', 'sfreq']

    for k in other_fields:
        info[k] = _merge_dict_values(infos, k)

    return info


def create_info(ch_names, sfreq, ch_types=None, montage=None):
    """Create a basic Info instance suitable for use with create_raw

    Parameters
    ----------
    ch_names : list of str | int
        Channel names. If an int, a list of channel names will be created
        from range(ch_names)
    sfreq : float
        Sample rate of the data.
    ch_types : list of str | str
        Channel types. If None, data are assumed to be misc.
        Currently supported fields are "mag", "grad", "eeg", and "misc".
        If str, then all channels are assumed to be of the same type.
    montage : None | str | Montage | DigMontage | list
        A montage containing channel positions. If str or Montage is
        specified, the channel info will be updated with the channel
        positions. Default is None. If DigMontage is specified, the
        digitizer information will be updated. A list of unique montages,
        can be specifed and applied to the info.

    Notes
    -----
    The info dictionary will be sparsely populated to enable functionality
    within the rest of the package. Advanced functionality such as source
    localization can only be obtained through substantial, proper
    modifications of the info structure (not recommended).

    Note that the MEG device-to-head transform ``info['dev_head_t']`` will
    be initialized to the identity transform.
    """
    if isinstance(ch_names, int):
        ch_names = list(np.arange(ch_names).astype(str))
    if not isinstance(ch_names, (list, tuple)):
        raise TypeError('ch_names must be a list, tuple, or int')
    sfreq = float(sfreq)
    if sfreq <= 0:
        raise ValueError('sfreq must be positive')
    nchan = len(ch_names)
    if ch_types is None:
        ch_types = ['misc'] * nchan
    if isinstance(ch_types, string_types):
        ch_types = [ch_types] * nchan
    if len(ch_types) != nchan:
        raise ValueError('ch_types and ch_names must be the same length')
    info = _empty_info()
    info['meas_date'] = np.array([0, 0], np.int32)
    info['sfreq'] = sfreq
    info['ch_names'] = ch_names
    info['nchan'] = nchan
    loc = np.concatenate((np.zeros(3), np.eye(3).ravel())).astype(np.float32)
    for ci, (name, kind) in enumerate(zip(ch_names, ch_types)):
        if not isinstance(name, string_types):
            raise TypeError('each entry in ch_names must be a string')
        if not isinstance(kind, string_types):
            raise TypeError('each entry in ch_types must be a string')
        if kind not in _kind_dict:
            raise KeyError('kind must be one of %s, not %s'
                           % (list(_kind_dict.keys()), kind))
        kind = _kind_dict[kind]
        chan_info = dict(loc=loc, eeg_loc=None, unit_mul=0, range=1., cal=1.,
                         coil_trans=None, kind=kind[0], coil_type=kind[1],
                         unit=kind[2], coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                         ch_name=name, scanno=ci + 1, logno=ci + 1)
        info['chs'].append(chan_info)
    if montage is not None:
        from ..channels.montage import (Montage, DigMontage, _set_montage,
                                        read_montage)
        if not isinstance(montage, list):
            montage = [montage]
        for montage_ in montage:
            if isinstance(montage_, (Montage, DigMontage)):
                _set_montage(info, montage_)
            elif isinstance(montage_, string_types):
                montage_ = read_montage(montage_)
                _set_montage(info, montage_)
            else:
                raise TypeError('Montage must be an instance of Montage, '
                                'DigMontage, a list of montages, or filepath, '
                                'not %s.' % type(montage))
    return info


RAW_INFO_FIELDS = (
    'acq_pars', 'acq_stim', 'bads', 'ch_names', 'chs', 'comps',
    'ctf_head_t', 'custom_ref_applied', 'description', 'dev_ctf_t',
    'dev_head_t', 'dig', 'experimenter', 'events',
    'file_id', 'filename', 'highpass', 'hpi_meas', 'hpi_results',
    'hpi_subsystem', 'line_freq', 'lowpass', 'meas_date', 'meas_id', 'nchan',
    'proj_id', 'proj_name', 'projs', 'sfreq', 'subject_info',
)


def _empty_info():
    """Create an empty info dictionary"""
    _none_keys = (
        'acq_pars', 'acq_stim', 'ctf_head_t', 'description',
        'dev_ctf_t', 'dig', 'experimenter',
        'file_id', 'filename', 'highpass', 'hpi_subsystem', 'line_freq',
        'lowpass', 'meas_date', 'meas_id', 'proj_id', 'proj_name',
        'subject_info',
    )
    _list_keys = (
        'bads', 'ch_names', 'chs', 'comps', 'events', 'hpi_meas',
        'hpi_results', 'projs',
    )
    info = Info()
    for k in _none_keys:
        info[k] = None
    for k in _list_keys:
        info[k] = list()
    info['custom_ref_applied'] = False
    info['nchan'] = info['sfreq'] = 0
    info['dev_head_t'] = {'from': FIFF.FIFFV_COORD_DEVICE,
                          'to': FIFF.FIFFV_COORD_HEAD, 'trans': np.eye(4)}
    assert set(info.keys()) == set(RAW_INFO_FIELDS)
    return info
