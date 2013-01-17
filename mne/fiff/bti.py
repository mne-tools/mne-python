# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Yuval Harpaz <yuvharpaz@gmail.com>
#
#          simplified bsd-3 license


from . import Raw, pick_types
from . constants import BTI
from . import FIFF

import os
import os.path as op
import struct
import time

from datetime import datetime
from itertools import count

import numpy as np

from ..utils import verbose

import logging
logger = logging.getLogger('mne')


FIFF_INFO_CHS_FIELDS = ('loc', 'ch_name', 'unit_mul', 'coil_trans',
    'coord_frame', 'coil_type', 'range', 'unit', 'cal', 'eeg_loc',
    'scanno', 'kind', 'logno')

FIFF_INFO_CHS_DEFAULTS = (np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                          dtype=np.float32), None, 0, None, 0, 0, 1.0,
                          107, 1.0, None, None, 402, None)

FIFF_INFO_DIG_FIELDS = ("kind", "ident", "r", "coord_frame")
FIFF_INFO_DIG_DEFAULTS = (None, None, None, FIFF.FIFFV_COORD_HEAD)

BTI_WH2500_REF_MAG = ['MxA', 'MyA', 'MzA', 'MxaA', 'MyaA', 'MzaA']
BTI_WH2500_REF_GRAD = ['GxxA', 'GyyA', 'GyxA', 'GzaA', 'GzyA']

dtypes = zip(range(1, 5), ('i2', 'i4', 'f4', 'f8'))
DTYPES = dict((i, np.dtype(t)) for i, t in dtypes)

###############################################################################
# Reading


def _unpack_matrix(fid, format, rows, cols, dtype):
    """ Aux Function """
    out = np.zeros([rows, cols], dtype=dtype)
    bsize = struct.calcsize(format)
    string = os.read(fid, bsize) if isinstance(fid, int) else fid.read(bsize)
    data = struct.unpack(format, string)
    iter_mat = [(r, c) for r in xrange(rows) for c in xrange(cols)]
    for idx, (row, col) in enumerate(iter_mat):
        out[row, col] = data[idx]

    return out


def _unpack_simple(fid, format, count):
    """ Aux Function """
    bsize = struct.calcsize(format)
    string = os.read(fid, bsize) if isinstance(fid, int) else fid.read(bsize)
    data = list(struct.unpack(format, string))

    out = data if count < 2 else list(data)
    if len(out) > 0:
        out = out[0]

    return out


def bti_read_str(fid, count=1):
    """ Read string """
    format = '>' + ('c' * count)
    data = list(struct.unpack(format, fid.read(struct.calcsize(format))))

    return ''.join(data[0:data.index('\x00') if '\x00' in data else count])


def bti_read_char(fid, count=1):
    " Read character from bti file """

    return _unpack_simple(fid, '>' + ('c' * count), count)


def bti_read_bool(fid, count=1):
    """ Read bool value from bti file """

    return _unpack_simple(fid, '>' + ('?' * count), count)


def bti_read_uint8(fid, count=1):
    """ Read unsigned 8bit integer from bti file """

    return _unpack_simple(fid, '>' + ('B' * count), count)


def bti_read_int8(fid, count=1):
    """ Read 8bit integer from bti file """

    return _unpack_simple(fid, '>' + ('b' * count),  count)


def bti_read_uint16(fid, count=1):
    """ Read unsigned 16bit integer from bti file """

    return _unpack_simple(fid, '>' + ('H' * count), count)


def bti_read_int16(fid, count=1):
    """ Read 16bit integer from bti file """

    return _unpack_simple(fid, '>' + ('H' * count),  count)


def bti_read_uint32(fid, count=1):
    """ Read unsigned 32bit integer from bti file """

    return _unpack_simple(fid, '>' + ('I' * count), count)


def bti_read_int32(fid, count=1):
    """ Read 32bit integer from bti file """

    return _unpack_simple(fid, '>' + ('i' * count), count)


def bti_read_uint64(fid, count=1):
    """ Read unsigned 64bit integer from bti file """

    return _unpack_simple(fid, '>' + ('Q' * count), count)


def bti_read_int64(fid, count=1):
    """ Read 64bit integer from bti file """

    return _unpack_simple(fid, '>' + ('q' * count), count)


def bti_read_float(fid, count=1):
    """ Read 32bit float from bti file """

    return _unpack_simple(fid, '>' + ('f' * count), count)


def bti_read_double(fid, count=1):
    """ Read 64bit float from bti file """

    return _unpack_simple(fid, '>' + ('d' * count), count)


def bti_read_int16_matrix(fid, rows, cols):
    """ Read 16bit integer matrix from bti file """

    format = '>' + ('h' * rows * cols)

    return _unpack_matrix(fid, format, rows, cols, np.int16)


def bti_read_float_matrix(fid, rows, cols):
    """ Read 32bit float matrix from bti file """

    format = '>' + ('f' * rows * cols)
    return _unpack_matrix(fid, format, rows, cols, 'f4')


def bti_read_double_matrix(fid, rows, cols):
    """ Read 64bit float matrix from bti file """

    format = '>' + ('d' * rows * cols)

    return _unpack_matrix(fid, format, rows, cols, 'f8')


def bti_read_transform(fid):
    """ Read 64bit float matrix transform from bti file """

    format = '>' + ('d' * 4 * 4)

    return _unpack_matrix(fid, format, 4, 4, 'f8')


###############################################################################
# Transforms

def _get_m_to_nm(adjust=None, translation=(0.0, 0.02, 0.11)):
    """ Get the general Magnes3600WH to Neuromag coordinate transform

    Parameters
    ----------
    adjust : int | None
        Degrees to tilt x-axis for sensor frame misalignment.
        If None, no adjustment will be applied.
    translation : array-like
        The translation to place the origin of coordinate system
        to the center of the head.

    Returns
    -------
    m_nm_t : ndarray
        4 x 4 rotation, translation, scaling matrix.

    """
    flip_t = np.array(BTI.T_ROT_VV, np.float64)
    adjust_t = np.array(BTI.T_IDENT, np.float64)
    adjust = 0 if adjust is None else adjust
    deg = np.deg2rad(float(adjust))
    adjust_t[[1, 2], [1, 2]] = np.cos(deg)
    adjust_t[[1, 2], [2, 1]] = -np.sin(deg), np.sin(deg)
    m_nm_t = np.ones([4, 4])
    m_nm_t[BTI.T_ROT_IX] = np.dot(flip_t[BTI.T_ROT_IX],
                                  adjust_t[BTI.T_ROT_IX])
    m_nm_t[BTI.T_TRANS_IX] = np.matrix(translation).T

    return m_nm_t


def _read_system_trans(fname):
    """ Read global 4-D system transform
    """
    trans = [l.strip().split(', ') for l in open(fname) if l.startswith(' ')]
    return np.array(trans, dtype=float).reshape([4, 4])


def _m_to_nm_coil_trans(ch_t, bti_t, nm_t, nm_default_scale=True):
    """ transforms 4D coil position to fiff / Neuromag
    """
    ch_t = np.array(ch_t.split(', '), dtype=float).reshape([4, 4])

    nm_coil_trans = _apply_t(_inverse_t(ch_t, bti_t), nm_t)

    if nm_default_scale:
        nm_coil_trans[3, :3] = 0.

    return nm_coil_trans


def _inverse_t(x, t, rot=BTI.T_ROT_IX, trans=BTI.T_TRANS_IX,
               scal=BTI.T_SCA_IX):
    """ Apply inverse transform
    """
    x = x.copy()
    x[scal] *= t[scal]
    x[rot] = np.dot(t[rot].T, x[rot])
    x[trans] -= t[trans]
    x[trans] = np.dot(t[rot].T, x[trans])

    return x


def _apply_t(x, t, rot=BTI.T_ROT_IX, trans=BTI.T_TRANS_IX, scal=BTI.T_SCA_IX):
    """ Apply transform
    """
    x = x.copy()
    x[rot] = np.dot(t[rot], x[rot])
    x[trans] = np.dot(t[rot], x[trans])
    x[trans] += t[trans]
    x[scal] *= t[scal]

    return x


def _merge_t(t1, t2):
    """ Merge two transforms """
    t = np.array(BTI.T_IDENT, dtype=np.float32)
    t[BTI.T_ROT_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_ROT_IX])
    t[BTI.T_TRANS_IX] = np.dot(t1[BTI.T_ROT_IX], t2[BTI.T_TRANS_IX])
    t[BTI.T_TRANS_IX] += t1[BTI.T_TRANS_IX]

    return t


def _rename_channels(names):
    """Renames appropriately ordered list of channel names

    Parameters
    ----------
    names : list of str
        Lists of 4-D channel names in ascending order

    Returns
    -------
    generator object :
        Generator object for iteration over MEG channels with Neuromag names.
    """
    ref_mag, ref_grad, eog, ext = ((lambda: count(1))() for i in range(4))
    for i, name in enumerate(names, 1):
        if name.startswith('A'):
            name = 'MEG %3.3d' % i
        elif name == 'RESPONSE':
            name = 'SRI 013'
        elif name == 'TRIGGER':
            name = 'STI 014'
        elif name.startswith('EOG'):
            name = 'EOG %3.3d' % eog.next()
        elif name == 'ECG':
            name = 'ECG 001'
        elif name == 'UACurrent':
            name = 'UTL 001'
        elif name.startswith('M'):
            name = 'RFM %3.3d' % ref_mag.next()
        elif name.startswith('G'):
            name = 'RFG %3.3d' % ref_grad.next()
        elif name.startswith('X'):
            name = 'EXT %3.3d' % ext.next()

        yield name


def convert_head_shape(idx_points, dig_points, use_hpi=False):
    """ Read digitation points from MagnesWH3600 and transform to Neuromag
        coordinates.

    Parameters
    ----------
    idx_points : ndarray
        The index points.
    dix_points : ndarray
        The digitization points.
    use_hpi : bool
        Whether to treat hpi coils as digitization points or not. If False,
        Hpi coils will be discarded.

    Returns
    -------
    dig : list
        A list of dictionaries including the dig points and additional info
    t : ndarray
        The 4 x 4 matrix describing the Magnes3600WH head to Neuromag head
        transformation.
    """

    fp = idx_points  # fiducial points
    dp = np.sum(fp[2] * (fp[0] - fp[1]))
    tmp1, tmp2 = np.sum(fp[2] ** 2), np.sum((fp[0] - fp[1]) ** 2)
    dcos = -dp / np.sqrt(tmp1 * tmp2)
    dsin = np.sqrt(1. - dcos * dcos)
    dt = dp / np.sqrt(tmp2)

    fiducials_nm = np.ones([len(fp), 3])

    for idx, f in enumerate(fp):
        fiducials_nm[idx, 0] = dcos * f[0] - dsin * f[1] + dt
        fiducials_nm[idx, 1] = dsin * f[0] + dcos * f[1]
        fiducials_nm[idx, 2] = f[2]

    # adjust order of fiducials to Neuromag
    fiducials_nm[[1, 2]] = fiducials_nm[[2, 1]]

    t = np.array(BTI.T_IDENT, dtype=np.float32)
    t[0, 0] = dcos
    t[0, 1] = -dsin
    t[1, 0] = dsin
    t[1, 1] = dcos
    t[0, 3] = dt

    dpnts = dig_points
    dig_points_nm = np.dot(t[BTI.T_ROT_IX], dpnts).T
    dig_points_nm += t[BTI.T_TRANS_IX].T

    all_points = np.r_[fiducials_nm, dig_points_nm]
    fiducials_idents = range(1, 4) + range(1, (len(fp) + 1) - 3)
    dig = []
    for idx in xrange(all_points.shape[0]):
        point_info = dict((k, v) for k, v in zip(FIFF_INFO_DIG_FIELDS,
                          FIFF_INFO_DIG_DEFAULTS))
        point_info['r'] = all_points[idx]
        if idx < 3:
            point_info['kind'] = FIFF.FIFFV_POINT_CARDINAL
            point_info['ident'] = fiducials_idents[idx]
        if 2 < idx < len(idx_points) and use_hpi:
            point_info['kind'] = FIFF.FIFFV_POINT_HPI
            point_info['ident'] = fiducials_idents[idx]
        elif idx > 4:
            point_info['kind'] = FIFF.FIFFV_POINT_EXTRA
            point_info['ident'] = (idx + 1) - len(fiducials_idents)

        if 2 < idx < len(idx_points) and not use_hpi:
            pass
        else:
            dig.append(point_info)

    return dig, t


###############################################################################
# Read files

def read_head_shape(fname):
    """Read index points and dig points from BTi head shape file

    Parameters
    ----------
    fname : str
        The absolute path to the headshape file

    Returns
    -------
    idx_points : ndarray
        The index or fiducial points.
    dig_points : ndarray
        The digitazation points.
    """
    with open(fname, 'rb') as fid:
        fid.seek(BTI.FILE_HS_N_DIGPOINTS)
        _n_dig_points = bti_read_int32(fid)
        idx_points = bti_read_double_matrix(fid, BTI.DATA_N_IDX_POINTS, 3)
        dig_points = bti_read_double_matrix(fid, _n_dig_points, 3)

    return idx_points, dig_points


def _correct_offset(fid):
    """Compensate offset padding"""
    current = fid.tell()
    fid.seek(0, 1)
    if ((current % BTI.FILE_CURPOS) != 0):
        offset = current % BTI.FILE_CURPOS
        fid.seek(BTI.FILE_CURPOS - (offset), 1)


def _read_bti_header(fid):
    """ Read bti PDF header
    """
    fid.seek(BTI.FILE_END, 2)
    ftr_pos = fid.tell()
    hdr_pos = bti_read_int64(fid)
    test_val = hdr_pos & BTI.FILE_MASK

    if ((ftr_pos + BTI.FILE_CURPOS - test_val) <= BTI.FILE_MASK):
        hdr_pos = test_val

    if ((hdr_pos % BTI.FILE_CURPOS) != 0):
        hdr_pos += (BTI.FILE_CURPOS - (hdr_pos % BTI.FILE_CURPOS))

    fid.seek(hdr_pos, 0)

    hdr_size = ftr_pos - hdr_pos

    out = dict(version=bti_read_int16(fid),
               file_type=bti_read_str(fid, BTI.FILE_PDF_H_FTYPE))

    fid.seek(BTI.FILE_PDF_H_ENTER, 1)

    out.update(dict(data_format=bti_read_int16(fid),
                acq_mode=bti_read_int16(fid),
                total_epochs=bti_read_int32(fid),
                input_epochs=bti_read_int32(fid),
                total_events=bti_read_int32(fid),
                total_fixed_events=bti_read_int32(fid),
                sample_period=bti_read_float(fid),
                xaxis_label=bti_read_str(fid, BTI.FILE_PDF_H_XLABEL),
                total_processes=bti_read_int32(fid),
                total_chans=bti_read_int16(fid)))

    fid.seek(BTI.FILE_PDF_H_NEXT, 1)
    out.update(dict(checksum=bti_read_int32(fid),
                total_ed_classes=bti_read_int32(fid),
                total_associated_files=bti_read_int16(fid),
                last_file_index=bti_read_int16(fid),
                timestamp=bti_read_int32(fid)))

    fid.seek(BTI.FILE_PDF_H_EXIT, 1)

    _correct_offset(fid)

    return out, hdr_size, ftr_pos


def _read_bti_epoch(fid):
    """Read BTi epoch"""
    out = dict(pts_in_epoch=bti_read_int32(fid),
                epoch_duration=bti_read_float(fid),
                expected_iti=bti_read_float(fid),
                actual_iti=bti_read_float(fid),
                total_var_events=bti_read_int32(fid),
                checksum=bti_read_int32(fid),
                epoch_timestamp=bti_read_int32(fid))

    fid.seek(BTI.FILE_PDF_EPOCH_EXIT, 1)

    return out


def _read_channel(fid):
    """Read BTi PDF channel"""
    out = dict(chan_label=bti_read_str(fid, BTI.FILE_PDF_CH_LABELSIZE),
                chan_no=bti_read_int16(fid),
                attributes=bti_read_int16(fid),
                scale=bti_read_float(fid),
                yaxis_label=bti_read_str(fid, BTI.FILE_PDF_CH_YLABEL),
                valid_min_max=bti_read_int16(fid))

    fid.seek(BTI.FILE_PDF_CH_NEXT, 1)

    out.update(dict(ymin=bti_read_double(fid),
                ymax=bti_read_double(fid),
                index=bti_read_int32(fid),
                checksum=bti_read_int32(fid),
                off_flag=bti_read_str(fid, BTI.FILE_PDF_CH_OFF_FLAG),
                offset=bti_read_float(fid)))

    fid.seek(BTI.FILE_PDF_CH_EXIT, 1)

    return out


def _read_event(fid):
    """Read BTi PDF event"""
    out = dict(event_name=bti_read_str(fid, BTI.FILE_PDF_EVENT_NAME),
                start_lat=bti_read_float(fid),
                end_lat=bti_read_float(fid),
                step_size=bti_read_float(fid),
                fixed_event=bti_read_int16(fid),
                checksum=bti_read_int32(fid))

    fid.seek(BTI.FILE_PDF_EVENT_EXIT, 1)
    _correct_offset(fid)

    return out


def _read_process(fid):
    """Read BTi PDF process"""

    out = dict(nbytes=bti_read_int32(fid),
                blocktype=bti_read_str(fid, BTI.FILE_PDF_PROCESS_BLOCKTYPE),
                checksum=bti_read_int32(fid),
                user=bti_read_str(fid, BTI.FILE_PDF_PROCESS_USER),
                timestamp=bti_read_int32(fid),
                filename=bti_read_str(fid, BTI.FILE_PDF_PROCESS_FNAME),
                total_steps=bti_read_int32(fid))

    fid.seek(BTI.FILE_PDF_PROCESS_EXIT, 1)

    _correct_offset(fid)

    return out


def _read_assoc_file(fid):
    """Read BTi PDF assocfile"""

    out = dict(file_id=bti_read_int16(fid),
                length=bti_read_int16(fid))

    fid.seek(BTI.FILE_PDF_ASSOC_NEXT, 1)
    out['checksum'] = bti_read_int32(fid)

    return out


def _read_pfid_ed(fid):
    """Read PDF ed file"""

    out = dict(comment_size=bti_read_int32(fid),
             name=bti_read_str(fid, BTI.FILE_PDFED_NAME))

    fid.seek(BTI.FILE_PDFED_NEXT, 1)
    out.update(dict(pdf_number=bti_read_int16(fid),
                    total_events=bti_read_int32(fid),
                    timestamp=bti_read_int32(fid),
                    flags=bti_read_int32(fid),
                    de_process=bti_read_int32(fid),
                    checksum=bti_read_int32(fid),
                    ed_id=bti_read_int32(fid),
                    win_width=bti_read_float(fid),
                    win_offset=bti_read_float(fid)))

    fid.seek(BTI.FILE_PDFED_NEXT, 1)

    return out


def _read_userblock(fid, blocks, arch):
    """ Read user block from config """

    _correct_offset(fid)

    cfg = dict()
    cfg['hdr'] = dict(
         nbytes=bti_read_int32(fid),
         kind=bti_read_str(fid, BTI.FILE_CONF_UBLOCK_TYPE),
         checksum=bti_read_int32(fid),
         username=bti_read_str(fid, BTI.FILE_CONF_UBLOCK_UNAME),
         timestamp=bti_read_int32(fid),
         user_space_size=bti_read_int32(fid),
         reserved=bti_read_char(fid, BTI.FILE_CONF_UBLOCK_RESERVED))

    cfg['data'] = dict()
    kind, dta = cfg['hdr']['kind'], cfg['data']
    if kind in [v for k, v in BTI.items() if k[:6] == 'UBLOCK']:
        if kind == BTI.UB_B_MAG_INFO:
            dta['version'] = bti_read_int32(fid)
            fid.seek(BTI.FILE_CONF_UBLOCK_PADDING, 1)
            dta['headers'] = list()
            for hdr in xrange(BTI.FILE_CONF_UBLOCK_BHRANGE):
                d = dict(name=bti_read_str(fid, BTI.FILE_CONF_UBLOCK_BHNAME),
                         transform=bti_read_transform(fid),
                         units_per_bit=bti_read_float(fid))
                dta['headers'] += [d]

        elif kind == BTI.UB_B_COH_POINTS:
            dta['num_points'] = bti_read_int32(fid)
            dta['status'] = bti_read_int32(fid)
            dta['points'] = []
            for pnt in xrange(BTI.FILE_CONF_UBLOCK_PRANGE):
                d = dict(pos=bti_read_double_matrix(fid, 1, 3),
                         direction=bti_read_double_matrix(fid, 1, 3),
                         error=bti_read_double(fid))
                dta['points'] += [d]

        elif kind == BTI.UB_B_CCP_XFM_BLOCK:
            dta['method'] = bti_read_int32(fid)
            if arch == 'solaris':
                size = BTI.FILE_CONF_UBLOCK_XFM_SOLARIS
            elif arch == 'linux':
                size = BTI.FILE_CONF_UBLOCK_XFM_LINUX
            fid.seek(size, 1)
            dta['transform'] = bti_read_transform(fid)

        elif kind == BTI.UB_B_EEG_LOCS:
            dta['electrodes'] = []
            while True:
                if d['label'] == BTI.FILE_CONF_UBLOCK_ELABEL_END:
                    break
                d = dict(label=bti_read_str(fid, BTI.FILE_CONF_UBLOCK_ELABEL),
                         location=bti_read_double_matrix(fid, 1, 3))
                dta['electrodes'] += [d]

        elif kind in [BTI.UB_B_WHC_CHAN_MAP_VER, BTI.UB_B_WHS_SUBSYS_VER]:
            dta['version'] = bti_read_int16(fid)
            dta['struct_size'] = bti_read_int16(fid)
            dta['entries'] = bti_read_int16(fid)
            fid.seek(BTI.FILE_CONF_UBLOCK_SVERS, 1)

        elif kind == BTI.UB_B_WHC_CHAN_MAP:
            num_channels = None
            for block in blocks:
                if block['hdr']['kind'] == BTI.UB_B_WHC_CHAN_MAP_VER:
                    num_channels = block['hdr']['entries']
                    break

            if num_channels is None:
                raise ValueError('Cannot find block %s to determine number'
                                 'of channels' % BTI.UB_B_WHC_CHAN_MAP_VER)

            dta['channels'] = list()
            for i in xrange(num_channels):
                d = dict(subsys_type=bti_read_int16(fid),
                         subsys_num=bti_read_int16(fid),
                         card_num=bti_read_int16(fid),
                         chan_num=bti_read_int16(fid),
                         recdspnum=bti_read_int16(fid))
                d += [d]

        elif kind == BTI.UB_B_WHS_SUB_SYS:
            num_subsys = None
            for block in blocks:
                if block['hdr']['kind'] == BTI.UB_B_WHS_SUBS_VER:
                    num_subsys = block['entries']
                    break

            if num_subsys is None:
                raise ValueError('Cannot find block %s to determine number o'
                                 'f subsystems' % BTI.UB_B_WHS_SUBS_VER)

            dta['subsys'] = list()
            for sub_key in range(num_subsys):
                d = dict(subsys_type=bti_read_int16(fid),
                        subsys_num=bti_read_int16(fid),
                        cards_per_sys=bti_read_int16(fid),
                        channels_per_card=bti_read_int16(fid),
                        card_version=bti_read_int16(fid))

                fid.seek(BTI.FILE_CONF_UBLOCK_WHC_SUB_PADDING, 1)
                d.update(dict(offsetdacgain=bti_read_float(fid),
                        squid_type=bti_read_int32(fid),
                        timesliceoffset=bti_read_int16(fid),
                        padding=bti_read_int16(fid),
                        volts_per_bit=bti_read_float(fid)))
                dta['subkeys'] += [d]

        elif kind == BTI.UB_B_CH_LABELS:
            dta['version'] = bti_read_int32(fid)
            dta['entries'] = bti_read_int32(fid)
            fid.seek(BTI.FILE_CONF_UBLOCK_CH_PADDING, 1)

            dta['labels'] = list()
            for label in xrange(dta['entries']):
                dta['labels'] += [bti_read_str(fid,
                                   BTI.FILE_CONF_UBLOCK_CH_LABEL)]

        elif kind == BTI.UB_B_CALIBRATION:
            dta['sensor_no'] = bti_read_int16(fid)
            fid.seek(fid, BTI.FILE_CONF_UBLOCK_CH_CAL_PADDING, 1)
            dta['timestamp'] = bti_read_int32(fid)
            dta['logdir'] = bti_read_str(fid, BTI.FILE_CONF_UBLOCK_CH_CAL)

        elif kind == BTI.UB_B_SYS_CONFIG_TIME:
            if arch == 'solaris':
                size = BTI.FILE_CONF_UBLOCK_SYS_CONF_SOLARIS
            elif arch == 'linux':
                size = BTI.FILE_CONF_UBLOCK_SYS_CONF_LINUX
            dta['sysconfig_name'] = bti_read_str(fid, size)
            dta['timestamp'] = bti_read_int32(fid)

        elif kind == BTI.UB_B_DELT_ENAABLED:
            dta['delta_enabled'] = bti_read_int16(fid)

        elif kind in [BTI.UB_B_E_TABLE_USED, BTI.UB_B_E_TABLE]:

            dta['hdr'] = dict(version=bti_read_int32(fid),
                              entry_size=bti_read_int32(fid),
                              num_entries=bti_read_int32(fid),
                              filtername=bti_read_str(fid,
                                            BTI.FILE_CONF_UBLOCK_ETAB_FTNAME),
                              num_E_values=bti_read_int32(fid),
                              reserved=bti_read_str(fid,
                                            BTI.FILE_CONF_UBLOCK_ETAB_RESERVED
                                            ))

            if dta['hdr']['version'] == BTI.FILE_CONF_UBLOCK_ETAB_HDR_VER:
                size = BTI.FILE_CONF_UBLOCK_ETAB_CH_NAME
                dta['ch_names'] = [bti_read_str(fid, size) for ch in
                                      range(dta['hdr']['num_entries'])]
                rows = dta['hdr']['num_entries']
                cols = dta['hdr']['num_E_values']
                dta['etable'] = bti_read_float_matrix(fid, rows)
            else:
                # handle MAGNES2500 naming scheme
                dta['ch_names'] = ['WH2500'] * dta['hdr']['num_E_values']
                dta['hdr']['num_E_values'] = BTI.FILE_CONF_UBLOCK_ETAB_WH2500
                dta['e_ch_names'] = BTI_WH2500_REF_MAG
                rows = dta['hdr']['num_entries']
                cols = dta['hdr']['num_E_values']
                dta['etable'] = bti_read_float_matrix(fid, rows, cols)

                _correct_offset(fid)

        elif any([kind == BTI.UB_B_WEIGHTS_USED,
                  kind[:4] == BTI.UB_B_WEIGHT_TABLE]):
            dta['hdr'] = dict(version=bti_read_int32(fid),
                                 entry_size=bti_read_int32(fid),
                                 num_entries=bti_read_int32(fid),
                                 name=bti_read_str(fid,
                                    BTI.FILE_CONF_UBLOCK_WEIGHT_NAME),
                                 description=bti_read_str(fid,
                                    BTI.FILE_CONF_UBLOCK_WEIGHT_DESCR),
                                 num_anlg=bti_read_int32(fid),
                                 num_dsp=bti_read_int32(fid),
                                 reserved=bti_read_str(fid,
                                    BTI.FILE_CONF_UBLOCK_WEIGHT_RESERVED))

            if dta['hdr']['version'] == BTI.FILE_CONF_UBLOCK_WEIGHT_HDR_VER:
                dta['ch_names'] = [bti_read_str(fid,
                                   BTI.FILE_CONF_UBLOCK_WEIGHT_CH_NAME)
                                   for v in range(dta['hdr']['num_entries'])]
                dta['anlg_ch_names'] = [bti_read_str(fid,
                                        BTI.FILE_CONF_UBLOCK_WEIGHT_CH_NAME)
                                        for v in range(dta['hdr']['num_anlg'])]

                dta['dsp_ch_names'] = [bti_read_str(fid,
                                       BTI.FILE_CONF_UBLOCK_WEIGHT_CH_NAME)
                                       for val in range(dta['hdr']['num_dsp'])]

                rows = dta['hdr']['num_entries']
                cols = dta['hdr']['num_anlg']
                dta['dsp_wts'] = bti_read_float_matrix(fid, rows, cols)
                dta['anlg_wts'] = bti_read_int16_matrix(fid, rows, cols)

            else:  # handle MAGNES2500 naming scheme
                dta['ch_names'] = ['WH2500'] * dta['hdr']['num_entries']
                dta['anlg_ch_names'] = BTI_WH2500_REF_MAG[:3]
                dta['hdr']['num_anlg'] = len(dta['anlg_ch_names'])
                dta['dsp_ch_names'] = BTI_WH2500_REF_GRAD
                dta['hdr.num_dsp'] = len(dta['dsp_ch_names'])
                dta_ = (dta['hdr']['num_entries'], dta['hdr']['num_anlg'])
                dta['anlg_wts'] = np.zeros(dta_, dtype='i2')
                dta['dsp_wts'] = np.zeros((dta['hdr']['num_entries'],
                                          dta['hdr']['num_dsp']), dtype='f4')
                for n in range(dta['hdr']['num_entries']):
                    dta['anlg_wts'][d, :] = bti_read_int16_matrix(fid, 1,
                                                    dta['hdr']['num_anlg'])
                    bti_read_int16(fid)
                    dta['dsp_wts'][d, :] = bti_read_float_matrix(fid, 1,
                                                dta['hdr']['num_dsp'])

                _correct_offset(fid)

        elif kind == BTI.UB_B_TRIG_MASK:
            dta['version'] = bti_read_int32(fid)
            dta['entries'] = bti_read_int32(fid)
            fid.seek(BTI.FILE_CONF_UBLOCK_MASK_PADDING, 1)

            dta['masks'] = []
            for entry in range(dta['entries']):
                d = dict(name=bti_read_str(fid,
                                BTI.FILE_CONF_UBLOCK_MASK_NAME),
                         nbits=bti_read_uint16(fid),
                         shift=bti_read_uint16(fid),
                         mask=bti_read_uint32(fid))
                dta['masks'] += [d]
                fid.seek(BTI.FILE_CONF_UBLOCK_MASK_HDR_OFFSET)

    else:
        dta['unknown'] = dict(hdr=bti_read_char(fid,
                              cfg['hdr']['user_space_size']))

    _correct_offset(fid)

    return cfg


def _read_dev_hdr(fid):
    """ Read device header """
    return dict(size=bti_read_int32(fid),
                checksum=bti_read_int32(fid),
                reserved=bti_read_str(fid, BTI.FILE_CONF_CH_RESERVED))


def _read_coil_def(fid):
    """ Read coil definition """
    coildef = dict(position=bti_read_double_matrix(fid, 1, 3),
                   orientation=bti_read_double_matrix(fid, 1, 3),
                   radius=bti_read_double(fid),
                   wire_radius=bti_read_double(fid),
                   turns=bti_read_int16(fid))
    fid.seek(fid, 2, 1)
    coildef['checksum'] = bti_read_int32(fid)
    coildef['reserved'] = bti_read_str(fid, 32)


def _read_ch_config(fid):
    """Read BTi channel config"""

    cfg = dict(name=bti_read_str(fid, BTI.FILE_CONF_CH_NAME),
            chan_no=bti_read_int16(fid),
            ch_type=bti_read_uint16(fid),
            sensor_no=bti_read_int16(fid))

    fid.seek(fid, BTI.FILE_CONF_CH_NEXT, 1)

    cfg.update(dict(
            gain=bti_read_float(fid),
            units_per_bit=bti_read_float(fid),
            yaxis_label=bti_read_str(fid, BTI.FILE_CONF_CH_YLABEL),
            aar_val=bti_read_double(fid),
            checksum=bti_read_int32(fid),
            reserved=bti_read_str(fid, BTI.FILE_CONF_CH_RESERVED)))

    _correct_offset(fid)

    # Then the channel info
    ch_type, chan = cfg['ch_type'], dict()
    chan['dev'] = _read_dev_hdr(fid)
    if ch_type in [BTI.CHTYPE_MEG, BTI.CHTYPE_REF]:
        chan['loops'] = [_read_coil_def(fid) for d in
                        range(chan['dev']['total_loops'])]

    elif ch_type == BTI.CHTYPE_EEG:
        chan['impedance'] = bti_read_float(fid)
        chan['padding'] = bti_read_str(fid, BTI.FILE_CONF_CH_PADDING)
        chan['transform'] = bti_read_transform(fid)
        chan['reserved'] = bti_read_char(fid, BTI.FILE_CONF_CH_RESERVED)

    elif ch_type in [BTI.CHTYPE_TRIGGER,  BTI.CHTYPE_EXTERNAL,
                     BTI.CHTYPE_UTILITY, BTI.CHTYPE_DERIVED]:
        chan['user_space_size'] = bti_read_int32(fid)
        if ch_type == BTI.CHTYPE_TRIGGER:
            fid.seek(2, 1)
        chan['reserved'] = bti_read_str(fid, BTI.FILE_CONF_CH_RESERVED)

    elif ch_type == BTI.CHTYPE_SHORTED:
        chan['reserved'] = bti_read_str(fid, BTI.FILE_CONF_CH_RESERVED)

    cfg['chan'] = chan

    _correct_offset(fid)

    return cfg


def read_raw_data(info, start=None, stop=None, order='by_name',
                  dtype='f8'):
    """ Read Bti processed data file (PDF)

    Parameters
    ----------
    info : dict
        The measurement info drawn from the BTi processed data file (PDF).
    start : int | None
        The number of the first time slice to read. If None, all data will
        be read from the begninning.
    stop : int | None
        The number of the last time slice to read. If None, all data will
        be read to the end.
    order : index | 'by_name' | None
        The order of the data along the channel axis. If 'by_name',
        data will be returnend sorted by channel names, if None, by
        acquisition order.
    dtype : str | dtype object
        The type the data are casted to.

    Returns
    -------
    data : ndarray
        The measurement data, a channels x timeslices array.
    """
    total_slices = info['total_slices']
    if start is None:
        start = 0
    if stop is None:
        stop = total_slices

    if any([start < 0, stop > total_slices, start >= stop]):
        raise RuntimeError('Invalid data range supplied:'
                           ' %d, %d' % (start, stop))

    info['fid'].seek(info['bytes_per_slice'] * start)

    cnt = (stop - start) * info['total_chans']
    shape = [stop - start, info['total_chans']]
    data = np.fromfile(info['fid'], dtype=info['dtype'],
                       count=cnt).reshape(shape).T.astype(dtype)

    if order == 'by_name':
        mapping = [(i, d['chan_label']) for i, d in
                   enumerate(info['channels'])]
        sort = sorted(mapping, key=lambda c: int(c[1][1:])
                      if c[1][0] == BTI.DATA_MEG_CH_CHAR else c[1])
        order = [idx[0] for idx in sort]

    return data if order is None else data[order]


def read_pdf_info(fname):
    """ Read Bti processed data file (PDF)

    Parameters
    ----------
    fname : str
        The file name of the bti processed data file (PDF)

    Returns
    -------
    info : dict
        The info structure from the BTi PDF header
    """

    with open(fname, 'rb') as fid:

        info, hdr_size, ftr_pos = _read_bti_header(fid)

        info['epochs'] = [_read_bti_epoch(fid) for epoch in
                          xrange(info['total_epochs'])]

        info['channels'] = [_read_channel(fid) for ch in
                            xrange(info['total_chans'])]

        info['events'] = [_read_event(fid) for event in
                          xrange(info['total_events'])]

        info['processes'] = [_read_process(fid) for process in
                             xrange(info['total_processes'])]

        info['assocfiles'] = [_read_assoc_file(fid) for af in
                              xrange(info['total_associated_files'])]

        info['edclasses'] = [_read_pfid_ed(fid) for ed_class in
                             range(info['total_ed_classes'])]

        # We load any remaining data
        fid.seek(0, 1)
        info['extradata'] = fid.read(ftr_pos - fid.tell())

        # copy of fid into info
        info['fid'] = os.fdopen(os.dup(fid.fileno()), 'r')

        # calculate n_tsl
        info['total_slices'] = sum(e['pts_in_epoch'] for e in
                                   info['epochs'])

        info['dtype'] = DTYPES[info['data_format']]

        bps = info['dtype'].itemsize * info['total_chans']
        info['bytes_per_slice'] = bps

        return info


def read_config(fname, arch):
    """Read BTi system config file

    Parameters
    ----------
    fname : str
        The absolute path to the config file
    arch : 'linux' | 'solaris'
        The architecture of the acuisition setup used.

    Returns
    -------
    data : ndarray
        The channels X time slices MEG measurments.

    """
    if arch not in ['linux', 'solaris']:
        raise ValueError('Arch must be \'linux\' or \'solaris\'')

    with open('config', 'rb') as fid:
        cfg = dict()
        cfg['hdr'] = dict(version=bti_read_int16(fid),
                        site_name=bti_read_str(fid, BTI.FILE_CONF_SITENAME),
                        dap_hostname=bti_read_str(fid, BTI.FILE_CONF_HOSTNAME),
                        sys_type=bti_read_int16(fid),
                        sys_options=bti_read_int32(fid),
                        supply_freq=bti_read_int16(fid),
                        total_chans=bti_read_int16(fid),
                        system_fixed_gain=bti_read_float(fid),
                        volts_per_bit=bti_read_float(fid),
                        total_sensors=bti_read_int16(fid),
                        total_user_blocks=bti_read_int16(fid),
                        next_der_chan_no=bti_read_int16(fid))

        fid.seek(BTI.FILE_CONF_NEXT, 1)
        cfg['checksum'] = bti_read_uint32(fid)
        cfg['reserved'] = bti_read_char(fid, BTI.FILE_CONF_RESERVED)
        cfg['transforms'] = [bti_read_transform(fid) for xfm in
                             xrange(cfg['hdr']['total_sensors'])]

        cfg['user_blocks'] = list()
        for block in xrange(cfg['hdr']['total_user_blocks']):
            cfg['user_blocks'] += [_read_userblock(fid, cfg['user_blocks'],
                                                   arch)]

        # Finally, the channel information
        # cfg['channels'] = [_read_ch_config(fid) for ch in
                           # xrange(cfg['total_chans'])]

        return cfg


class RawBTi(Raw):
    """ Raw object from 4-D Neuroimaging MagnesWH3600 data

    Parameters
    ----------
    pdf_fname : str | None
        absolute path to the processed data file (PDF)
    config_fname : str | None
        absolute path to system confnig file. If None, it is assumed to be in
        the same directory.
    head_shape_fname : str
        absolute path to the head shape file. If None, it is assumed to be in
        the same directory.
    rotation_x : float | int | None
        Degrees to tilt x-axis for sensor frame misalignment.
        If None, no adjustment will be applied.
    translation : array-like
        The translation to place the origin of coordinate system
        to the center of the head.
    use_hpi : bool
        Whether to treat hpi coils as digitization points or not. If
        False, HPI coils will be discarded.
    force_units : bool | float
        If True and MEG sensors are scaled to 1, data will be scaled to
        base_units. If float, data will be scaled to the value supplied.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Attributes & Methods
    --------------------
    See documentation for mne.fiff.Raw

    """
    @verbose
    def __init__(self, pdf_fname, config_fname='config',
                 head_shape_fname='hs_file', rotation_x=2,
                 translation=(0.0, 0.02, 0.11), use_hpi=False,
                 force_units=False, verbose=True):

        logger.info('Reading 4D PDF file %s...' % pdf_fname)
        self.bti_info = read_pdf_info(pdf_fname)
        self._use_hpi = use_hpi
        self.bti_to_nm = _get_m_to_nm(rotation_x, translation)

        logger.info('Creating Neuromag info structure ...')
        info = self._create_raw_info()

        cals = np.zeros(info['nchan'])
        for k in range(info['nchan']):
            cals[k] = info['chs'][k]['range'] * info['chs'][k]['cal']

        self.verbose = verbose
        self.cals = cals
        self.rawdir = None
        self.proj = None
        self.comp = None
        self.fids = list()
        self._preloaded = True
        self._projector_hashes = [None]
        self.info = info

        logger.info('Reading raw data from %s...' % pdf_fname)
        # rescale
        self._data = read_raw_data()
        self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
        assert len(self._data) == len(self.info['ch_names'])
        self._times = np.arange(self.first_samp, \
                                self.last_samp + 1) / info['sfreq']
        self._projectors = [None]
        logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs' % (
                   self.first_samp, self.last_samp,
                   float(self.first_samp) / info['sfreq'],
                   float(self.last_samp) / info['sfreq']))

        if force_units is not None:
            pick_mag = pick_types(info, meg='mag', eeg=False, misc=False,
                                  eog=False, ecg=False)
            scale = 1e-15 if force_units != True else force_units
            logger.info('    Scaling raw data to %s' % str(scale))
            self._data[pick_mag] *= scale

        # remove subclass helper attributes to create a proper Raw object.
        for attr in self.__dict__:
            if attr not in Raw.__dict__:
                del attr
        logger.info('Ready.')

    @verbose
    def _create_raw_info(self):
        """ Fills list of dicts for initializing empty fiff with 4D data
        """
        info = {}
        sep = self.sep
        d = datetime.strptime(self.hdr[BTI.HDR_DATAFILE]['Session'],
                              '%d' + sep + '%y' + sep + '%m %H:%M')

        sec = time.mktime(d.timetuple())
        info['projs'] = []
        info['comps'] = []
        info['meas_date'] = np.array([sec, 0], dtype=np.int32)
        sfreq = self.hdr[BTI.HDR_FILEINFO]['Sample Frequency'][:-2]
        info['sfreq'] = float(sfreq)
        info['nchan'] = int(self.hdr[BTI.HDR_CH_GROUPS]['CHANNELS'])
        ch_names = [d['ch_label'] for d in self.bti_info.channels]
        ch_names = sorted(ch_names, key=lambda c: int(c[0:])
                          if c[0] == BTI.DATA_MEG_CH_CHAR else c)
        info['ch_names'] = list(_rename_channels(ch_names))
        ch_mapping = zip(ch_names, info['ch_names'])

        sensor_trans = dict((dict(ch_mapping)[k], v) for k, v in
                             self.hdr[BTI.HDR_CH_TRANS].items())
        info['bads'] = []  # TODO

        fspec = info.get(BTI.HDR_DATAFILE, None)
        if fspec is not None:
            fspec = fspec.split(',')[2].split('ord')[0]
            ffreqs = fspec.replace('fwsbp', '').split('-')
        else:
            logger.info('... Cannot find any filter specification' \
                  ' No filter info will be set.')
            ffreqs = 0, 1000

        info['highpass'], info['lowpass'] = ffreqs
        info['acq_pars'], info['acq_stim'] = None, None
        info['filename'] = None
        info['ctf_head_t'] = None
        info['dev_ctf_t'] = []
        info['filenames'] = []
        chs = []

        # get 4-D head_dev_t needed for appropriate
        sensor_coord_frame = FIFF.FIFFV_COORD_DEVICE
        use_identity = False
        if isinstance(self.dev_head_t_fname, str):
            logger.info('... Reading device to head transform from %s.' %
                        self.dev_head_t_fname)
            try:
                bti_sys_trans = _read_system_trans(self.dev_head_t_fname)
            except:
                logger.info('... Could not read headshape data. '
                            'Using identity transform instead')
                use_identity = True
        else:
            logger.info('... No device to head transform specified. '
                        'Using identity transform instead')
            use_identity = True

        if use_identity:
            sensor_coord_frame = FIFF.FIFFV_COORD_HEAD
            bti_sys_trans = np.array(BTI.T_IDENT, np.float32)

        logger.info('... Setting channel info structure.')
        for idx, (chan_4d, chan_vv) in enumerate(ch_mapping, 1):
            chan_info = dict((k, v) for k, v in zip(FIFF_INFO_CHS_FIELDS,
                             FIFF_INFO_CHS_DEFAULTS))
            chan_info['ch_name'] = chan_vv
            chan_info['logno'] = idx
            chan_info['scanno'] = idx

            if any([chan_vv.startswith(k) for k in ('MEG', 'RFG', 'RFM')]):
                t = _m_to_nm_coil_trans(sensor_trans[chan_vv], bti_sys_trans,
                                        self.bti_to_nm)
                chan_info['coil_trans'] = t
                chan_info['loc'] = np.roll(t.copy().T, 1, 0)[:, :3].flatten()
                chan_info['logno'] = idx

            if chan_vv.startswith('MEG'):
                chan_info['kind'] = FIFF.FIFFV_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_MAGNES_MAG
                chan_info['coord_frame'] = sensor_coord_frame
                chan_info['unit'] = FIFF.FIFF_UNIT_T

            elif chan_vv.startswith('RFM'):
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH
                chan_info['coil_type'] = FIFF.FIFFV_COIL_MAGNES_R_MAG
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['unit'] = FIFF.FIFF_UNIT_T

            elif chan_vv.startswith('RFG'):
                chan_info['kind'] = FIFF.FIFFV_REF_MEG_CH
                chan_info['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                chan_info['unit'] = FIFF.FIFF_UNIT_T_M
                if chan_4d in ('GxxA', 'GyyA'):
                    chan_info['coil_type'] = FIFF.FIFFV_COIL_MAGNES_R_GRAD_DIA
                elif chan_4d in ('GyxA', 'GzxA', 'GzyA'):
                    chan_info['coil_type'] = FIFF.FIFFV_COIL_MAGNES_R_GRAD_OFF

            elif chan_vv == 'STI 014':
                chan_info['kind'] = FIFF.FIFFV_STIM_CH
            elif chan_vv.startswith('EOG'):
                chan_info['kind'] = FIFF.FIFFV_EOG_CH
            elif chan_vv == 'ECG 001':
                chan_info['kind'] = FIFF.FIFFV_ECG_CH
            elif chan_vv == 'RSP 001':
                chan_info['kind'] = FIFF.FIFFV_RESP_CH
            elif chan_vv.startswith('EXT'):
                chan_info['kind'] = FIFF.FIFFV_MISC_CH
            elif chan_vv.startswith('UTL'):
                chan_info['kind'] = FIFF.FIFFV_MISC_CH

            chs.append(chan_info)

        info['chs'] = chs
        info['meas_id'] = None
        info['file_id'] = None

        identity = np.array(BTI.T_IDENT, np.float32)
        if self.head_shape_fname is not None:
            logger.info('... Reading digitization points from %s' %
                        self.head_shape_fname)
            info['dig'], m_h_nm_h = read_head_shape(self.head_shape_fname,
                                                     self._use_hpi)
            if m_h_nm_h is None:
                logger.info('Could not read head shape data. '
                           'Sensor data will stay in head coordinate frame')
                nm_dev_head_t = identity
            else:
                nm_to_m_sensor = _inverse_t(identity, self.bti_to_nm)
                nm_sensor_m_head = _merge_t(bti_sys_trans, nm_to_m_sensor)
                nm_dev_head_t = _merge_t(m_h_nm_h, nm_sensor_m_head)
                nm_dev_head_t[3, :3] = 0.
        else:
            logger.info('Warning. No head shape file provided. '
                        'Sensor data will stay in head coordinate frame')
            nm_dev_head_t = identity

        info['dev_head_t'] = {}
        info['dev_head_t']['from'] = FIFF.FIFFV_COORD_DEVICE
        info['dev_head_t']['to'] = FIFF.FIFFV_COORD_HEAD
        info['dev_head_t']['trans'] = nm_dev_head_t

        logger.info('Done.')
        return info
