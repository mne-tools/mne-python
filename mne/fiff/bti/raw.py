# Authors: Denis A. Engemann  <d.engemann@fz-juelich.de>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Yuval Harpaz <yuvharpaz@gmail.com>
#
#          simplified bsd-3 license

from mne.fiff import Raw, pick_types
from mne.fiff.constants import BTI, Bunch
from mne.fiff import FIFF
from mne.fiff.bti.read import read_int32, read_int16, read_str,\
                  read_float,  read_double, read_transform,\
                  read_char, read_int64, read_uint16, \
                  read_uint32, read_double_matrix, \
                  read_float_matrix, read_int16_matrix

from mne.fiff.bti.transforms import bti_to_vv_trans, bti_to_vv_coil_trans,\
                                    inverse_trans, merge_trans

import time
import os.path as op
from copy import deepcopy

from datetime import datetime
from itertools import count

import numpy as np

from mne.utils import verbose

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
        _n_dig_points = read_int32(fid)
        idx_points = read_double_matrix(fid, BTI.DATA_N_IDX_POINTS, 3)
        dig_points = read_double_matrix(fid, _n_dig_points, 3)

    return idx_points, dig_points


def _correct_offset(fid):
    """Align fid pointer"""
    current = fid.tell()
    if ((current % BTI.FILE_CURPOS) != 0):
        offset = current % BTI.FILE_CURPOS
        fid.seek(BTI.FILE_CURPOS - (offset), 1)


def _read_bti_header(fid):
    """ Read bti PDF header
    """
    fid.seek(BTI.FILE_END, 2)
    ftr_pos = fid.tell()
    hdr_pos = read_int64(fid)
    check_value = hdr_pos & BTI.FILE_MASK

    if ((ftr_pos + BTI.FILE_CURPOS - check_value) <= BTI.FILE_MASK):
        hdr_pos = check_value

    if ((hdr_pos % BTI.FILE_CURPOS) != 0):
        hdr_pos += (BTI.FILE_CURPOS - (hdr_pos % BTI.FILE_CURPOS))

    fid.seek(hdr_pos, 0)

    hdr_size = ftr_pos - hdr_pos

    out = {'version': read_int16(fid),
           'file_type': read_str(fid, 5)}

    fid.seek(1, 1)

    out.update({'data_format': read_int16(fid),
                'acq_mode': read_int16(fid),
                'total_epochs': read_int32(fid),
                'input_epochs': read_int32(fid),
                'total_events': read_int32(fid),
                'total_fixed_events': read_int32(fid),
                'sample_period': read_float(fid),
                'xaxis_label': read_str(fid, 16),
                'total_processes': read_int32(fid),
                'total_chans': read_int16(fid)})

    fid.seek(2, 1)
    out.update({'checksum': read_int32(fid),
                'total_ed_classes': read_int32(fid),
                'total_associated_files': read_int16(fid),
                'last_file_index': read_int16(fid),
                'timestamp': read_int32(fid)})

    fid.seek(20, 1)
    _correct_offset(fid)

    return out, hdr_size, ftr_pos


def _read_epoch(fid):
    """Read BTi PDF epoch"""
    out = {'pts_in_epoch': read_int32(fid),
           'epoch_duration': read_float(fid),
           'expected_iti': read_float(fid),
           'actual_iti': read_float(fid),
           'total_var_events': read_int32(fid),
           'checksum': read_int32(fid),
           'epoch_timestamp': read_int32(fid)}

    fid.seek(28, 1)

    return out


def _read_channel(fid):
    """Read BTi PDF channel"""
    out = {'chan_label': read_str(fid, 16),
           'chan_no': read_int16(fid),
           'attributes': read_int16(fid),
           'scale': read_float(fid),
           'yaxis_label': read_str(fid, 16),
           'valid_min_max': read_int16(fid)}

    fid.seek(6, 1)
    out.update({'ymin': read_double(fid),
                'ymax': read_double(fid),
                'index': read_int32(fid),
                'checksum': read_int32(fid),
                'off_flag': read_str(fid, 16),
                'offset': read_float(fid)})

    fid.seek(12, 1)

    return out


def _read_event(fid):
    """Read BTi PDF event"""
    out = {'event_name': read_str(fid, 16),
           'start_lat': read_float(fid),
           'end_lat': read_float(fid),
           'step_size': read_float(fid),
           'fixed_event': read_int16(fid),
           'checksum': read_int32(fid)}

    fid.seek(32, 1)
    _correct_offset(fid)

    return out


def _read_process(fid):
    """Read BTi PDF process"""

    out = {'nbytes': read_int32(fid),
           'blocktype': read_str(fid, 20),
           'checksum': read_int32(fid),
           'user': read_str(fid, 32),
           'timestamp': read_int32(fid),
           'filename': read_str(fid, 256),
           'total_steps': read_int32(fid)}

    fid.seek(32, 1)
    _correct_offset(fid)

    return out


def _read_assoc_file(fid):
    """Read BTi PDF assocfile"""

    out = {'file_id': read_int16(fid),
           'length': read_int16(fid)}

    fid.seek(32, 1)
    out['checksum'] = read_int32(fid)

    return out


def _read_pfid_ed(fid):
    """Read PDF ed file"""

    out = dict(comment_size=read_int32(fid),
             name=read_str(fid, 17))

    fid.seek(9, 1)
    out.update({'pdf_number': read_int16(fid),
                'total_events': read_int32(fid),
                'timestamp': read_int32(fid),
                'flags': read_int32(fid),
                'de_process': read_int32(fid),
                'checksum': read_int32(fid),
                'ed_id': read_int32(fid),
                'win_width': read_float(fid),
                'win_offset': read_float(fid)})

    fid.seek(8, 1)

    return out


def _read_coil_def(fid):
    """ Read coil definition """
    coildef = {'position': read_double_matrix(fid, 1, 3),
               'orientation': read_double_matrix(fid, 1, 3),
               'radius': read_double(fid),
               'wire_radius': read_double(fid),
               'turns': read_int16(fid)}

    fid.seek(fid, 2, 1)
    coildef['checksum'] = read_int32(fid)
    coildef['reserved'] = read_str(fid, 32)


def _read_ch_config(fid):
    """Read BTi channel config"""

    cfg = dict(name=read_str(fid, BTI.FILE_CONF_CH_NAME),
            chan_no=read_int16(fid),
            ch_type=read_uint16(fid),
            sensor_no=read_int16(fid))

    fid.seek(fid, BTI.FILE_CONF_CH_NEXT, 1)

    cfg.update(dict(
            gain=read_float(fid),
            units_per_bit=read_float(fid),
            yaxis_label=read_str(fid, BTI.FILE_CONF_CH_YLABEL),
            aar_val=read_double(fid),
            checksum=read_int32(fid),
            reserved=read_str(fid, BTI.FILE_CONF_CH_RESERVED)))

    _correct_offset(fid)

    # Then the channel info
    ch_type, chan = cfg['ch_type'], dict()
    chan['dev'] = {'size': read_int32(fid),
                   'checksum': read_int32(fid),
                   'reserved': read_str(fid, 32)}
    if ch_type in [BTI.CHTYPE_MEG, BTI.CHTYPE_REF]:
        chan['loops'] = [_read_coil_def(fid) for d in
                        range(chan['dev']['total_loops'])]

    elif ch_type == BTI.CHTYPE_EEG:
        chan['impedance'] = read_float(fid)
        chan['padding'] = read_str(fid, BTI.FILE_CONF_CH_PADDING)
        chan['transform'] = read_transform(fid)
        chan['reserved'] = read_char(fid, BTI.FILE_CONF_CH_RESERVED)

    elif ch_type in [BTI.CHTYPE_TRIGGER,  BTI.CHTYPE_EXTERNAL,
                     BTI.CHTYPE_UTILITY, BTI.CHTYPE_DERIVED]:
        chan['user_space_size'] = read_int32(fid)
        if ch_type == BTI.CHTYPE_TRIGGER:
            fid.seek(2, 1)
        chan['reserved'] = read_str(fid, BTI.FILE_CONF_CH_RESERVED)

    elif ch_type == BTI.CHTYPE_SHORTED:
        chan['reserved'] = read_str(fid, BTI.FILE_CONF_CH_RESERVED)

    cfg['chan'] = chan

    _correct_offset(fid)

    return cfg


def read_config(fname):
    """Read BTi system config file

    Parameters
    ----------
    fname : str
        The absolute path to the config file

    Returns
    -------
    cfg : Bunch
        A dict like structure including the config blocks found.

    """
    fid = open(fname, 'rb')

    cfg = Bunch()

    cfg.hdr = {'version': read_int16(fid),
               'site_name': read_str(fid, 32),
               'dap_hostname': read_str(fid, 16),
               'sys_type': read_int16(fid),
               'sys_options': read_int32(fid),
               'supply_freq': read_int16(fid),
               'total_chans': read_int16(fid),
               'system_fixed_gain': read_float(fid),
               'volts_per_bit': read_float(fid),
               'total_sensors': read_int16(fid),
               'total_user_blocks': read_int16(fid),
               'next_der_chan_no': read_int16(fid)}

    fid.seek(2, 1)  # compensate for padding

    cfg.checksum = read_uint32(fid)
    cfg.reserved = read_char(fid, 32)
    cfg.transforms = read_transform(fid)

    cfg.user_blocks = list()
    for block in range(cfg.hdr['total_user_blocks']):
        _correct_offset(fid)
        ub = dict()
        cfg.user_blocks += [ub]
        ub['hdr'] = {'nbytes': read_int32(fid),
                     'kind': read_str(fid, 20),
                     'checksum': read_int32(fid),
                     'username': read_str(fid, 32),
                     'timestamp': read_int32(fid),
                     'user_space_size': read_int32(fid),
                     'reserved': read_char(fid, 32)}

        ub['data'] = dict()
        kind, dta = ub['hdr']['kind'], ub['data']
        fid.seek(4, 1)  # very important anti-padding (see bst / FT)
        if kind in [v for k, v in BTI.items() if k[:5] == 'UB_B_']:
            if kind == BTI.UB_B_MAG_INFO:
                dta['version'] = read_int32(fid)
                fid.seek(20, 1)
                dta['headers'] = list()
                for hdr in range(6):
                    d = {'name': read_str(fid, 16),
                         'transform': read_transform(fid),
                         'units_per_bit': read_float(fid)}
                    dta['headers'] += [d]
                    fid.seek(20, 1)

            elif kind == BTI.UB_B_COH_POINTS:
                dta['n_points'] = read_int32(fid)
                dta['status'] = read_int32(fid)
                dta['points'] = []
                for pnt in xrange(16):
                    d = {'pos': read_double_matrix(fid, 1, 3),
                         'direction': read_double_matrix(fid, 1, 3),
                         'error': read_double(fid)}
                    dta['points'] += [d]

            elif kind == BTI.UB_B_CCP_XFM_BLOCK:
                dta['method'] = read_int32(fid)
                fid.seek(4, 1)
                dta['transform'] = read_transform(fid)

            elif kind == BTI.UB_B_EEG_LOCS:
                dta['electrodes'] = []
                while True:
                    d = {'label': read_str(fid, 16),
                         'location': read_double_matrix(fid, 1, 3)}
                    if d['label'] == BTI.FILE_CONF_UBLOCK_ELABEL_END:
                        break
                    dta['electrodes'] += [d]

            elif kind in [BTI.UB_B_WHC_CHAN_MAP_VER,
                          BTI.UB_B_WHS_SUBSYS_VER]:
                dta['version'] = read_int16(fid)
                dta['struct_size'] = read_int16(fid)
                dta['entries'] = read_int16(fid)
                fid.seek(8, 1)

            elif kind == BTI.UB_B_WHC_CHAN_MAP:
                num_channels = None
                for block in cfg['user_blocks']:
                    if block['hdr']['kind'] == BTI.UB_B_WHC_CHAN_MAP_VER:
                        num_channels = block['data']['entries']
                        break

                if num_channels is None:
                    raise ValueError('Cannot find block %s to determine number'
                                     'of channels' % BTI.UB_B_WHC_CHAN_MAP_VER)

                dta['channels'] = list()
                for i in xrange(num_channels):
                    d = {'subsys_type': read_int16(fid),
                         'subsys_num': read_int16(fid),
                         'card_num': read_int16(fid),
                         'chan_num': read_int16(fid),
                         'recdspnum': read_int16(fid)}
                    dta['channels'] += [d]
                    fid.seek(8, 1)

            elif kind == BTI.UB_B_WHS_SUBSYS:
                num_subsys = None
                for block in cfg['user_blocks']:
                    if block['hdr']['kind'] == BTI.UB_B_WHS_SUBSYS_VER:
                        num_subsys = block['data']['entries']
                        break

                if num_subsys is None:
                    raise ValueError('Cannot find block %s to determine'
                                     ' number o'
                                     'f subsystems' % BTI.UB_B_WHS_SUBSYS_VER)

                dta['subsys'] = list()
                for sub_key in range(num_subsys):
                    d = {'subsys_type': read_int16(fid),
                         'subsys_num': read_int16(fid),
                         'cards_per_sys': read_int16(fid),
                         'channels_per_card': read_int16(fid),
                         'card_version': read_int16(fid)}

                    fid.seek(2, 1)

                    d.update({'offsetdacgain': read_float(fid),
                              'squid_type': read_int32(fid),
                              'timesliceoffset': read_int16(fid),
                              'padding': read_int16(fid),
                              'volts_per_bit': read_float(fid)})

                    dta['subsys'] += [d]

            elif kind == BTI.UB_B_CH_LABELS:
                dta['version'] = read_int32(fid)
                dta['entries'] = read_int32(fid)
                fid.seek(16, 1)

                dta['labels'] = list()
                for label in xrange(dta['entries']):
                    dta['labels'] += [read_str(fid, 16)]

            elif kind == BTI.UB_B_CALIBRATION:
                dta['sensor_no'] = read_int16(fid)
                fid.seek(2, 1)
                dta['timestamp'] = read_int32(fid)
                dta['logdir'] = read_str(fid, 256)

            elif kind == BTI.UB_B_SYS_CONFIG_TIME:
                dta['sysconfig_name'] = read_str(fid, 512)
                dta['timestamp'] = read_int32(fid)

            elif kind == BTI.UB_B_DELTA_ENABLED:
                dta['delta_enabled'] = read_int16(fid)

            elif kind in [BTI.UB_B_E_TABLE_USED, BTI.UB_B_E_TABLE]:

                dta['hdr'] = {'version': read_int32(fid),
                              'entry_size': read_int32(fid),
                              'n_entries': read_int32(fid),
                              'filtername': read_str(fid, 16),
                              'n_e_values': read_int32(fid),
                              'reserved': read_str(fid, 28)}

                if dta['hdr']['version'] == BTI.FILE_CONF_UBLOCK_ETAB_HDR_VER:
                    size = BTI.FILE_CONF_UBLOCK_ETAB_CH_NAME
                    dta['ch_names'] = [read_str(fid, size) for ch in
                                          range(dta['hdr']['n_entries'])]
                    dta['e_ch_names'] = [read_str(fid, size) for ch in
                                          range(dta['hdr']['n_e_values'])]

                    rows = dta['hdr']['n_entries']
                    cols = dta['hdr']['n_e_values']
                    dta['etable'] = read_float_matrix(fid, rows, cols)
                else:
                    # handle MAGNES2500 naming scheme
                    dta['ch_names'] = ['WH2500'] * dta['hdr']['n_e_values']
                    dta['hdr']['n_e_values'] = 6
                    dta['e_ch_names'] = BTI_WH2500_REF_MAG
                    rows = dta['hdr']['n_entries']
                    cols = dta['hdr']['n_e_values']
                    dta['etable'] = read_float_matrix(fid, rows, cols)

                    _correct_offset(fid)

            elif any([kind == BTI.UB_B_WEIGHTS_USED,
                      kind[:4] == BTI.UB_B_WEIGHT_TABLE]):
                dta['hdr'] = {'version': read_int32(fid),
                              'entry_size': read_int32(fid),
                              'n_entries': read_int32(fid),
                              'name': read_str(fid, 32),
                              'description': read_str(fid, 80),
                              'n_anlg': read_int32(fid),
                              'n_dsp': read_int32(fid),
                              'reserved': read_str(fid, 72)}

                if dta['hdr']['version'] == 2:
                    dta['ch_names'] = [read_str(fid, 16) for ch in
                                       range(dta['hdr']['n_entries'])]
                    dta['anlg_ch_names'] = [read_str(fid, 16) for ch in
                                            range(dta['hdr']['n_anlg'])]

                    dta['dsp_ch_names'] = [read_str(fid, 16) for ch in
                                           range(dta['hdr']['n_dsp'])]

                    rows = dta['hdr']['n_entries']
                    cols = dta['hdr']['n_anlg']
                    dta['dsp_wts'] = read_float_matrix(fid, rows, cols)
                    dta['anlg_wts'] = read_int16_matrix(fid, rows, cols)

                else:  # handle MAGNES2500 naming scheme
                    dta['ch_names'] = ['WH2500'] * dta['hdr']['n_entries']
                    dta['anlg_ch_names'] = BTI_WH2500_REF_MAG[:3]
                    dta['hdr']['n_anlg'] = len(dta['anlg_ch_names'])
                    dta['dsp_ch_names'] = BTI_WH2500_REF_GRAD
                    dta['hdr.n_dsp'] = len(dta['dsp_ch_names'])
                    dta['anlg_wts'] = np.zeros((dta['hdr']['n_entries'],
                                            dta['hdr']['n_anlg']), dtype='i2')
                    dta['dsp_wts'] = np.zeros((dta['hdr']['n_entries'],
                                            dta['hdr']['n_dsp']), dtype='f4')
                    for n in range(dta['hdr']['n_entries']):
                        dta['anlg_wts'][d, :] = read_int16_matrix(fid, 1,
                                                        dta['hdr']['n_anlg'])
                        read_int16(fid)
                        dta['dsp_wts'][d, :] = read_float_matrix(fid, 1,
                                                    dta['hdr']['n_dsp'])

                    _correct_offset(fid)

            elif kind == BTI.UB_B_TRIG_MASK:
                dta['version'] = read_int32(fid)
                dta['entries'] = read_int32(fid)
                fid.seek(16, 1)

                dta['masks'] = []
                for entry in range(dta['entries']):
                    d = {'name': read_str(fid,
                                    BTI.FILE_CONF_UBLOCK_MASK_NAME),
                         'nbits': read_uint16(fid),
                         'shift': read_uint16(fid),
                         'mask': read_uint32(fid)}
                    dta['masks'] += [d]
                    fid.seek(8, 1)

        else:
            dta['unknown'] = {'hdr': read_char(fid,
                                ub['hdr']['user_space_size'])}

        # _correct_offset(fid)  # after reading.
        # fid.seek(ub['hdr']['user_space_size'], 1)

    cfg.channels = []

    for channel in range(cfg.hdr['total_chans']):
        ch = dict()
        cfg.channels += [ch]
        ch['hdr'] = {'name': read_str(fid, 16),
                     'chan_no': read_int16(fid),
                     'ch_type': read_uint16(fid),
                     'sensor_no': read_int16(fid)}

        fid.seek(2, 1)
        ch.update({'gain': read_float(fid),
                   'units_per_bit': read_float(fid),
                   'yaxis_label': read_str(fid, 16),
                   'aar_val': read_double(fid),
                   'checksum': read_int32(fid),
                   'reserved': read_str(fid, 32)})

        _correct_offset(fid)  # before and after
        dev_header = lambda x: {'size': read_int32(x),
                                'checksum': read_int32(x),
                                'reserved': read_str(x, 32)}
        dta = ch['data'] = dict()
        if ch['hdr']['ch_type'] in [BTI.CHTYPE_MEG,
                                    BTI.CHTYPE_REFERENCE]:
            dev = {'hdr': dev_header(fid),
                   'inductance': read_float(fid),
                   'padding': read_str(fid, 4),
                   'transform': read_transform(fid),
                   'xform_flag': read_int16(fid),
                   'total_loops': read_int16(fid)}

            fid.seek(4, 1)
            dev['reserved'] = read_str(fid, 32)
            dta.update({'dev': dev, 'loops': []})
            for loop in range(dev['total_loops']):
                d = {'position': read_double_matrix(fid, 1, 3),
                     'orientation': read_double_matrix(fid, 1, 3),
                     'radius': read_double(fid),
                     'wire_radius': read_double(fid),
                     'turns': read_int16(fid)}
                fid.seek(2, 1)
                d['checksum'] = read_int32(fid)
                d['reserved'] = read_str(fid, 32)
                dta['loops'] += [d]

        elif ch['hdr']['ch_type'] == BTI.CHTYPE_EEG:
            dta = {'hrd': dev_header(fid),
                   'impedance': read_float(fid),
                   'padding': read_str(fid, 4),
                   'transform': read_transform(fid),
                   'reserved': read_char(fid, 32)}

        elif ch['hdr']['ch_type'] == BTI.CHTYPE_EXTERNAL:
            dta = {'hdr': dev_header(fid),
                   'user_space_size': read_int32(fid),
                   'reserved': read_str(fid, 32)}

        elif ch['hdr']['ch_type'] == BTI.CHTYPE_TRIGGER:
            dta = {'hdr': dev_header(fid),
                   'user_space_size': read_int32(fid)}
            fid.seek(2, 1)
            dta['reserved'] = read_str(fid, 32)

        elif ch['hdr']['ch_type'] in [BTI.CHTYPE_UTILITY, BTI.CHTYPE_DERIVED]:
            dta = {'hdr': dev_header(fid),
                   'user_space_size': read_int32(fid),
                   'reserved': read_str(fid, 32)}

        elif ch['hdr']['ch_type'] == BTI.CHTYPE_SHORTED:
            dta = {'hdr': dev_header(fid),
                   'reserved': read_str(fid, 32)}

        _correct_offset(fid)  # after each reading

    return cfg


def _read_data(self, fname, config_fname, start=None, stop=None,
               dtype='f8'):
    """ Helper function: read Bti processed data file (PDF)

    Parameters
    ----------
    fname
    start : int | None
        The number of the first time slice to read. If None, all data will
        be read from the begninning.
    stop : int | None
        The number of the last time slice to read. If None, all data will
        be read to the end.
    dtype : str | dtype object
        The type the data are casted to.

    Returns
    -------
    data : ndarray
        The measurement data, a channels x timeslices array.
    """

    fid = open(self.fname, 'rb')

    info, hdr_size, ftr_pos = _read_bti_header(fid)

    info['epochs'] = [_read_epoch(fid) for epoch in
                       range(info['total_epochs'])]

    info['chs'] = [_read_channel(fid) for ch in
                   range(info['total_chans'])]

    info['events'] = [_read_event(fid) for event in
                      range(info['total_events'])]

    info['processes'] = [_read_process(fid) for process in
                         range(info['total_processes'])]

    info['assocfiles'] = [_read_assoc_file(fid) for af in
                          range(info['total_associated_files'])]

    info['edclasses'] = [_read_pfid_ed(fid) for ed_class in
                         range(info['total_ed_classes'])]

    fid.seek(0, 1)
    info['extradata'] = fid.read(ftr_pos - fid.tell())
    info['fid'] = fid

    info['total_slices'] = sum(e['pts_in_epoch'] for e in
                               info['epochs'])

    info['dtype'] = DTYPES[info['data_format']]
    bps = info['dtype'].itemsize * info['total_chans']
    info['bytes_per_slice'] = bps

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

    mapping = [(i, d['chan_label']) for i, d in
               enumerate(info['channels'])]
    sort = sorted(mapping, key=lambda c: int(c[1][1:])
                  if c[1][0] == BTI.DATA_MEG_CH_CHAR else c[1])
    order = [idx[0] for idx in sort]

    cfg = read_config(config_fname)
    info['bti_transform'] = cfg.transforms

    # augment channel list by according info from config.
    # get channels from config present in PDF
    chans = sorted(info['channels'], key=lambda c: c['chan_no'])
    chans_cfg = sorted(cfg.channels, key=lambda c: c['hdr']['chan_no'])

    # check we got everything
    assert [c['hdr']['chan_no'] for c in chans_cfg] == \
           [c['chan_no'] for c in chans]

    # transfer channel info from config to channel info
    for ch, ch_cfg in zip(chans, chans_cfg):
        ch['upb'] = ch_cfg['units_per_bit'].copy()
        ch['gain'] = deepcopy(ch_cfg['gain'])
        ch['name'] = ch_cfg['name']
        ch['loc'] = ch_cfg['data']['transform'].copy()

    for idx in order:  # unpack data as float
        cal = info['chs'][idx]['scale']
        cal *= info['chs'][idx]['gain']
        cal /= info['chs'][idx]['upb']
        data[idx] *= cal

    return info, data


class Raw(Raw):
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
        if not op.exists(op.join(op.abspath(op.curdir),
            op.basename(config_fname))):
            raise ValueError('Could not find the config file %s. Please check'
                             ' whether you are in the right directory '
                             'or pass the full name' % config_fname)

        bti_info, bti_data = _read_data(pdf_fname, config_fname)
        self._use_hpi = use_hpi
        self.bti_sys_trans = bti_info['bti_transform']

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
        self._data = bti_data
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
        logger.info('... Setting channel info structure.')
        for idx, (chan_4d, chan_vv) in enumerate(ch_mapping, 1):
            chan_info = dict((k, v) for k, v in zip(FIFF_INFO_CHS_FIELDS,
                             FIFF_INFO_CHS_DEFAULTS))
            chan_info['ch_name'] = chan_vv
            chan_info['logno'] = idx
            chan_info['scanno'] = idx

            if any([chan_vv.startswith(k) for k in ('MEG', 'RFG', 'RFM')]):
                t = bti_to_vv_coil_trans(sensor_trans[chan_vv],
                                         self.bti_sys_trans,
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
                nm_to_m_sensor = inverse_trans(identity, self.bti_to_nm)
                nm_sensor_m_head = merge_trans(bti_sys_trans, nm_to_m_sensor)
                nm_dev_head_t = merge_trans(m_h_nm_h, nm_sensor_m_head)
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


def read_raw_bti(raw_fname):
    pass
