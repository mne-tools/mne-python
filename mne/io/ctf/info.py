"""Populate measurement info
"""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD (3-clause)

from time import strptime
from calendar import timegm

import numpy as np

from ...utils import logger
from ...transforms import (apply_trans, _coord_frame_name, invert_transform,
                           combine_transforms)

from ..meas_info import _empty_info
from ..write import get_new_file_id
from ..ctf_comp import _add_kind, _calibrate_comp
from ..constants import FIFF

from .constants import CTF


def _pick_isotrak_and_hpi_coils(res4, coils, t):
    """Pick the HPI coil locations given in device coordinates"""
    if coils is None:
        return list(), list()
    dig = list()
    hpi_result = dict(dig_points=list())
    n_coil_dev = 0
    n_coil_head = 0
    for p in coils:
        if p['valid']:
            if p['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_DEVICE:
                if t is None or t['t_ctf_dev_dev'] is None:
                    raise RuntimeError('No coordinate transformation '
                                       'available for HPI coil locations')
                d = dict(kind=FIFF.FIFFV_POINT_HPI, ident=p['kind'],
                         r=apply_trans(t['t_ctf_dev_dev'], p['r']),
                         coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
                hpi_result['dig_points'].append(d)
                n_coil_dev += 1
            elif p['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
                if t is None or t['t_ctf_head_head'] is None:
                    raise RuntimeError('No coordinate transformation '
                                       'available for (virtual) Polhemus data')
                d = dict(kind=FIFF.FIFFV_POINT_HPI, ident=p['kind'],
                         r=apply_trans(t['t_ctf_head_head'], p['r']),
                         coord_frame=FIFF.FIFFV_COORD_HEAD)
                dig.append(d)
                n_coil_head += 1
    if n_coil_head > 0:
        logger.info('    Polhemus data for %d HPI coils added' % n_coil_head)
    if n_coil_dev > 0:
        logger.info('    Device coordinate locations for %d HPI coils added'
                    % n_coil_dev)
    return dig, [hpi_result]


def _convert_time(date_str, time_str):
    """Convert date and time strings to float time"""
    for fmt in ("%d/%m/%Y", "%d-%b-%Y", "%a, %b %d, %Y"):
        try:
            date = strptime(date_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError("Illegal date: %s" % date)
    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            time = strptime(time_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError('Illegal time: %s' % time)
    # MNE-C uses mktime which uses local time, but here we instead decouple
    # conversion location from the process, and instead assume that the
    # acquisiton was in GMT. This will be wrong for most sites, but at least
    # the value we obtain here won't depend on the geographical location
    # that the file was converted.
    res = timegm((date.tm_year, date.tm_mon, date.tm_mday,
                  time.tm_hour, time.tm_min, time.tm_sec,
                  date.tm_wday, date.tm_yday, date.tm_isdst))
    return res


def _get_plane_vectors(ez):
    """Get two orthogonal vectors orthogonal to ez (ez will be modified)"""
    assert ez.shape == (3,)
    ez_len = np.sqrt(np.sum(ez * ez))
    if ez_len == 0:
        raise RuntimeError('Zero length normal. Cannot proceed.')
    if np.abs(ez_len - np.abs(ez[2])) < 1e-5:  # ez already in z-direction
        ex = np.array([1., 0., 0.])
    else:
        ex = np.zeros(3)
        if ez[1] < ez[2]:
            ex[0 if ez[0] < ez[1] else 1] = 1.
        else:
            ex[0 if ez[0] < ez[2] else 2] = 1.
    ez /= ez_len
    ex -= np.dot(ez, ex) * ez
    ex /= np.sqrt(np.sum(ex * ex))
    ey = np.cross(ez, ex)
    return ex, ey


def _at_origin(x):
    """Determine if a vector is at the origin"""
    return (np.sum(x * x) < 1e-8)


def _convert_channel_info(res4, t, use_eeg_pos):
    """Convert CTF channel information to fif format"""
    nmeg = neeg = nstim = nmisc = nref = 0
    chs = list()
    for k, cch in enumerate(res4['chs']):
        cal = float(1. / (cch['proper_gain'] * cch['qgain']))
        ch = dict(scanno=k + 1, range=1., cal=cal, loc=np.zeros(12),
                  unit_mul=FIFF.FIFF_UNITM_NONE, ch_name=cch['ch_name'][:15],
                  coil_type=FIFF.FIFFV_COIL_NONE)
        del k
        chs.append(ch)
        # Create the channel position information
        pos = dict(r0=ch['loc'][:3], ex=ch['loc'][3:6], ey=ch['loc'][6:9],
                   ez=ch['loc'][9:12])
        if cch['sensor_type_index'] in (CTF.CTFV_REF_MAG_CH,
                                        CTF.CTFV_REF_GRAD_CH,
                                        CTF.CTFV_MEG_CH):
            ch['unit'] = FIFF.FIFF_UNIT_T
            # Set up the local coordinate frame
            pos['r0'][:] = cch['coil']['pos'][0]
            pos['ez'][:] = cch['coil']['norm'][0]
            # It turns out that positive proper_gain requires swapping
            # of the normal direction
            if cch['proper_gain'] > 0.0:
                pos['ez'] *= -1
            # Check how the other vectors should be defined
            off_diag = False
            if cch['sensor_type_index'] == CTF.CTFV_REF_GRAD_CH:
                # We use the same convention for ex as for Neuromag planar
                # gradiometers: pointing in the positive gradient direction
                diff = cch['coil']['pos'][0] - cch['coil']['pos'][1]
                size = np.sqrt(np.sum(diff * diff))
                if size > 0.:
                    diff /= size
                if np.abs(np.dot(diff, pos['ez'])) < 1e-3:
                    off_diag = True
                if off_diag:
                    # The off-diagonal gradiometers are an exception
                    pos['r0'] -= size * diff / 2.0
                    pos['ex'][:] = diff
                    pos['ey'][:] = np.cross(pos['ez'], pos['ex'])
            else:
                # ex and ey are arbitrary in the plane normal to ex
                pos['ex'][:], pos['ey'][:] = _get_plane_vectors(pos['ez'])
            # Transform into a Neuromag-like coordinate system
            pos['r0'][:] = apply_trans(t['t_ctf_dev_dev'], pos['r0'])
            for key in ('ex', 'ey', 'ez'):
                pos[key][:] = apply_trans(t['t_ctf_dev_dev'], pos[key],
                                          move=False)
            # Set the coil type
            if cch['sensor_type_index'] == CTF.CTFV_REF_MAG_CH:
                ch['kind'] = FIFF.FIFFV_REF_MEG_CH
                ch['coil_type'] = FIFF.FIFFV_COIL_CTF_REF_MAG
                nref += 1
                ch['logno'] = nref
            elif cch['sensor_type_index'] == CTF.CTFV_REF_GRAD_CH:
                ch['kind'] = FIFF.FIFFV_REF_MEG_CH
                if off_diag:
                    ch['coil_type'] = FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD
                else:
                    ch['coil_type'] = FIFF.FIFFV_COIL_CTF_REF_GRAD
                nref += 1
                ch['logno'] = nref
            else:
                ch['kind'] = FIFF.FIFFV_MEG_CH
                ch['coil_type'] = FIFF.FIFFV_COIL_CTF_GRAD
                nmeg += 1
                ch['logno'] = nmeg
            # Encode the software gradiometer order
            ch['coil_type'] = ch['coil_type'] | (cch['grad_order_no'] << 16)
            ch['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
        elif cch['sensor_type_index'] == CTF.CTFV_EEG_CH:
            coord_frame = FIFF.FIFFV_COORD_HEAD
            if use_eeg_pos:
                # EEG electrode coordinates may be present but in the
                # CTF head frame
                pos['r0'][:] = cch['coil']['pos'][0]
                if not _at_origin(pos['r0']):
                    if t['t_ctf_head_head'] is None:
                        logger.warning('EEG electrode (%s) location omitted '
                                       'because of missing HPI information'
                                       % (ch['ch_name']))
                        pos['r0'][:] = np.zeros(3)
                        coord_frame = FIFF.FIFFV_COORD_CTF_HEAD
                    else:
                        pos['r0'][:] = apply_trans(t['t_ctf_head_head'],
                                                   pos['r0'])
            neeg += 1
            ch['logno'] = neeg
            ch['kind'] = FIFF.FIFFV_EEG_CH
            ch['unit'] = FIFF.FIFF_UNIT_V
            ch['coord_frame'] = coord_frame
        elif cch['sensor_type_index'] == CTF.CTFV_STIM_CH:
            nstim += 1
            ch['logno'] = nstim
            ch['kind'] = FIFF.FIFFV_STIM_CH
            ch['unit'] = FIFF.FIFF_UNIT_V
            ch['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
        else:
            nmisc += 1
            ch['logno'] = nmisc
            ch['kind'] = FIFF.FIFFV_MISC_CH
            ch['unit'] = FIFF.FIFF_UNIT_V
            ch['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
    return chs


def _comp_sort_keys(c):
    """This is for sorting the compensation data"""
    return (int(c['coeff_type']), int(c['scanno']))


def _check_comp(comp):
    """Check that conversion to named matrices is, indeed possible"""
    ref_sens = None
    kind = -1
    for k, c_k in enumerate(comp):
        if c_k['coeff_type'] != kind:
            c_ref = c_k
            ref_sens = c_ref['sensors']
            kind = c_k['coeff_type']
        elif not c_k['sensors'] == ref_sens:
            raise RuntimeError('Cannot use an uneven compensation matrix')


def _conv_comp(comp, first, last, chs):
    """Add a new converted compensation data item"""
    ccomp = dict(ctfkind=np.array([comp[first]['coeff_type']]),
                 save_calibrated=False)
    _add_kind(ccomp)
    n_col = comp[first]['ncoeff']
    n_row = last - first + 1
    col_names = comp[first]['sensors'][:n_col]
    row_names = [comp[p]['sensor_name'] for p in range(first, last + 1)]
    data = np.empty((n_row, n_col))
    for ii, coeffs in enumerate(comp[first:last + 1]):
        # Pick the elements to the matrix
        data[ii, :] = coeffs['coeffs'][:]
    ccomp['data'] = dict(row_names=row_names, col_names=col_names,
                         data=data, nrow=len(row_names), ncol=len(col_names))
    mk = ('proper_gain', 'qgain')
    _calibrate_comp(ccomp, chs, row_names, col_names, mult_keys=mk, flip=True)
    return ccomp


def _convert_comp_data(res4):
    """Convert the compensation data into named matrices"""
    if res4['ncomp'] == 0:
        return
    # Sort the coefficients in our favorite order
    res4['comp'] = sorted(res4['comp'], key=_comp_sort_keys)
    # Check that all items for a given compensation type have the correct
    # number of channels
    _check_comp(res4['comp'])
    # Create named matrices
    first = 0
    kind = -1
    comps = list()
    for k in range(len(res4['comp'])):
        if res4['comp'][k]['coeff_type'] != kind:
            if k > 0:
                comps.append(_conv_comp(res4['comp'], first, k - 1,
                                        res4['chs']))
            kind = res4['comp'][k]['coeff_type']
            first = k
    comps.append(_conv_comp(res4['comp'], first, k, res4['chs']))
    return comps


def _pick_eeg_pos(c):
    """Pick EEG positions"""
    eeg = dict(coord_frame=FIFF.FIFFV_COORD_HEAD, assign_to_chs=False,
               labels=list(), ids=list(), rr=list(), kinds=list(), np=0)
    for ch in c['chs']:
        if ch['kind'] == FIFF.FIFFV_EEG_CH and not _at_origin(ch['loc'][:3]):
            eeg['labels'].append(ch['ch_name'])
            eeg['ids'].append(ch['logno'])
            eeg['rr'].append(ch['loc'][:3])
            eeg['kinds'].append(FIFF.FIFFV_POINT_EEG)
            eeg['np'] += 1
    if eeg['np'] == 0:
        return None
    logger.info('Picked positions of %d EEG channels from channel info'
                % eeg['np'])
    return eeg


def _add_eeg_pos(eeg, t, c):
    """Pick the (virtual) EEG position data"""
    if eeg is None:
        return
    if t is None or t['t_ctf_head_head'] is None:
        raise RuntimeError('No coordinate transformation available for EEG '
                           'position data')
    eeg_assigned = 0
    if eeg['assign_to_chs']:
        for k in range(eeg['np']):
            # Look for a channel name match
            for ch in c['chs']:
                if ch['ch_name'].lower() == eeg['labels'][k].lower():
                    r0 = ch['loc'][:3]
                    r0[:] = eeg['rr'][k]
                    if eeg['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
                        r0[:] = apply_trans(t['t_ctf_head_head'], r0)
                    elif eeg['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
                        raise RuntimeError(
                            'Illegal coordinate frame for EEG electrode '
                            'positions : %s'
                            % _coord_frame_name(eeg['coord_frame']))
                    # Use the logical channel number as an identifier
                    eeg['ids'][k] = ch['logno']
                    eeg['kinds'][k] = FIFF.FIFFV_POINT_EEG
                    eeg_assigned += 1
                    break

    # Add these to the Polhemus data
    fid_count = eeg_count = extra_count = 0
    for k in range(eeg['np']):
        d = dict(r=eeg['rr'][k].copy(), kind=eeg['kinds'][k],
                 ident=eeg['ids'][k], coord_frame=FIFF.FIFFV_COORD_HEAD)
        c['dig'].append(d)
        if eeg['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
            d['r'] = apply_trans(t['t_ctf_head_head'], d['r'])
        elif eeg['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
            raise RuntimeError('Illegal coordinate frame for EEG electrode '
                               'positions: %s'
                               % _coord_frame_name(eeg['coord_frame']))
        if eeg['kinds'][k] == FIFF.FIFFV_POINT_CARDINAL:
            fid_count += 1
        elif eeg['kinds'][k] == FIFF.FIFFV_POINT_EEG:
            eeg_count += 1
        else:
            extra_count += 1
    if eeg_assigned > 0:
        logger.info('    %d EEG electrode locations assigned to channel info.'
                    % eeg_assigned)
    for count, kind in zip((fid_count, eeg_count, extra_count),
                           ('fiducials', 'EEG locations', 'extra points')):
        if count > 0:
            logger.info('    %d %s added to Polhemus data.' % (count, kind))


_filt_map = {CTF.CTFV_FILTER_LOWPASS: 'lowpass',
             CTF.CTFV_FILTER_HIGHPASS: 'highpass'}


def _compose_meas_info(res4, coils, trans, eeg):
    """Create meas info from CTF data"""
    info = _empty_info(res4['sfreq'])

    # Collect all the necessary data from the structures read
    info['meas_id'] = get_new_file_id()
    info['meas_id']['usecs'] = 0
    info['meas_id']['secs'] = _convert_time(res4['data_date'],
                                            res4['data_time'])
    info['experimenter'] = res4['nf_operator']
    info['subject_info'] = dict(his_id=res4['nf_subject_id'])
    for filt in res4['filters']:
        if filt['type'] in _filt_map:
            info[_filt_map[filt['type']]] = filt['freq']
    info['dig'], info['hpi_results'] = _pick_isotrak_and_hpi_coils(
        res4, coils, trans)
    if trans is not None:
        if len(info['hpi_results']) > 0:
            info['hpi_results'][0]['coord_trans'] = trans['t_ctf_head_head']
        if trans['t_dev_head'] is not None:
            info['dev_head_t'] = trans['t_dev_head']
            info['dev_ctf_t'] = combine_transforms(
                trans['t_dev_head'],
                invert_transform(trans['t_ctf_head_head']),
                FIFF.FIFFV_COORD_DEVICE, FIFF.FIFFV_MNE_COORD_CTF_HEAD)
        if trans['t_ctf_head_head'] is not None:
            info['ctf_head_t'] = trans['t_ctf_head_head']
    info['chs'] = _convert_channel_info(res4, trans, eeg is None)
    info['nchan'] = len(info['chs'])
    info['comps'] = _convert_comp_data(res4)
    if eeg is None:
        # Pick EEG locations from chan info if not read from a separate file
        eeg = _pick_eeg_pos(info)
    _add_eeg_pos(eeg, trans, info)
    info['ch_names'] = [ch['ch_name'] for ch in info['chs']]
    logger.info('    Measurement info composed.')
    info._check_consistency()
    return info
