"""Populate measurement info."""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD-3-Clause

from time import strptime
from calendar import timegm
import os.path as op

import numpy as np

from ...utils import logger, warn, _clean_names
from ...transforms import (apply_trans, _coord_frame_name, invert_transform,
                           combine_transforms)
from ...annotations import Annotations

from ..meas_info import _empty_info
from ..write import get_new_file_id
from ..ctf_comp import _add_kind, _calibrate_comp
from ..constants import FIFF

from .constants import CTF


_ctf_to_fiff = {CTF.CTFV_COIL_LPA: FIFF.FIFFV_POINT_LPA,
                CTF.CTFV_COIL_RPA: FIFF.FIFFV_POINT_RPA,
                CTF.CTFV_COIL_NAS: FIFF.FIFFV_POINT_NASION}


def _pick_isotrak_and_hpi_coils(res4, coils, t):
    """Pick the HPI coil locations given in device coordinates."""
    if coils is None:
        return list(), list()
    dig = list()
    hpi_result = dict(dig_points=list())
    n_coil_dev = 0
    n_coil_head = 0
    for p in coils:
        if p['valid']:
            if p['kind'] in [CTF.CTFV_COIL_LPA, CTF.CTFV_COIL_RPA,
                             CTF.CTFV_COIL_NAS]:
                kind = FIFF.FIFFV_POINT_CARDINAL
                ident = _ctf_to_fiff[p['kind']]
            else:  # CTF.CTFV_COIL_SPARE
                kind = FIFF.FIFFV_POINT_HPI
                ident = p['kind']
            if p['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_DEVICE:
                if t is None or t['t_ctf_dev_dev'] is None:
                    raise RuntimeError('No coordinate transformation '
                                       'available for HPI coil locations')
                d = dict(kind=kind, ident=ident,
                         r=apply_trans(t['t_ctf_dev_dev'], p['r']),
                         coord_frame=FIFF.FIFFV_COORD_UNKNOWN)
                hpi_result['dig_points'].append(d)
                n_coil_dev += 1
            elif p['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
                if t is None or t['t_ctf_head_head'] is None:
                    raise RuntimeError('No coordinate transformation '
                                       'available for (virtual) Polhemus data')
                d = dict(kind=kind, ident=ident,
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
    """Convert date and time strings to float time."""
    if date_str == time_str == '':
        date_str = '01/01/1970'
        time_str = '00:00:00'
        logger.info('No date or time found, setting to the start of the '
                    'POSIX epoch (1970/01/01 midnight)')

    for fmt in ("%d/%m/%Y", "%d-%b-%Y", "%a, %b %d, %Y", "%Y/%m/%d"):
        try:
            date = strptime(date_str.strip(), fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError(
            'Illegal date: %s.\nIf the language of the date does not '
            'correspond to your local machine\'s language try to set the '
            'locale to the language of the date string:\n'
            'locale.setlocale(locale.LC_ALL, "en_US")' % date_str)

    for fmt in ('%H:%M:%S', '%H:%M'):
        try:
            time = strptime(time_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError('Illegal time: %s' % time_str)
    # MNE-C uses mktime which uses local time, but here we instead decouple
    # conversion location from the process, and instead assume that the
    # acquisition was in GMT. This will be wrong for most sites, but at least
    # the value we obtain here won't depend on the geographical location
    # that the file was converted.
    res = timegm((date.tm_year, date.tm_mon, date.tm_mday,
                  time.tm_hour, time.tm_min, time.tm_sec,
                  date.tm_wday, date.tm_yday, date.tm_isdst))
    return res


def _get_plane_vectors(ez):
    """Get two orthogonal vectors orthogonal to ez (ez will be modified)."""
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
    """Determine if a vector is at the origin."""
    return (np.sum(x * x) < 1e-8)


def _check_comp_ch(cch, kind, desired=None):
    if desired is None:
        desired = cch['grad_order_no']
    if cch['grad_order_no'] != desired:
        raise RuntimeError('%s channel with inconsistent compensation '
                           'grade %s, should be %s'
                           % (kind, cch['grad_order_no'], desired))
    return desired


def _convert_channel_info(res4, t, use_eeg_pos):
    """Convert CTF channel information to fif format."""
    nmeg = neeg = nstim = nmisc = nref = 0
    chs = list()
    this_comp = None
    for k, cch in enumerate(res4['chs']):
        cal = float(1. / (cch['proper_gain'] * cch['qgain']))
        ch = dict(scanno=k + 1, range=1., cal=cal, loc=np.full(12, np.nan),
                  unit_mul=FIFF.FIFF_UNITM_NONE, ch_name=cch['ch_name'][:15],
                  coil_type=FIFF.FIFFV_COIL_NONE)
        del k
        chs.append(ch)
        # Create the channel position information
        if cch['sensor_type_index'] in (CTF.CTFV_REF_MAG_CH,
                                        CTF.CTFV_REF_GRAD_CH,
                                        CTF.CTFV_MEG_CH):
            # Extra check for a valid MEG channel
            if np.sum(cch['coil']['pos'][0] ** 2) < 1e-6 or \
                    np.sum(cch['coil']['norm'][0] ** 2) < 1e-6:
                nmisc += 1
                ch.update(logno=nmisc, coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                          kind=FIFF.FIFFV_MISC_CH, unit=FIFF.FIFF_UNIT_V)
                text = 'MEG'
                if cch['sensor_type_index'] != CTF.CTFV_MEG_CH:
                    text += ' ref'
                warn('%s channel %s did not have position assigned, so '
                     'it was changed to a MISC channel'
                     % (text, ch['ch_name']))
                continue
            ch['unit'] = FIFF.FIFF_UNIT_T
            # Set up the local coordinate frame
            r0 = cch['coil']['pos'][0].copy()
            ez = cch['coil']['norm'][0].copy()
            # It turns out that positive proper_gain requires swapping
            # of the normal direction
            if cch['proper_gain'] > 0.0:
                ez *= -1
            # Check how the other vectors should be defined
            off_diag = False
            # Default: ex and ey are arbitrary in the plane normal to ez
            if cch['sensor_type_index'] == CTF.CTFV_REF_GRAD_CH:
                # The off-diagonal gradiometers are an exception:
                #
                # We use the same convention for ex as for Neuromag planar
                # gradiometers: ex pointing in the positive gradient direction
                diff = cch['coil']['pos'][0] - cch['coil']['pos'][1]
                size = np.sqrt(np.sum(diff * diff))
                if size > 0.:
                    diff /= size
                # Is ez normal to the line joining the coils?
                if np.abs(np.dot(diff, ez)) < 1e-3:
                    off_diag = True
                    # Handle the off-diagonal gradiometer coordinate system
                    r0 -= size * diff / 2.0
                    ex = diff
                    ey = np.cross(ez, ex)
                else:
                    ex, ey = _get_plane_vectors(ez)
            else:
                ex, ey = _get_plane_vectors(ez)
            # Transform into a Neuromag-like device coordinate system
            ch['loc'] = np.concatenate([
                apply_trans(t['t_ctf_dev_dev'], r0),
                apply_trans(t['t_ctf_dev_dev'], ex, move=False),
                apply_trans(t['t_ctf_dev_dev'], ey, move=False),
                apply_trans(t['t_ctf_dev_dev'], ez, move=False)])
            del r0, ex, ey, ez
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
                this_comp = _check_comp_ch(cch, 'Gradiometer', this_comp)
                ch['kind'] = FIFF.FIFFV_MEG_CH
                ch['coil_type'] = FIFF.FIFFV_COIL_CTF_GRAD
                nmeg += 1
                ch['logno'] = nmeg
            # Encode the software gradiometer order
            ch['coil_type'] = int(
                ch['coil_type'] | (cch['grad_order_no'] << 16))
            ch['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
        elif cch['sensor_type_index'] == CTF.CTFV_EEG_CH:
            coord_frame = FIFF.FIFFV_COORD_HEAD
            if use_eeg_pos:
                # EEG electrode coordinates may be present but in the
                # CTF head frame
                ch['loc'][:3] = cch['coil']['pos'][0]
                if not _at_origin(ch['loc'][:3]):
                    if t['t_ctf_head_head'] is None:
                        warn('EEG electrode (%s) location omitted because of '
                             'missing HPI information' % ch['ch_name'])
                        ch['loc'].fill(np.nan)
                        coord_frame = FIFF.FIFFV_MNE_COORD_CTF_HEAD
                    else:
                        ch['loc'][:3] = apply_trans(
                            t['t_ctf_head_head'], ch['loc'][:3])
            neeg += 1
            ch.update(logno=neeg, kind=FIFF.FIFFV_EEG_CH,
                      unit=FIFF.FIFF_UNIT_V, coord_frame=coord_frame,
                      coil_type=FIFF.FIFFV_COIL_EEG)
        elif cch['sensor_type_index'] == CTF.CTFV_STIM_CH:
            nstim += 1
            ch.update(logno=nstim, coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                      kind=FIFF.FIFFV_STIM_CH, unit=FIFF.FIFF_UNIT_V)
        else:
            nmisc += 1
            ch.update(logno=nmisc, coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                      kind=FIFF.FIFFV_MISC_CH, unit=FIFF.FIFF_UNIT_V)
    return chs


def _comp_sort_keys(c):
    """Sort the compensation data."""
    return (int(c['coeff_type']), int(c['scanno']))


def _check_comp(comp):
    """Check that conversion to named matrices is possible."""
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
    """Add a new converted compensation data item."""
    ch_names = [c['ch_name'] for c in chs]
    n_col = comp[first]['ncoeff']
    col_names = comp[first]['sensors'][:n_col]
    row_names = [comp[p]['sensor_name'] for p in range(first, last + 1)]
    mask = np.in1d(col_names, ch_names)  # missing channels excluded
    col_names = np.array(col_names)[mask].tolist()
    n_col = len(col_names)
    n_row = len(row_names)
    ccomp = dict(ctfkind=np.array([comp[first]['coeff_type']]),
                 save_calibrated=False)
    _add_kind(ccomp)

    data = np.empty((n_row, n_col))
    for ii, coeffs in enumerate(comp[first:last + 1]):
        # Pick the elements to the matrix
        data[ii, :] = coeffs['coeffs'][mask]
    ccomp['data'] = dict(row_names=row_names, col_names=col_names,
                         data=data, nrow=len(row_names), ncol=len(col_names))
    mk = ('proper_gain', 'qgain')
    _calibrate_comp(ccomp, chs, row_names, col_names, mult_keys=mk, flip=True)
    return ccomp


def _convert_comp_data(res4):
    """Convert the compensation data into named matrices."""
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
    """Pick EEG positions."""
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
    """Pick the (virtual) EEG position data."""
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
    """Create meas info from CTF data."""
    info = _empty_info(res4['sfreq'])

    # Collect all the necessary data from the structures read
    info['meas_id'] = get_new_file_id()
    info['meas_id']['usecs'] = 0
    info['meas_id']['secs'] = _convert_time(res4['data_date'],
                                            res4['data_time'])
    info['meas_date'] = (info['meas_id']['secs'], info['meas_id']['usecs'])
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
    info['comps'] = _convert_comp_data(res4)
    if eeg is None:
        # Pick EEG locations from chan info if not read from a separate file
        eeg = _pick_eeg_pos(info)
    _add_eeg_pos(eeg, trans, info)
    logger.info('    Measurement info composed.')
    info._unlocked = False
    info._update_redundant()
    return info


def _read_bad_chans(directory, info):
    """Read Bad channel list and match to internal names."""
    fname = op.join(directory, 'BadChannels')
    if not op.exists(fname):
        return []
    mapping = dict(zip(_clean_names(info['ch_names']), info['ch_names']))
    with open(fname, 'r') as fid:
        bad_chans = [mapping[f.strip()] for f in fid.readlines()]
    return bad_chans


def _annotate_bad_segments(directory, start_time, meas_date):
    fname = op.join(directory, 'bad.segments')
    if not op.exists(fname):
        return None

    # read in bad segment file
    onsets = []
    durations = []
    desc = []
    with open(fname, 'r') as fid:
        for f in fid.readlines():
            tmp = f.strip().split()
            desc.append('bad_%s' % tmp[0])
            onsets.append(np.float64(tmp[1]) - start_time)
            durations.append(np.float64(tmp[2]) - np.float64(tmp[1]))
    # return None if there are no bad segments
    if len(onsets) == 0:
        return None

    return Annotations(onsets, durations, desc, meas_date)
