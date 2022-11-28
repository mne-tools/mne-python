# -*- coding: UTF-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@stud.sbg.ac.at>
#
#
# License: BSD-3-Clause

import os.path as op
from collections import namedtuple
import re
import numpy as np
from datetime import datetime, timezone

from .._digitization import _make_dig_points
from ..base import BaseRaw
from ..meas_info import create_info
from ..tag import _coil_trans_to_loc
from ..utils import _read_segments_file, _mult_cal_one
from ..constants import FIFF
from ..ctf.trans import _quaternion_align
from ...surface import _normal_orth
from ...transforms import (apply_trans, Transform, get_ras_to_neuromag_trans,
                           combine_transforms, invert_transform,
                           _angle_between_quats, rot_to_quat)
from ...utils import check_fname, logger, verbose, _check_fname
from ...annotations import Annotations

FILE_EXTENSIONS = {
    "Curry 7": {
        "info": ".dap",
        "data": ".dat",
        "labels": ".rs3",
        "events_cef": ".cef",
        "events_ceo": ".ceo",
        "hpi": ".hpi",
    },
    "Curry 8": {
        "info": ".cdt.dpa",
        "data": ".cdt",
        "labels": ".cdt.dpa",
        "events_cef": ".cdt.cef",
        "events_ceo": ".cdt.ceo",
        "hpi": ".cdt.hpi",
    }
}
CHANTYPES = {"meg": "_MAG1", "eeg": "", "misc": "_OTHERS"}
FIFFV_CHANTYPES = {"meg": FIFF.FIFFV_MEG_CH, "eeg": FIFF.FIFFV_EEG_CH,
                   "misc": FIFF.FIFFV_MISC_CH}
FIFFV_COILTYPES = {"meg": FIFF.FIFFV_COIL_CTF_GRAD, "eeg": FIFF.FIFFV_COIL_EEG,
                   "misc": FIFF.FIFFV_COIL_NONE}
SI_UNITS = dict(V=FIFF.FIFF_UNIT_V, T=FIFF.FIFF_UNIT_T)
SI_UNIT_SCALE = dict(c=1e-2, m=1e-3, u=1e-6, µ=1e-6, n=1e-9, p=1e-12, f=1e-15)

CurryParameters = namedtuple('CurryParameters',
                             'n_samples, sfreq, is_ascii, unit_dict, '
                             'n_chans, dt_start, chanidx_in_file')


def _get_curry_version(file_extension):
    """Check out the curry file version."""
    return "Curry 8" if "cdt" in file_extension else "Curry 7"


def _get_curry_file_structure(fname, required=()):
    """Store paths to a dict and check for required files."""
    _msg = "The following required files cannot be found: {0}.\nPlease make " \
           "sure all required files are located in the same directory as {1}."
    fname = _check_fname(fname, 'read', True, 'fname')

    # we don't use os.path.splitext to also handle extensions like .cdt.dpa
    fname_base, ext = fname.split(".", maxsplit=1)
    version = _get_curry_version(ext)
    my_curry = dict()
    for key in ('info', 'data', 'labels', 'events_cef', 'events_ceo', 'hpi'):
        fname = fname_base + FILE_EXTENSIONS[version][key]
        if op.isfile(fname):
            _key = 'events' if key.startswith('events') else key
            my_curry[_key] = fname

    missing = [field for field in required if field not in my_curry]
    if missing:
        raise FileNotFoundError(_msg.format(np.unique(missing), fname))

    return my_curry


def _read_curry_lines(fname, regex_list):
    """Read through the lines of a curry parameter files and save data.

    Parameters
    ----------
    fname : str
        Path to a curry file.
    regex_list : list of str
        A list of strings or regular expressions to search within the file.
        Each element `regex` in `regex_list` must be formulated so that
        `regex + " START_LIST"` initiates the start and `regex + " END_LIST"`
        initiates the end of the elements that should be saved.

    Returns
    -------
    data_dict : dict
        A dictionary containing the extracted data. For each element `regex`
        in `regex_list` a dictionary key `data_dict[regex]` is created, which
        contains a list of the according data.

    """
    save_lines = {}
    data_dict = {}

    for regex in regex_list:
        save_lines[regex] = False
        data_dict[regex] = []

    with open(fname) as fid:
        for line in fid:
            for regex in regex_list:
                if re.match(regex + " END_LIST", line):
                    save_lines[regex] = False

                if save_lines[regex] and line != "\n":
                    result = line.replace("\n", "")
                    if "\t" in result:
                        result = result.split("\t")
                    data_dict[regex].append(result)

                if re.match(regex + " START_LIST", line):
                    save_lines[regex] = True

    return data_dict


def _read_curry_parameters(fname):
    """Extract Curry params from a Curry info file."""
    _msg_match = "The sampling frequency and the time steps extracted from " \
                 "the parameter file do not match."
    _msg_invalid = "sfreq must be greater than 0. Got sfreq = {0}"

    var_names = ['NumSamples', 'SampleFreqHz',
                 'DataFormat', 'SampleTimeUsec',
                 'NumChannels',
                 'StartYear', 'StartMonth', 'StartDay', 'StartHour',
                 'StartMin', 'StartSec', 'StartMillisec',
                 'NUM_SAMPLES', 'SAMPLE_FREQ_HZ',
                 'DATA_FORMAT', 'SAMPLE_TIME_USEC',
                 'NUM_CHANNELS',
                 'START_YEAR', 'START_MONTH', 'START_DAY', 'START_HOUR',
                 'START_MIN', 'START_SEC', 'START_MILLISEC']

    param_dict = dict()
    unit_dict = dict()

    with open(fname) as fid:
        for line in iter(fid):
            if any(var_name in line for var_name in var_names):
                key, val = line.replace(" ", "").replace("\n", "").split("=")
                param_dict[key.lower().replace("_", "")] = val
            for type in CHANTYPES:
                if "DEVICE_PARAMETERS" + CHANTYPES[type] + " START" in line:
                    data_unit = next(fid)
                    unit_dict[type] = data_unit.replace(" ", "") \
                        .replace("\n", "").split("=")[-1]

    # look for CHAN_IN_FILE sections, which may or may not exist; issue #8391
    types = ["meg", "eeg", "misc"]
    chanidx_in_file = _read_curry_lines(fname,
                                        ["CHAN_IN_FILE" +
                                         CHANTYPES[key] for key in types])

    n_samples = int(param_dict["numsamples"])
    sfreq = float(param_dict["samplefreqhz"])
    time_step = float(param_dict["sampletimeusec"]) * 1e-6
    is_ascii = param_dict["dataformat"] == "ASCII"
    n_channels = int(param_dict["numchannels"])
    try:
        dt_start = datetime(int(param_dict["startyear"]),
                            int(param_dict["startmonth"]),
                            int(param_dict["startday"]),
                            int(param_dict["starthour"]),
                            int(param_dict["startmin"]),
                            int(param_dict["startsec"]),
                            int(param_dict["startmillisec"]) * 1000,
                            timezone.utc)
        # Note that the time zone information is not stored in the Curry info
        # file, and it seems the start time info is in the local timezone
        # of the acquisition system (which is unknown); therefore, just set
        # the timezone to be UTC.  If the user knows otherwise, they can
        # change it later.  (Some Curry files might include StartOffsetUTCMin,
        # but its presence is unpredictable, so we won't rely on it.)
    except (ValueError, KeyError):
        dt_start = None  # if missing keywords or illegal values, don't set

    if time_step == 0:
        true_sfreq = sfreq
    elif sfreq == 0:
        true_sfreq = 1 / time_step
    elif not np.isclose(sfreq, 1 / time_step):
        raise ValueError(_msg_match)
    else:  # they're equal and != 0
        true_sfreq = sfreq
    if true_sfreq <= 0:
        raise ValueError(_msg_invalid.format(true_sfreq))

    return CurryParameters(n_samples, true_sfreq, is_ascii, unit_dict,
                           n_channels, dt_start, chanidx_in_file)


def _read_curry_info(curry_paths):
    """Extract info from curry parameter files."""
    curry_params = _read_curry_parameters(curry_paths['info'])
    R = np.eye(4)
    R[[0, 1], [0, 1]] = -1  # rotate 180 deg
    # shift down and back
    # (chosen by eyeballing to make the CTF helmet look roughly correct)
    R[:3, 3] = [0., -0.015, -0.12]
    curry_dev_dev_t = Transform('ctf_meg', 'meg', R)

    # read labels from label files
    label_fname = curry_paths['labels']
    types = ["meg", "eeg", "misc"]
    labels = _read_curry_lines(label_fname,
                               ["LABELS" + CHANTYPES[key] for key in types])
    sensors = _read_curry_lines(label_fname,
                                ["SENSORS" + CHANTYPES[key] for key in types])
    normals = _read_curry_lines(label_fname,
                                ['NORMALS' + CHANTYPES[key] for key in types])
    assert len(labels) == len(sensors) == len(normals)

    all_chans = list()
    dig_ch_pos = dict()
    for key in ["meg", "eeg", "misc"]:
        chanidx_is_explicit = (len(curry_params.chanidx_in_file["CHAN_IN_FILE"
                                   + CHANTYPES[key]]) > 0)    # channel index
        # position in the datafile may or may not be explicitly declared,
        # based on the CHAN_IN_FILE section in info file
        for ind, chan in enumerate(labels["LABELS" + CHANTYPES[key]]):
            chanidx = len(all_chans) + 1    # by default, just assume the
            # channel index in the datafile is in order of the channel
            # names as we found them in the labels file
            if chanidx_is_explicit:  # but, if explicitly declared, use
                # that index number
                chanidx = int(curry_params.chanidx_in_file["CHAN_IN_FILE"
                              + CHANTYPES[key]][ind])
            if chanidx <= 0:   # if chanidx was explicitly declared to be ' 0',
                # it means the channel is not actually saved in the data file
                # (e.g. the "Ref" channel), so don't add it to our list.
                # Git issue #8391
                continue
            ch = {"ch_name": chan,
                  "unit": curry_params.unit_dict[key],
                  "kind": FIFFV_CHANTYPES[key],
                  "coil_type": FIFFV_COILTYPES[key],
                  "ch_idx": chanidx
                  }
            if key == "eeg":
                loc = np.array(sensors["SENSORS" + CHANTYPES[key]][ind], float)
                # XXX just the sensor, where is ref (next 3)?
                assert loc.shape == (3,)
                loc /= 1000.  # to meters
                loc = np.concatenate([loc, np.zeros(9)])
                ch['loc'] = loc
                # XXX need to check/ensure this
                ch['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                dig_ch_pos[chan] = loc[:3]
            elif key == 'meg':
                pos = np.array(sensors["SENSORS" + CHANTYPES[key]][ind], float)
                pos /= 1000.  # to meters
                pos = pos[:3]  # just the inner coil
                pos = apply_trans(curry_dev_dev_t, pos)
                nn = np.array(normals["NORMALS" + CHANTYPES[key]][ind], float)
                assert np.isclose(np.linalg.norm(nn), 1., atol=1e-4)
                nn /= np.linalg.norm(nn)
                nn = apply_trans(curry_dev_dev_t, nn, move=False)
                trans = np.eye(4)
                trans[:3, 3] = pos
                trans[:3, :3] = _normal_orth(nn).T
                ch['loc'] = _coil_trans_to_loc(trans)
                ch['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
            all_chans.append(ch)
    dig = _make_dig_points(
        dig_ch_pos=dig_ch_pos, coord_frame='head', add_missing_fiducials=True)
    del dig_ch_pos

    ch_count = len(all_chans)
    assert (ch_count == curry_params.n_chans)  # ensure that we have assembled
    # the same number of channels as declared in the info (.DAP) file in the
    # DATA_PARAMETERS section. Git issue #8391

    # sort the channels to assure they are in the order that matches how
    # recorded in the datafile.  In general they most likely are already in
    # the correct order, but if the channel index in the data file was
    # explicitly declared we might as well use it.
    all_chans = sorted(all_chans, key=lambda ch: ch['ch_idx'])

    ch_names = [chan["ch_name"] for chan in all_chans]
    info = create_info(ch_names, curry_params.sfreq)
    with info._unlock():
        info['meas_date'] = curry_params.dt_start  # for Git issue #8398
        info['dig'] = dig
    _make_trans_dig(curry_paths, info, curry_dev_dev_t)

    for ind, ch_dict in enumerate(info["chs"]):
        all_chans[ind].pop('ch_idx')
        ch_dict.update(all_chans[ind])
        assert ch_dict['loc'].shape == (12,)
        ch_dict['unit'] = SI_UNITS[all_chans[ind]['unit'][1]]
        ch_dict['cal'] = SI_UNIT_SCALE[all_chans[ind]['unit'][0]]

    return info, curry_params.n_samples, curry_params.is_ascii


_card_dict = {'Left ear': FIFF.FIFFV_POINT_LPA,
              'Nasion': FIFF.FIFFV_POINT_NASION,
              'Right ear': FIFF.FIFFV_POINT_RPA}


def _make_trans_dig(curry_paths, info, curry_dev_dev_t):
    # Coordinate frame transformations and definitions
    no_msg = 'Leaving device<->head transform as None'
    info['dev_head_t'] = None
    label_fname = curry_paths['labels']
    key = 'LANDMARKS' + CHANTYPES['meg']
    lm = _read_curry_lines(label_fname, [key])[key]
    lm = np.array(lm, float)
    lm.shape = (-1, 3)
    if len(lm) == 0:
        # no dig
        logger.info(no_msg + ' (no landmarks found)')
        return
    lm /= 1000.
    key = 'LM_REMARKS' + CHANTYPES['meg']
    remarks = _read_curry_lines(label_fname, [key])[key]
    assert len(remarks) == len(lm)
    with info._unlock():
        info['dig'] = list()
    cards = dict()
    for remark, r in zip(remarks, lm):
        kind = ident = None
        if remark in _card_dict:
            kind = FIFF.FIFFV_POINT_CARDINAL
            ident = _card_dict[remark]
            cards[ident] = r
        elif remark.startswith('HPI'):
            kind = FIFF.FIFFV_POINT_HPI
            ident = int(remark[3:]) - 1
        if kind is not None:
            info['dig'].append(dict(
                kind=kind, ident=ident, r=r,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN))
    with info._unlock():
        info['dig'].sort(key=lambda x: (x['kind'], x['ident']))
    has_cards = len(cards) == 3
    has_hpi = 'hpi' in curry_paths
    if has_cards and has_hpi:  # have all three
        logger.info('Composing device<->head transformation from dig points')
        hpi_u = np.array([d['r'] for d in info['dig']
                          if d['kind'] == FIFF.FIFFV_POINT_HPI], float)
        hpi_c = np.ascontiguousarray(
            _first_hpi(curry_paths['hpi'])[:len(hpi_u), 1:4])
        unknown_curry_t = _quaternion_align(
            'unknown', 'ctf_meg', hpi_u, hpi_c, 1e-2)
        angle = np.rad2deg(_angle_between_quats(
            np.zeros(3), rot_to_quat(unknown_curry_t['trans'][:3, :3])))
        dist = 1000 * np.linalg.norm(unknown_curry_t['trans'][:3, 3])
        logger.info('   Fit a %0.1f° rotation, %0.1f mm translation'
                    % (angle, dist))
        unknown_dev_t = combine_transforms(
            unknown_curry_t, curry_dev_dev_t, 'unknown', 'meg')
        unknown_head_t = Transform(
            'unknown', 'head',
            get_ras_to_neuromag_trans(
                *(cards[key] for key in (FIFF.FIFFV_POINT_NASION,
                                         FIFF.FIFFV_POINT_LPA,
                                         FIFF.FIFFV_POINT_RPA))))
        with info._unlock():
            info['dev_head_t'] = combine_transforms(
                invert_transform(unknown_dev_t), unknown_head_t, 'meg', 'head')
            for d in info['dig']:
                d.update(coord_frame=FIFF.FIFFV_COORD_HEAD,
                         r=apply_trans(unknown_head_t, d['r']))
    else:
        if has_cards:
            no_msg += ' (no .hpi file found)'
        elif has_hpi:
            no_msg += ' (not all cardinal points found)'
        else:
            no_msg += ' (neither cardinal points nor .hpi file found)'
        logger.info(no_msg)


def _first_hpi(fname):
    # Get the first HPI result
    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if any(x in line for x in ('FileVersion', 'NumCoils')) or not line:
                continue
            hpi = np.array(line.split(), float)
            break
        else:
            raise RuntimeError('Could not find valid HPI in %s' % (fname,))
    # t is the first entry
    assert hpi.ndim == 1
    hpi = hpi[1:]
    hpi.shape = (-1, 5)
    hpi /= 1000.
    return hpi


def _read_events_curry(fname):
    """Read events from Curry event files.

    Parameters
    ----------
    fname : str
        Path to a curry event file with extensions .cef, .ceo,
        .cdt.cef, or .cdt.ceo

    Returns
    -------
    events : ndarray, shape (n_events, 3)
        The array of events.
    """
    check_fname(fname, 'curry event', ('.cef', '.ceo', '.cdt.cef', '.cdt.ceo'),
                endings_err=('.cef', '.ceo', '.cdt.cef', '.cdt.ceo'))

    events_dict = _read_curry_lines(fname, ["NUMBER_LIST"])
    # The first 3 column seem to contain the event information
    curry_events = np.array(events_dict["NUMBER_LIST"], dtype=int)[:, 0:3]

    return curry_events


def _read_annotations_curry(fname, sfreq='auto'):
    r"""Read events from Curry event files.

    Parameters
    ----------
    fname : str
        The filename.
    sfreq : float | 'auto'
        The sampling frequency in the file. If set to 'auto' then the
        ``sfreq`` is taken from the respective info file of the same name with
        according file extension (\*.dap for Curry 7; \*.cdt.dpa for Curry8).
        So data.cef looks in data.dap and data.cdt.cef looks in data.cdt.dpa.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """
    required = ["events", "info"] if sfreq == 'auto' else ["events"]
    curry_paths = _get_curry_file_structure(fname, required)
    events = _read_events_curry(curry_paths['events'])

    if sfreq == 'auto':
        sfreq = _read_curry_parameters(curry_paths['info']).sfreq

    onset = events[:, 0] / sfreq
    duration = np.zeros(events.shape[0])
    description = events[:, 2]

    return Annotations(onset, duration, description)


@verbose
def read_raw_curry(fname, preload=False, verbose=None):
    """Read raw data from Curry files.

    Parameters
    ----------
    fname : str
        Path to a curry file with extensions .dat, .dap, .rs3, .cdt, cdt.dpa,
        .cdt.cef or .cef.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawCurry
        A Raw object containing Curry data.
    """
    return RawCurry(fname, preload, verbose)


class RawCurry(BaseRaw):
    """Raw object from Curry file.

    Parameters
    ----------
    fname : str
        Path to a curry file with extensions .dat, .dap, .rs3, .cdt, cdt.dpa,
        .cdt.cef or .cef.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):

        curry_paths = _get_curry_file_structure(
            fname, required=["info", "data", "labels"])

        data_fname = op.abspath(curry_paths['data'])

        info, n_samples, is_ascii = _read_curry_info(curry_paths)

        last_samps = [n_samples - 1]
        raw_extras = dict(is_ascii=is_ascii)

        super(RawCurry, self).__init__(
            info, preload, filenames=[data_fname], last_samps=last_samps,
            orig_format='int', raw_extras=[raw_extras], verbose=verbose)

        if 'events' in curry_paths:
            logger.info('Event file found. Extracting Annotations from'
                        ' %s...' % curry_paths['events'])
            annots = _read_annotations_curry(curry_paths['events'],
                                             sfreq=self.info["sfreq"])
            self.set_annotations(annots)
        else:
            logger.info('Event file not found. No Annotations set.')

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        if self._raw_extras[fi]['is_ascii']:
            if isinstance(idx, slice):
                idx = np.arange(idx.start, idx.stop)
            block = np.loadtxt(
                self._filenames[0], skiprows=start, max_rows=stop - start,
                ndmin=2).T
            _mult_cal_one(data, block, idx, cals, mult)

        else:
            _read_segments_file(
                self, data, idx, fi, start, stop, cals, mult, dtype="<f4")
