# Author: Dominik Welke <dominik.welke@web.de>
#
# License: BSD-3-Clause

import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ..constants import FIFF

from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, warn


@fill_doc
def read_raw_eyelink(fname, preload=False, verbose=None,
                     annotate_missing=False, interpolate_missing=False,
                     read_eye_events=True):
    """Reader for an XXX file.

    Parameters
    ----------
    fname : str
        Path to the XXX data file (.XXX).
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawEyetrack
        A Raw object containing eyetracker data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawEyelink(fname, preload=preload, verbose=verbose,
                      annotate_missing=annotate_missing,
                      interpolate_missing=interpolate_missing,
                      read_eye_events=read_eye_events)


def set_channelinfo_eyetrack(inst, channel_dict=None):
    """
    updates channel information for eyetrack channels
    (channel type has to be set to "eyetrack" before, as other types will be
    ignored)
    this handles the deeper implementation using FIFF types

    user can either rely on automatic extraction from channel name (str), or
    provide a dictionary with entries for each channel name (e.g. "R-AREA")

    example:
    channel_dict = {
        "L-GAZE-X": {
            "ch_type": "GAZE",
            "ch_unit": "PIX",
            "ch_eye": "L",
            "ch_xy": "X"
            },
        "R-AREA": {
            "ch_type": "PUPIL",
            "ch_unit": "AU",
            "ch_eye": "R"
            }
        }
    """
    inst.info = _set_fiff_channelinfo_eyetrack(inst.info, channel_dict=None)
    return inst


@fill_doc
class RawEyelink(BaseRaw):
    """Raw object from an XXX file.

    Parameters
    ----------
    fname : str
        Path to the data file (.XXX).
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None,
                 annotate_missing=True, interpolate_missing=True,
                 read_eye_events=True):
        from ...preprocessing import annotate_nan
        from ...preprocessing.interpolate import interpolate_nan

        logger.info('Loading {}'.format(fname))

        # load data
        ftype = fname.split('.')[-1]
        if ftype == 'asc':
            sfreq = 500
            eye = 'BINO'
            pos = True
            pupil = True

            info, data, first_sample, annot, meas_date = \
                self._parse_eyelink_asc(
                    fname,
                    sfreq=sfreq,
                    eye=eye,
                    pos=pos,
                    pupil=pupil,
                    read_eye_events=read_eye_events)
        elif ftype == 'edf':
            raise NotImplementedError('Eyelink .edf files not supported, yet')
        else:
            raise ValueError(
                'This reader can only read eyelink .asc (or .edf) files. '
                'Got .{} instead'.format(ftype))

        # create mne object
        super(RawEyelink, self).__init__(  # or just super().__init__( ?
            info, preload=data, filenames=[fname], verbose=verbose)
        # set meas_date
        self.set_meas_date(meas_date)
        # set annotiations
        self.set_annotations(annot)
        # annotate missing data
        if annotate_missing:
            annot_bad = annotate_nan(self)
            self.set_annotations(self.annotations.__add__(annot_bad))
        # interpolate missing data
        if interpolate_missing:
            self = interpolate_nan(self)

    def _parse_eyelink_asc(self, fname, sfreq, eye='BINO', pos=True,
                           pupil=True, read_eye_events=True):
        from .ParseEyeLinkAscFiles_ import ParseEyeLinkAsc_
        import datetime as dt

        # read the header (to extract relevant information)
        with open(fname, 'r') as f:
            d_header = []
            for l in f.readlines()[:100]:  # restrict to first 100 lines
                d_header.append(l) if ('**' in l) else None

        for l in d_header:
            if 'DATE:' in l:
                datetime_str = l.strip('\n').split('DATE: ')[-1]
            # we can get more, e.g. camera settings
            if 'CAMERA:' in l:
                cam = l.strip('\n').split('CAMERA: ')[-1]

        meas_date = None
        if 'datetime_str' in locals():
            meas_date = dt.datetime.strptime(datetime_str,
                                             '%a %b %d %H:%M:%S %Y')
            meas_date = meas_date.replace(tzinfo=dt.timezone.utc)

        if meas_date is None:
            warn("Extraction of measurement date from asc file failed. "
                 "Please report this as a github issue. "
                 "The date is being set to January 1st, 2000, ")
            meas_date = dt.datetime(2000, 1, 1, 0, 0, 0,
                                    tzinfo=dt.timezone.utc)

        # set parameter
        ch_names = []
        if pos:
            ch_names.append('X')
            ch_names.append('Y')
        if pupil:
            ch_names.append('Pupil')
        if eye == 'LEFT':
            ch_names = ['L' + ch for ch in ch_names]
        elif eye == 'RIGHT':
            ch_names = ['R' + ch for ch in ch_names]
        elif eye == 'BINO':
            ch_names = [['R' + ch, 'L' + ch] for ch in ch_names]
            ch_names = [x for xs in ch_names for x in xs]
        ch_names.sort()

        n_chan = len(ch_names)
        ch_types = ['eyetrack'] * n_chan  # ch_types = ['eyetrack'] * n_chan

        info = create_info(ch_names, sfreq, ch_types)

        # set correct channel type and location
        info = _set_fiff_channelinfo_eyetrack(info)

        # load data
        df_recalibration, df_msg, df_fix, df_sacc, df_blink, df_samples = \
            ParseEyeLinkAsc_(fname)

        # clean out misread data
        # assert tSample > 0
        df_samples = df_samples[df_samples.tSample > 0]
        # clean out rows with duplicate sampletime, as these are prob errors
        # tbd
        # also clean out rows where number of nans doesnt fit!
        # nan_lines = df_samples.isna().any(axis=1)
        # tmp = df_samples[nan_lines]

        # mod_factor = 2 if (pos and not pupil) else 3 if (pos and pupil) else1
        # (df_samples.isna().sum(axis=1) % mod_factor) != 0

        # transpose to correct sfreq
        # samples = df_samples['tSample'].apply(lambda x: x*sfreq/1000.)
        # first_sample = samples.min()
        shiftfun = lambda a, b: a + b - (a % b) if (a % b != 0) else a

        sinterval = int(1000 / sfreq)  # d btw samples in ms
        df_samples['tSample'] = df_samples['tSample'].apply(
            lambda x: shiftfun(x, sinterval))

        # fix epoched recording by making it contiuous
        df_samples['tSample'] = df_samples['tSample'].astype(int)
        df_samples.index = df_samples.tSample

        tmin = shiftfun(df_msg['time'].min(), sinterval)
        tmax = df_samples.tail()['tSample'].max()

        samples_new = list(range(tmin, tmax + 1, sinterval))
        df_samples = df_samples.reindex(samples_new)

        # get data for selected channels
        try:
            data = df_samples[ch_names].to_numpy().T
        except KeyError:
            raise ValueError(
                "provided eye={} parameter doesn't match the data".format(eye))

        # make annotations
        # transpose to [s] relative to tmin
        onset_msg = list((df_msg['time'] - tmin) / 1000)  # data comes in ms
        duration_msg = [0] * len(onset_msg)
        description_msg = list(df_msg['text'])
        annot = Annotations(onset_msg, duration_msg, description_msg,
                            orig_time=None, ch_names=None)

        if read_eye_events:
            # add blinks
            onset_blink = list((df_blink['tStart'] - tmin) / 1000)
            duration_blink = list(df_blink['duration'] / 1000)
            description_blink = list(list('Blink - ' + df_blink['eye']))
            annot.append(onset_blink, duration_blink, description_blink,
                         ch_names=None)
            # add saccades
            onset_sacc = list((df_sacc['tStart'] - tmin) / 1000)
            duration_sacc = list(df_sacc['duration'] / 1000)
            description_sacc = list('Saccade - ' + df_sacc['eye'])
            annot.append(onset_sacc, duration_sacc, description_sacc,
                         ch_names=None)

        first_sample = int(tmin / sfreq)  # good enough

        return info, data, first_sample, annot, meas_date


def _eyetrack_channelinfo_from_chname(ch_name):
    """
    try finding out what specs the eyetrack channel has
    possible units:
    pupil - arbitrary units, mm (diameter)
    gaze position - px, deg, ...
    eye - left, right
    """

    # try improveing detection by spliting the channel name
    seplist = ['-', '_', '/']  # assume only one is present
    sep_present = False
    ch_parts = []
    for sep in seplist:
        if sep in ch_name:
            sep_present = True
            ch_parts = ch_name.lower().split(sep)

    # extract info
    ch_type = (
        'PUPIL' if (
                'pupil' in ch_parts or
                'area' in ch_parts or
                'pupil' in ch_name.lower() or
                'area' in ch_name.lower()) else
        'GAZE' if (
                'gaze' in ch_parts or
                'pos' in ch_parts or
                'y' in ch_parts or
                'x' in ch_parts or
                'gaze' in ch_name.lower() or
                'pos' in ch_name.lower() or
                'y' in ch_name.lower() or
                'x' in ch_name.lower()
        # potentially problematic, can be in "px" (no prob though, as this would be pos anyway)
        ) else
        'unknown')

    ch_unit = (
        'PIX' if (
                'pixel' in ch_name.lower() or
                'pix' in ch_name.lower()) or
                'px' in ch_name.lower() else
        'DEG' if (
                'deg' in ch_name.lower()) else
        'MM' if (
                'mm' in ch_name.lower()) else
        'AU' if (
                'au' in ch_name.lower()) else
        'unknown')

    ch_eye = (
        'L' if (
                'l' in ch_parts or
                'left' in ch_parts or
                'left' in ch_name.lower()) else
        'R' if (
                'r' in ch_parts or
                'right' in ch_parts or
                'right' in ch_name.lower()) else
        'unknown')

    ch_xy = (
        'Y' if ('y' in ch_parts or
                'y' in ch_name.lower()) else
        'X' if ('x' in ch_parts or  # potentially problematic, as x is in px.
                ('x' in ch_name.lower() and not 'px' in ch_name.lower()) or
                ('x' in ch_name.lower() and not 'pix' in ch_name.lower())) else
        'unknown'
    )
    return ch_type, ch_unit, ch_eye, ch_xy


def _set_fiff_channelinfo_eyetrack(info, channel_dict=None):
    """
    set the correct FIFF constants for eyetrack channels
    """
    # set correct channel type and location
    loc = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    for i_ch, ch_name in enumerate(info.ch_names):
        # only do it for eyetrack channels
        if info['chs'][i_ch]['kind'] != FIFF.FIFFV_EYETRACK_CH:
            continue

        # if no channel dict is provided, try finding out what happened
        # possible units:
        # pupil - arbitrary units, mm (diameter)
        # gaze position - px, deg, ...
        # eye - left, right, (binocular average)
        ch_type, ch_unit, ch_eye, ch_xy = _eyetrack_channelinfo_from_chname(ch_name)
        if type(channel_dict) == dict:  # if there is user input, overwrite guesses
            if ch_name not in channel_dict.keys():
                Warning("'channel_dict' was provided, but contains no info "
                        "on channel '{}'".format(ch_name) )
            else:
                keys = channel_dict[ch_name].keys()
                ch_type = channel_dict[ch_name]['ch_type'] if 'ch_type' in keys else 'unknown'
                ch_unit = channel_dict[ch_name]['ch_unit'] if 'ch_unit' in keys else 'unknown'
                ch_eye = channel_dict[ch_name]['ch_eye'] if 'ch_eye' in keys else 'unknown'
                ch_xy = channel_dict[ch_name]['ch_xy'] if 'ch_xy' in keys else 'unknown'

        # set coil type
        if ch_type not in ['pupil', 'gaze']:
            Warning("couldn't determine channel type for channel {}."
                    "defaults to 'gaze'.".format(ch_name))
        coil_type = (
            FIFF.FIFFV_COIL_EYETRACK_PUPIL if (ch_type == 'PUPIL') else
            FIFF.FIFFV_COIL_EYETRACK_POS)
        info['chs'][i_ch]['coil_type'] = coil_type

        # set unit
        if ch_unit not in ['PIX', 'AU']:  # add DEG, MM2 later
            Warning("couldn't determine unit for channel {}."
                    "defaults to 'arbitrary units'.".format(ch_name))
        unit = (FIFF.FIFF_UNIT_PX if (ch_unit == 'PIX') else
                FIFF.FIFF_UNIT_DEG if (ch_unit == 'DEG') else
                FIFF.FIFF_UNIT_MM if (ch_unit == 'MM') else
                FIFF.FIFF_UNIT_UNITLESS)
        info['chs'][i_ch]['unit'] = unit

        # set eye and x/y coordinate
        if ch_eye not in ['L', 'R']:
            Warning("couldn't determine eye for channel {}."
                    "defaults to 'NaN'.".format(ch_name))
        loc[3] = (-1 if (ch_eye == 'L') else
                  1 if (ch_eye == 'R') else
                  np.nan)

        if (ch_xy not in ['X', 'Y']) and (ch_type == 'GAZE'):
            Warning("couldn't determine X/Y for channel {}."
                    "defaults to 'NaN'.".format(ch_name))
        loc[4] = (-1 if (ch_xy == 'X') else
                  1 if (ch_xy == 'Y') else
                  np.nan)
        info['chs'][i_ch]['loc'] = loc.copy()

    return info
