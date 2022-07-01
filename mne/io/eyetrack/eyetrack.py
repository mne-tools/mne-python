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
        loc = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for i_ch, ch_name in enumerate(ch_names):
            coil_type = (FIFF.FIFFV_COIL_EYETRACK_PUPIL if ('Pupil' in ch_name)
                         else FIFF.FIFFV_COIL_EYETRACK_POS)

            loc[3] = (-1 if ('L' in ch_name) else
                      1 if ('R' in ch_name) else
                      np.nan)
            loc[4] = (-1 if ('X' in ch_name) else
                      1 if ('Y' in ch_name) else
                      np.nan)
            info['chs'][i_ch]['coil_type'] = coil_type
            info['chs'][i_ch]['loc'] = loc.copy()

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
