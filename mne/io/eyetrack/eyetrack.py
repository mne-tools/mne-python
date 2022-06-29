# Author: Dominik Welke <dominik.welke@web.de>
#
# License: BSD-3-Clause

from ..base import BaseRaw
from ..meas_info import create_info

from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, warn


@fill_doc
def read_raw_eyelink(fname, interpolate_missing=True, preload=False, verbose=None):
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
    return RawEyelink(fname, interpolate_missing, preload, verbose)


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
    def __init__(self, fname, interpolate_missing=True, preload=False, verbose=None):
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
                    interpolate_missing=interpolate_missing)
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

    def _parse_eyelink_asc(self, fname, sfreq, eye='BINO', pos=True,
                           pupil=True, interpolate_missing=True):
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
        ch_types = ['misc'] * n_chan  # ch_types = ['eyetrack'] * n_chan

        info = create_info(ch_names, sfreq, ch_types)

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

        # interpolate missing samples for gaze position and pupil data
        # this has to happen before making the data continuous
        if interpolate_missing:
            for k in ch_names:
                df_samples[k] = df_samples[k].interpolate()
        # section needs work! annotate interpolated segments as bad?
        # make it optional?

        # transpose to correct sfreq
        # samples = df_samples['tSample'].apply(lambda x: x*sfreq/1000.)
        # first_sample = samples.min()
        shiftfun = lambda a, b: a + b - (a % b) if (a % b != 0) else a

        sinterval = int(1000 / sfreq)  # d btw samples in ms
        #df_samples['tSample'] = df_samples['tSample'].apply(
        #    lambda x: x + sinterval - (x % sinterval) if (x % sinterval != 0)
        #    else x)  # shift samples forward
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
        onset = list((df_msg['time'] - tmin) / 1000)  # data comes in ms
        duration = [0] * len(onset)
        description = list(df_msg['text'])
        annot = Annotations(onset, duration, description,
                            orig_time=None, ch_names=None)

        first_sample = int(tmin / sfreq)  # good enough

        return info, data, first_sample, annot, meas_date
