# Author: Dominik Welke <dominik.welke@web.de>
#
# License: BSD-3-Clause

from ..base import BaseRaw
from ..meas_info import create_info

from ...utils import logger, verbose, fill_doc

@fill_doc
def read_raw_eyelink(fname, preload=False, verbose=None):
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
    return RawEyelink(fname, preload, verbose)


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
    def __init__(self, fname, preload=False, verbose=None):
        logger.info('Loading {}'.format(fname))

        # load data
        ftype = fname.split('.')[-1]
        if ftype == 'asc':
            sfreq = 1000
            eye = 'BINO'
            pos = True
            pupil = True

            info, data, first_sample = self._parse_eyelink_asc(
                fname,
                sfreq=sfreq,
                eye=eye,
                pos=pos,
                pupil=pupil)
        elif ftype == 'edf':
            raise NotImplementedError('Eyelink .edf files not supported, yet')
        else:
            raise ValueError(
                'This reader can only read eyelink .asc (or .edf) files. '
                'Got .{} instead'.format(ftype))

        # create mne object
        super(RawEyelink, self).__init__(  # or just super().__init__( ?
            info, preload=data, filenames=[fname], verbose=verbose)

    def _parse_eyelink_asc(self, fname, sfreq=1000., eye='BINO', pos=True,
                           pupil=True):
        from .ParseEyeLinkAscFiles_ import ParseEyeLinkAsc_

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

        # transpose to correct sfreq
        #samples = df_samples['tSample'].apply(lambda x: x*sfreq/1000.)
        #first_sample = samples.min()

        # clean out misread data
        # tSample > 0
        df_samples = df_samples[df_samples.tSample > 0]
        # also clean out rows where number of nans doesnt fit!
        # tbd

        # fix epoched recording by making it contiuous
        df_samples['tSample'] = df_samples['tSample'].astype(int)
        df_samples.index = df_samples.tSample

        tmin = df_msg['time'].min()
        tmax = df_samples.tail()['tSample'].max()

        samples_new = list(range(tmin, tmax+1))
        df_samples = df_samples.reindex(samples_new)

        # get data for selected channels
        try:
            data = df_samples[ch_names].to_numpy().T
        except KeyError:
            raise ValueError(
                "provided eye={} parameter doesn't match the data".format(eye))

        return info, data, first_sample
