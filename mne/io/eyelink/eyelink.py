# Authors: Dominik Welke <dominik.welke@web.de>
#          Scott Huberty <seh33@uw.edu>
#          Christian O'Reilly <christian.oreilly@sc.edu>
#
# License: BSD-3-Clause

from pathlib import Path

import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, _check_pandas_installed

EYELINK_COLS = {'timestamp': ('time',),
                'gaze': {'monocular': ('x', 'y', 'pupil'),
                         'binocular': ('x_left', 'y_left', 'pupil_left',
                                       'x_right', 'y_right', 'pupil_right')},
                'velocity': {'monocular': ('x_vel', 'y_vel'),
                             'binocular': ('x_vel_left', 'y_vel_left',
                                           'x_vel_right', 'y_vel_right')},
                'resolution': ('x_res', 'y_res'),
                'input': ('DIN',),
                'flags': ('flags',),
                'remote': ('head_target_x', 'head_target_y',
                           'head_target_distance'),
                'remote_flags': ('head_target_flags',),
                'block_num': ('block',),
                'eye_event': ('eye', 'time', 'end_time', 'duration'),
                'fixation': ('fix_avg_x', 'fix_avg_y',
                             'fix_avg_pupil_size'),
                'saccade': ('sacc_start_x', 'sacc_start_y',
                            'sacc_end_x', 'sacc_end_y',
                            'sacc_visual_angle', 'peak_velocity')}


def _isfloat(token):
    '''boolean test for whether string can be of type float.

       token (str): single element from tokens list'''

    if isinstance(token, str):
        try:
            float(token)
            return True
        except ValueError:
            return False
    else:
        raise ValueError('input should be a string,'
                         f' but {token} is of type {type(token)}')


def _convert_types(tokens):
    """Converts the type of each token in list, which are read in as strings.
       Posix timestamp strings can be integers, eye gaze position and
       pupil size can be floats. flags token ("...") remains as string.
       Missing eye/head-target data (indicated by '.' or 'MISSING_DATA')
       are replaced by np.nan.

       Parameters
       ----------
       tokens (list): list of string elements.

       returns: tokens list with elements of various types."""

    return [int(token) if token.isdigit()  # execute this before _isfloat()
            else float(token) if _isfloat(token)
            else np.nan if token in ('.', 'MISSING_DATA')
            else token  # remains as string
            for token in tokens]


def _parse_line(line):
    """takes a tab deliminited string from eyelink file,
       splits it into a list of tokens, and converts the type
       for each token in the list"""

    if len(line):
        tokens = line.split()
        return _convert_types(tokens)
    else:
        raise ValueError('line is empty, nothing to parse')


def _is_sys_msg(line):
    """Some lines in eyelink files are system outputs usually
       only meant for Eyelinks DataViewer application to read.
       These shouldn't need to be parsed.

       Parameters
       ----------
       line (string): single line from Eyelink asc file

       Returns: True if any of the following strings that are
       known to indicate a system message are in the line"""

    return any(['!V' in line,
                '!MODE' in line,
                ';' in line])


def _get_sfreq(rec_info):
    """
       rec_info (list):
           the first list in self._event_lines['SAMPLES'].
           The sfreq occurs after RATE: i.e. [..., RATE, 1000, ...].

        returns: sfreq
    """
    for i, token in enumerate(rec_info):
        if token == 'RATE':
            # sfreq is the first token after RATE
            return rec_info[i + 1]


def _sort_by_time(df, col='time'):
    df.sort_values(col, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)


def _convert_times(df, first_samp, col='time'):
    """Set initial time to 0, converts from ms to seconds in place.
       Each sample in an Eyelink file has a posix timestamp string.
       Subtracts the "first" sample's timestamp from each timestamp.
       The "first" sample is inferred to be the first sample of
       the first recording block, i.e. the first "START" line.

       df pandas.DataFrame:
           One of the dataframes in the self.dataframes dict.

       first_samp int:
           timestamp of the first sample of the recording. This should
           be the first sample of the first recording block.
        col str (default 'time'):
            column name to sort pandas.DataFrame by"""

    _sort_by_time(df, col)

    for col in df.columns:
        if col.endswith('time'):  # 'time' and 'end_time' cols
            df[col] -= first_samp
            df[col] /= 1000
        if col in ['duration', 'offset']:
            df[col] /= 1000


def _fill_times(df, sfreq, time_col='time',):
    """Fills missing timestamps in cases where there are multiple
       recording blocks, which cause missing samples from the
       gap periods between the blocks.

        parameters
        ---------
        df : pandas.DataFrame:
           dataframe of the eyetracking data samples, BEFORE
           _convert_times() is applied to the dataframe

        sfreq : int, float:
           sampling frequency of the data

        time_col : str (default 'time'):
           name of column with the timestamps (e.g. 9511881, 9511882, ...)

        returns
        -------
        pandas DataFrame with previously missing timestamps included
        """
    pd = _check_pandas_installed()

    first = df[time_col].iloc[0]
    last = df[time_col].iloc[-1]
    step = 1000 / sfreq
    times = pd.DataFrame(np.arange(first, last + step, step),
                         columns=[time_col])
    times[time_col] = times[time_col].astype(int)
    return pd.merge(times, df, on=time_col, how='left')


def _find_overlaps(df, max_time=0.05):
    """
    Combine left and right eye events if their start times and their stop times
    are both not separated by more than `max_time`.

    df : pandas.DataFrames:
        Pandas DataFrame with occular events (fixations, saccades, blinks)
    max_time : float (default value 0.05)
        Time in seconds.

    Returns:
    --------
    DataFrame: Instance of a Pandas DataFrame
        DataFrame specifying overlapped eye events, if any
    Notes
    -----
    The idea is to cumulative sum the boolean values for rows with start/end
    time differences (against the previous row) that are greater than the
    max_time. If start and end diffs are less than max_time then no_overlap
    will become False. Alternatively, if either the start or end diff is
    greater than max_time, no_overlap becomes True. Cumulatively summing over
    these boolean values will leave rows with no_overlap == False unchanged
    and hence the same group number.
    """
    pd = _check_pandas_installed()

    df = df.copy()
    df["overlap_start"] = df.sort_values("time")["time"]\
                            .diff()\
                            .lt(max_time)

    df["overlap_end"] = (df["end_time"]
                         .diff().abs()
                         .lt(max_time))

    df["no_overlap"] = ~(df["overlap_end"]
                         & df["overlap_start"])
    df["group"] = df["no_overlap"].cumsum()

    # now use groupby on `'group'`. If one left and one right eye in group
    # the new start/end times are the mean of the two eyes
    ovrlp = pd.concat([pd.DataFrame(g[1].drop(columns="eye").mean()).T
                       if (len(g[1]) == 2) and (len(g[1].eye.unique()) == 2)
                       else g[1]  # not an overlap, return group unchanged
                       for g in df.groupby("group")]
                      )
    # overlapped events get a "both" value in the "eye" col
    if "eye" in ovrlp.columns:
        ovrlp["eye"] = ovrlp["eye"].fillna("both")
    else:
        ovrlp["eye"] = "both"
    tmp_cols = ["overlap_start", "overlap_end", "no_overlap", "group"]
    return ovrlp.drop(columns=tmp_cols).reset_index(drop=True)


@fill_doc
def read_raw_eyelink(fname, preload=False, verbose=None,
                     create_annotations=True, apply_offsets=False,
                     find_overlaps=False, overlap_threshold=0.05):
    """Reader for an Eyelink .asc file.

    Parameters
    ----------
    fname : str
        Path to the eyelink file (.asc).
    create_annotations : boolean or list (Default True)
        Whether to create mne.Annotations from occular events
        (blinks, fixations, saccades) and experiment messages. If a list, must
        contain one or more of ['fixations', 'saccades',' blinks', messages'].
        If True, creates mne.Annotations for both occular events and experiment
        messages.
    apply_offsets : boolean (Default False)
        Adjusts the onset time of the mne.Annotations created from Eyelink
        experiment messages, if offset values exist in
        self.dataframes['messages'].
    find_overlaps : boolean (Default False)
        Combine left and right eye `Annotations` (blinks, fixations, saccades)
        if their start times and their stop times are both not separated by
        more than `overlap_threshold`.
    overlap_threshold : float (Default 0.05)
        Time in seconds. Threshold of allowable time-gap between the start and
        stop times of the left and right eyes. If gap is larger than threshold,
        the `Annotations` will be kept separate (i.e. "blink_L", "blink_R"). If
        the gap is smaller than the threshold, the `Annotations` will be merged
        (i.e. "blink_both")
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

    extension = Path(fname).suffix
    if extension not in '.asc':
        raise ValueError('This reader can only read eyelink .asc files.'
                         f' Got extension {extension} instead. consult eyelink'
                         ' manual for converting eyelink data format (.edf)'
                         ' files to .asc format.')

    return RawEyelink(fname, preload=preload, verbose=verbose,
                      create_annotations=create_annotations,
                      apply_offsets=apply_offsets,
                      find_overlaps=find_overlaps,
                      overlap_threshold=overlap_threshold)


@fill_doc
class RawEyelink(BaseRaw):
    """Raw object from an XXX file.

    Parameters
    ----------
    fname : str
        Path to the data file (.XXX).
    create_annotations : boolean or list (Default True)
        Whether to create mne.Annotations from occular events
        (blinks, fixations, saccades) and experiment messages. If a list, must
        contain one or more of ['fixations', 'saccades',' blinks', messages'].
        If True, creates mne.Annotations for both occular events and experiment
        messages.
    apply_offsets : boolean (Default False)
        Adjusts the onset time of the mne.Annotations created from Eyelink
        experiment messages, if offset values exist in
        `raw.dataframes['messages']`.
     find_overlaps : boolean (Default False)
        Combine left and right eye `Annotations` (blinks, fixations, saccades)
        if their start times and their stop times are both not separated by
        more than `overlap_threshold`.
    overlap_threshold : float (Default 0.05)
        Time in seconds. Threshold of allowable time-gap between the start and
        stop times of the left and right eyes. If gap is larger than threshold,
        the `Annotations` will be kept separate (i.e. "blink_L", "blink_R"). If
        the gap is smaller than the threshold, the `Annotations` will be merged
        (i.e. "blink")
    %(preload)s
    %(verbose)s

    Attributes
    ----------
    fname (pathlib.Path object):
        Eyelink filename
    dataframes (dictionary):
        Dictionary of pandas DataFrames. One for eyetracking samples,
        and one for each type of eyelink event (blinks, messages, etc)
    _sample_lines (list):
        List of lists, each list is one sample containing eyetracking
        X/Y and pupil channel data (+ other channels, if they exist)
    _event_lines (dict):
        Each key contains a list of lists, for an event-type that occurred
        during the recording period. Events can vary, from occular events
        (blinks, saccades, fixations), to messages from the stimulus
        presentation software, or info from a response controller.
    _system_lines (list):
        List of space delimited strings. Each string is a system message,
        that in most cases aren't needed. System messages occur for
        Eyelinks DataViewer application.
    _tracking_mode (str):
        Whether whether a single eye was tracked ('monocular'), or both
        ('binocular').

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None,
                 create_annotations=True,
                 apply_offsets=False, find_overlaps=False,
                 overlap_threshold=0.05):

        logger.info('Loading {}'.format(fname))

        self.fname = Path(fname)
        self._sample_lines = None
        self._event_lines = None
        self._system_lines = None
        self._tracking_mode = None  # assigned in self._infer_col_names
        self.dataframes = {}

        self._parse_recording_blocks()  # sets sample, event, & system lines

        sfreq = _get_sfreq(self._event_lines['SAMPLES'][0])
        col_names, ch_names = self._infer_col_names()
        self._create_dataframes(col_names, sfreq, find_overlaps=find_overlaps,
                                threshold=overlap_threshold)
        info = self._create_info(ch_names, sfreq)
        eye_ch_data = self.dataframes['samples'][ch_names]
        eye_ch_data = eye_ch_data.to_numpy().T

        # create mne object
        super(RawEyelink, self).__init__(info, preload=eye_ch_data,
                                         filenames=[self.fname],
                                         verbose=verbose)

        if create_annotations:
            annots = self._make_eyelink_annots(self.dataframes,
                                               create_annotations,
                                               apply_offsets)
            self.set_annotations(annots)

    def _parse_recording_blocks(self):
        '''Eyelink samples occur within START and END blocks.
           samples lines start with a posix-like string,
           and contain eyetracking sample info. Event Lines
           start with an upper case string and contain info
           about occular events (i.e. blink/saccade), or experiment
           messages sent by the stimulus presentation software.'''

        with self.fname.open() as file:
            block_num = 1
            self._sample_lines = []
            self._event_lines = {'START': [], 'END': [], 'SAMPLES': [],
                                 'EVENTS': [], 'ESACC': [], 'EBLINK': [],
                                 'EFIX': [], 'MSG': [], 'INPUT': [],
                                 'BUTTON': []}
            self._system_lines = []

            is_recording_block = False
            for line in file:
                if line.startswith('START'):  # start of recording block
                    is_recording_block = True
                if is_recording_block:
                    if _is_sys_msg(line):
                        self._system_lines.append(line)
                        continue  # system messages don't need to be parsed.
                    tokens = _parse_line(line)
                    tokens.append(block_num)  # add current block number
                    if isinstance(tokens[0], (int, float)):  # Samples
                        self._sample_lines.append(tokens)
                    elif tokens[0] in self._event_lines.keys():
                        event_key, event_info = tokens[0], tokens[1:]
                        self._event_lines[event_key].append(event_info)
                    if tokens[0] == 'END':  # end of recording block
                        is_recording_block = False
                        block_num += 1
            if not self._sample_lines:  # no samples parsed
                raise ValueError(f"Couldn't find any samples in {self.fname}")

    def _infer_col_names(self):
        """Returns the expected column names for the sample lines and event
           lines, to be passed into pd.DataFrame. Sample and event lines in
           eyelink files have a fixed order of columns, but the columns that
           are present can vary. The order that col_names is built below should
           NOT change."""

        col_names = {}

        # a list of keywords specifying what type of data is present
        rec_info = self._event_lines['SAMPLES'][0]

        # initiate the column names for the sample lines
        col_names['sample'] = list(EYELINK_COLS['timestamp'])

        # and for the eye message lines
        col_names['blink'] = list(EYELINK_COLS['eye_event'])
        col_names['fixation'] = list(EYELINK_COLS['eye_event'] +
                                     EYELINK_COLS['fixation'])
        col_names['saccade'] = list(EYELINK_COLS['eye_event'] +
                                    EYELINK_COLS['saccade'])

        # Recording was either binocular or monocular
        eyes_tracked = ('binocular' if
                        ('LEFT' in rec_info) and ('RIGHT' in rec_info)
                        else 'monocular')
        self._tracking_mode = eyes_tracked
        # If monocular, find out which eye was tracked and append to ch_name
        if self._tracking_mode == 'monocular':
            assert rec_info[1] in ['LEFT', 'RIGHT']
            mono_eye = rec_info[1].lower()

        ch_names = list(EYELINK_COLS['gaze'][eyes_tracked])
        if self._tracking_mode == 'monocular':
            ch_names = [f'{name}_{mono_eye}'
                        for name in ch_names]  # x_left, ... y_right
        col_names['sample'].extend(ch_names)

        # The order of these if statements should not be changed.
        if 'VEL' in rec_info:  # If velocity data are reported
            ch_names.extend(EYELINK_COLS['velocity'][eyes_tracked])
            col_names['sample'].extend(EYELINK_COLS['velocity'][eyes_tracked])
        # if resolution data are reported
        if 'RES' in rec_info:
            ch_names.extend(EYELINK_COLS['resolution'])
            col_names['sample'].extend(EYELINK_COLS['resolution'])
            col_names['fixation'].extend(EYELINK_COLS['resolution'])
            col_names['saccade'].extend(EYELINK_COLS['resolution'])
        # if digital input port values are reported
        if 'INPUT' in rec_info:
            ch_names.extend(EYELINK_COLS['input'])
            col_names['sample'].extend(EYELINK_COLS['input'])

        # add flags column
        col_names['sample'].extend(EYELINK_COLS['flags'])

        # if head target info was reported, add its cols after flags col.
        if 'HTARGET' in rec_info:
            ch_names.extend(EYELINK_COLS['remote'])
            col_names['sample'].extend(EYELINK_COLS['remote']
                                       + EYELINK_COLS['remote_flags'])

        # finally add a column for recording block number
        # FYI this column does not exist in the asc file..
        # but it is added during _parse_recording_blocks
        for col in col_names.values():
            col.extend(EYELINK_COLS['block_num'])

        return col_names, ch_names

    def _create_dataframes(self, col_names, sfreq, find_overlaps=False,
                           threshold=0.05):
        """creates a pandas DataFrame for self._sample_lines and for each
           non-empty key in self._event_lines"""

        pd = _check_pandas_installed()

        # First sample should be the first line of the first recording block
        first_samp = self._event_lines['START'][0][0]

        # dataframe for samples
        self.dataframes['samples'] = pd.DataFrame(self._sample_lines,
                                                  columns=col_names['sample'])

        if len(self._event_lines['START']) > 1:
            logger.debug('There is more than one recording block in this'
                         ' file. Accounting for times between the blocks')
            # if there is more than 1 recording block we must account for
            # the missing timestamps and samples bt the blocks
            self.dataframes['samples'] = _fill_times(self.dataframes
                                                     ['samples'],
                                                     sfreq=sfreq)
        _convert_times(self.dataframes['samples'], first_samp)

        # dataframe for each type of occular event
        for event, columns, label in zip(['EFIX', 'ESACC', 'EBLINK'],
                                         [col_names['fixation'],
                                          col_names['saccade'],
                                          col_names['blink']],
                                         ['fixations',
                                          'saccades',
                                          'blinks']
                                         ):
            if self._event_lines[event]:  # an empty list returns False
                self.dataframes[label] = pd.DataFrame(self._event_lines[event],
                                                      columns=columns)
                _convert_times(self.dataframes[label], first_samp)

                if find_overlaps is True:
                    if self._tracking_mode == 'monocular':
                        raise ValueError('`find_overlaps` is only valid with'
                                         ' binocular recordings, this file is'
                                         f' {self._tracking_mode}')
                    df = _find_overlaps(self.dataframes[label],
                                        max_time=threshold)
                    self.dataframes[label] = df

            else:
                logger.info(f'No {label} were found in this file. '
                            f'Not returning any info on {label}')

        # make dataframe for experiment messages
        if self._event_lines['MSG']:
            msgs = []
            for tokens in self._event_lines['MSG']:
                timestamp = tokens[0]
                block = tokens[-1]
                # if offset token exists, it will be the 1st index
                # and is an int or float
                if isinstance(tokens[1], (int, float)):
                    offset = tokens[1]
                    msg = ' '.join(str(x) for x in tokens[2:-1])
                else:
                    # there is no offset token
                    offset = np.nan
                    msg = ' '.join(str(x) for x in tokens[1:-1])
                msgs.append([timestamp, offset, msg, block])

            cols = ['time', 'offset', 'event_msg', 'block']
            self.dataframes['messages'] = (pd.DataFrame(msgs,
                                                        columns=cols))
            _convert_times(self.dataframes['messages'], first_samp)

        # make dataframe for recording block start, end times
        assert (len(self._event_lines['START'])
                == len(self._event_lines['END'])
                )
        blocks = [[bgn[0], end[0], bgn[-1]]  # start, end, block_num
                  for bgn, end in zip(self._event_lines['START'],
                                      self._event_lines['END'])
                  ]
        cols = ['time', 'end_time', 'block']
        self.dataframes['recording_blocks'] = pd.DataFrame(blocks,
                                                           columns=cols)
        _convert_times(self.dataframes['recording_blocks'], first_samp)

        # make dataframe for digital input port
        if self._event_lines['INPUT']:
            cols = ['time', 'DIN', 'block']
            self.dataframes['DINS'] = pd.DataFrame(self._event_lines['INPUT'],
                                                   columns=cols)
            _convert_times(self.dataframes['DINS'], first_samp)

        # TODO: Make dataframes for other eyelink events (Buttons)

    def _create_info(self, ch_names, sfreq):
        # assign channel type from ch_name
        pos_names = ('x_left', 'x_right', 'y_left', 'y_right')
        pupil_names = ('pupil_left', 'pupil_right')
        ch_types = ['eyetrack_pos' if ch in pos_names
                    else 'eyetrack_pupil' if ch in pupil_names
                    else 'stim' if ch == 'DIN'
                    else 'misc'
                    for ch in ch_names]
        info = create_info(ch_names,
                           sfreq,
                           ch_types)
        # set correct loc for eyepos and pupil channels
        for ch_dict in info['chs']:
            # loc index 3 can indicate left or right eye
            if ch_dict['ch_name'].endswith('left'):  # [x,y,pupil]_left
                ch_dict['loc'][3] = -1  # left eye
            elif ch_dict['ch_name'].endswith('right'):  # [x,y,pupil]_right
                ch_dict['loc'][3] = 1  # right eye
            else:
                logger.debug(f"leaving index 3 of loc array as"
                             f" {ch_dict['loc'][3]} for {ch_dict['ch_name']}")
            # loc index 4 can indicate x/y coord
            if ch_dict['ch_name'].startswith('x'):
                ch_dict['loc'][4] = -1  # x-coord
            elif ch_dict['ch_name'].startswith('y'):
                ch_dict['loc'][4] = 1  # y-coord
            else:
                logger.debug(f"leaving index 4 of loc array as"
                             f" {ch_dict['loc'][4]} for {ch_dict['ch_name']}")
        return info

    def _make_eyelink_annots(self, df_dict, create_annots, apply_offsets):
        """Creates Annotations for each df in self.dataframes"""

        valid_descs = ['blinks', 'saccades', 'fixations', 'messages']
        msg = ("create_annotations must be `True` or a list containing one or"
               " more of ['blinks', 'saccades', 'fixations', 'messages'].")
        wrong_type = (msg + f' Got a {type(create_annots)} instead.')
        if create_annots is True:
            descs = valid_descs
        else:
            assert isinstance(create_annots, list), wrong_type
            for desc in create_annots:
                assert desc in valid_descs, msg + f" Got '{desc}' instead"
            descs = create_annots

        annots = None
        for key, df in df_dict.items():
            eye_annot_cond = ((key in ['blinks', 'fixations', 'saccades'])
                              and (key in descs))
            if eye_annot_cond:
                onsets = df['time']
                durations = df['duration']
                # Create annotations for both eyes
                descriptions = f'{key[:-1]}_' + df['eye']  # i.e "blink_r"
                this_annot = Annotations(onset=onsets,
                                         duration=durations,
                                         description=descriptions,
                                         orig_time=None)
            elif (key in ['messages']) and (key in descs):
                if apply_offsets:
                    if df['offset'].isnull().all():
                        logger.warn('There are no offsets for the messages in'
                                    f' {self.fname}. Not applying any offset')
                    # If df['offset] is all NaNs, time is not changed
                    onsets = df['time'] + df['offset'].fillna(0)
                else:
                    onsets = df['time']
                durations = [0] * onsets
                descriptions = df['event_msg']
                this_annot = Annotations(onset=onsets,
                                         duration=durations,
                                         description=descriptions)
            elif (key in ['recording_blocks']) and ('messages' in descs):
                start_onsets = df['time'].tolist()
                end_onsets = df['end_time'][:-1].tolist()
                onsets = start_onsets + end_onsets
                durations = [0] * len(onsets)
                blocks = (df['block']
                          .astype(int)
                          .astype(str)
                          .tolist())
                start_desc = ['start_block_' + num for num in blocks]
                end_desc = ['end_block_' + num for num in blocks[:-1]]
                descriptions = start_desc + end_desc
                this_annot = Annotations(onset=onsets,
                                         duration=durations,
                                         description=descriptions)
            else:
                continue  # TODO make df and annotations for Buttons
            if not annots:
                annots = this_annot
            elif annots:
                annots += this_annot
        return annots
