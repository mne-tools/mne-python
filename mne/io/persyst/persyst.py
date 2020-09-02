# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import os
import os.path as op
import time
from datetime import datetime, timezone

import numpy as np

from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import create_info
from ...annotations import Annotations
from ...utils import logger, verbose, fill_doc, warn

# number of bytes for 16 and 32 bit binary .dat file
_fmt_byte_dict = dict(short=2, long=4)


@fill_doc
def read_raw_persyst(fname, preload=False, verbose=None):
    """Reader for a Persyst (.lay/.dat) recording.

    Parameters
    ----------
    fname : str
        Path to the Persyst header (.lay) file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawPersyst
        A Raw object containing Persyst data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawPersyst(fname, preload, verbose)


def _open(fname):
    return open(fname, 'r', encoding='latin-1')


@fill_doc
class RawPersyst(BaseRaw):
    """Raw object from a Persyst file.

    Parameters
    ----------
    fname : str
        Path to the Persyst header (.lay) file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        logger.info('Loading %s' % fname)

        if not fname.endswith('.lay'):
            fname = fname + '.lay'
        curr_path, lay_fname = op.dirname(fname), op.basename(fname)
        if not op.exists(fname):
            raise FileNotFoundError(f'The path you specified, '
                                    f'"{lay_fname}",does not exist.')

        # takes ~8 min for a 1.5GB file
        start_time = time.time()

        # sections and subsections currently unused
        keys, data, sections, subsections = _read_lay_contents(fname)

        # these are the section headers in the Persyst file layout
        fileinfo_dict = dict()
        channelmap_dict = dict()
        sampletimes_dict = dict()
        patient_dict = dict()
        comments_dict = dict()

        # loop through each line in the lay file
        for key, val, section, subsection in \
                zip(keys, data, sections, subsections):
            if key == '':
                continue

            # make sure key are lowercase
            if key is not None:
                key = key.lower()

            # FileInfo
            if section == 'fileinfo':
                # extract the .dat file name
                if key == 'file':
                    dat_fname = val
                    dat_path = op.dirname(dat_fname)
                    dat_fpath = op.join(curr_path, op.basename(dat_fname))

                    # determine if .dat file exists where it should
                    error_msg = f'The data path you specified ' \
                                f'does not exist for the lay path, {lay_fname}'
                    if op.isabs(dat_path) and not op.exists(dat_fname):
                        raise FileNotFoundError(error_msg)
                    if not op.exists(dat_fpath):
                        raise FileNotFoundError(error_msg)
                fileinfo_dict[key] = val
            # ChannelMap
            elif section == 'channelmap':
                # channel map has <channel_name>=<number> for <key>=<val>
                channelmap_dict[key] = val
            # SampleTimes (optional)
            elif section == 'sampletimes':
                sampletimes_dict[key] = val
            # Patient (All optional)
            elif section == 'patient':
                patient_dict[key] = val
            elif section == 'comments':
                comments_dict[key] = val

        # get numerical metadata
        # datatype is either 7 for 32 bit, or 0 for 16 bit
        datatype = fileinfo_dict.get('datatype')
        cal = float(fileinfo_dict.get('calibration'))
        n_chs = int(fileinfo_dict.get('waveformcount'))

        # Store subject information from lay file in mne format
        # Note: Persyst also records "Physician", "Technician",
        #       "Medications", "History", and "Comments1" and "Comments2"
        #       and this information is currently discarded
        subject_info = _get_subjectinfo(patient_dict)

        # set measurement date
        testdate = patient_dict.get('testdate')
        if testdate is not None:
            if '/' in testdate:
                time.strptime()
                testdate = datetime.strptime(testdate, '%m/%d/%Y')
            elif '-' in testdate:
                testdate = datetime.strptime(testdate, '%d-%m-%Y')
            elif '.' in testdate:
                testdate = datetime.strptime(testdate, '%Y.%m.%d')

            # TODO: Persyst may change its internal date schemas
            #  without notice
            if not isinstance(testdate, datetime):
                warn('Cannot read in the measurement date due '
                     'to incompatible format. Please set manually '
                     'for %s ' % lay_fname)
            else:
                testtime = datetime.strptime(patient_dict.get('testtime'),
                                             '%H:%M:%S')
                meas_date = datetime(
                    year=testdate.year, month=testdate.month,
                    day=testdate.day, hour=testtime.hour,
                    minute=testtime.minute, second=testtime.second,
                    tzinfo=timezone.utc)

        # Create mne structure
        ch_names = list(channelmap_dict.keys())
        if n_chs != len(ch_names):
            raise RuntimeError('Channels in lay file do not '
                               'match the number of channels '
                               'in the .dat file.')  # noqa
        # get rid of the "-Ref" in channel names
        ch_names = [ch.split('-ref')[0] for ch in ch_names]

        # get the sampling rate and default channel types to EEG
        sfreq = fileinfo_dict.get('samplingrate')
        ch_types = 'eeg'
        info = create_info(ch_names, sfreq, ch_types=ch_types)
        info.update(subject_info=subject_info)
        for idx in range(n_chs):
            info['chs'][idx]['cal'] = cal
        info['meas_date'] = meas_date

        # determine number of samples in file
        # Note: We do not use the lay file to do this
        # because clips in time may be generated by Persyst that
        # DO NOT modify the "SampleTimes" section
        with open(dat_fpath, 'rb') as f:
            # determine the precision
            if int(datatype) == 7:
                # 32 bit
                dtype_bytes = _fmt_byte_dict['long']
            elif int(datatype) == 0:
                # 16 bit
                dtype_bytes = _fmt_byte_dict['short']

            # allow offset to occur
            f.seek(0, os.SEEK_END)
            n_samples = f.tell()
            n_samples = n_samples // (dtype_bytes * n_chs)

            if verbose:
                print(f'Loaded {n_samples} samples '
                      f'for {n_chs} channels.')

        raw_extras = {
            'datatype': datatype,
            'n_chs': n_chs,
            'n_samples': n_samples
        }
        # create Raw object
        super(RawPersyst, self).__init__(
            info, preload, filenames=[dat_fpath],
            last_samps=[n_samples - 1],
            raw_extras=[raw_extras], verbose=verbose)

        # set annotations based on the comments read in
        num_comments = len(comments_dict.keys())
        onset = np.zeros(num_comments, float)
        duration = np.zeros(num_comments, float)
        description = [''] * num_comments
        for t_idx, (_description, (_onset, _duration)) in \
                enumerate(comments_dict.items()):
            # extract the onset, duration, description to
            # create an Annotations object
            onset[t_idx] = _onset
            duration[t_idx] = _duration
            description[t_idx] = _description
        annot = Annotations(onset, duration, description)
        self.set_annotations(annot)

        # elapsed time (in min)
        elapsed = (time.time() - start_time) / 60
        logger.info('Read in file in %s minutes.' % elapsed)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        The Persyst software  records raw data in either 16 or 32 bit
        binary files. In addition, it stores the calibration to convert
        data to uV in the lay file.
        """
        datatype = self._raw_extras[fi]['datatype']
        n_chs = self._raw_extras[fi]['n_chs']
        dat_fname = self._filenames[fi]

        # compute samples count based on start and stop
        time_length_samps = stop - start

        # read data from .dat file into array of correct size, then calibrate
        # records = recnum rows x inf columns
        if time_length_samps == -1:
            count = -1  # elements of size precision to read
        else:
            count = time_length_samps * n_chs

        # seek the dat file
        # dat file
        with open(dat_fname, 'rb') as dat_file_ID:
            # allow offset to occur
            if int(datatype) == 7:
                precision = np.int32
                dat_file_ID.seek(n_chs * 4 * start, 1)
            else:
                precision = np.int16
                dat_file_ID.seek(n_chs * 2 * start, 1)

            # read in the actual record starting at possibly offset
            record = np.fromfile(dat_file_ID, dtype=precision,
                                 count=count)

        # chs * rows
        record = np.reshape(record, (n_chs, -1), 'F')
        # calibrate to convert to uV
        data[...] = record[idx, ...] * cals
        # cast as float32; more than enough precision
        # then multiply to get V
        data = data.astype(np.float32) / 10. ** 6


def _get_subjectinfo(patient_dict):
    # attempt to parse out the birthdate, but if it doesn't
    # meet spec, then it will set to None
    birthdate = patient_dict.get('birthdate')
    if '/' in birthdate:
        try:
            birthdate = datetime.strptime(birthdate, '%m/%d/%y')
        except ValueError:
            birthdate = None
            print('Unable to process birthdate of %s ' % birthdate)
    elif '-' in birthdate:
        try:
            birthdate = datetime.strptime(birthdate, '%d-%m-%y')
        except ValueError:
            birthdate = None
            print('Unable to process birthdate of %s ' % birthdate)

    subject_info = {
        'first_name': patient_dict.get('first'),
        'middle_name': patient_dict.get('middle'),
        'last_name': patient_dict.get('last'),
        'sex': patient_dict.get('sex'),
        'hand': patient_dict.get('hand'),
        'his_id': patient_dict.get('id'),
        'birthday': birthdate,
    }

    # Recode sex values
    if subject_info['sex'] in ('m', 'male'):
        subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
    elif subject_info['sex'] in ('f', 'female'):
        subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
    else:
        subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_UNKNOWN
    # Recode hand values
    if subject_info['hand'] in ('m', 'male'):
        subject_info['hand'] = FIFF.FIFFV_SUBJ_HAND_RIGHT
    elif subject_info['hand'] in ('f', 'female'):
        subject_info['hand'] = FIFF.FIFFV_SUBJ_HAND_LEFT
    elif subject_info['hand'] in ('a', 'ambidextrous'):
        subject_info['hand'] = FIFF.FIFFV_SUBJ_HAND_AMBI
    else:
        # no handedness is set when unknown
        subject_info.pop('hand')

    return subject_info


def _read_lay_contents(fname):
    """Lay file are laid out like a INI file."""
    # keep track of sections, subsections, keys and data
    sections = []
    subsections = []
    keys, data = [], []

    # initialize all section/subsections to empty str
    section, subsection = '', ''
    with open(fname, 'r') as fin:
        for line in fin:
            # break a line into a status, key and value
            status, key, val = _process_lay_line(line, section)

            # handle keys and values if they are
            # Section, Subsections, or Line items
            if status == 1:  # Section was found
                section = val.lower()
                continue
            elif status == 2:  # Subsection was found
                subsection = val.lower()
                continue

            # keep track of all sections, subsections,
            # keys and the data of the file
            sections.append(section)
            subsections.append(subsection)
            data.append(val)
            keys.append(key)
    return keys, data, sections, subsections


def _process_lay_line(line, section):
    """Process a line read from the Lay (INI) file.

    Each line in the .lay file will be processed
    into a structured ``status``, ``key`` and ``value``.

    Parameters
    ----------
    line : str
        The actual line in the Lay file.

    Returns
    -------
    status : int
        Returns the following integers based on status.
        -1  => unknown string found
        0   => empty line found
        1   => section found
        2   => subsection found
        3   => key-value pair found
        4   => comment line found (starting with ;)
    key : str
        The string before the ``'='`` character. If section is "Comments",
        then returns the text comment description.
    value : str
        The string from the line after the ``'='`` character. If section is
        "Comments", then returns the onset and duration as a tuple.
    """
    key = ''  # default; only return value possibly not set
    line = line.strip()  # remove leading and trailing spaces
    end_idx = len(line) - 1  # get the last index of the line

    # empty sequence evaluates to false
    if not line:
        status = 0
        key = ''
        value = ''
        return status, key, value
    # comment found
    elif line[0] == ';':
        status = 4
        value = line[1:end_idx + 1]
    # section found
    elif (line[0] == '[') and (line[end_idx] == ']') \
            and (end_idx + 1 >= 3):
        status = 1
        value = line[1:end_idx].lower()
    # subsection found
    elif (line[0] == '{') and (line[end_idx] == '}') \
            and (end_idx + 1 >= 3):
        status = 2
        value = line[1:end_idx].lower()
    # key found
    else:
        # handle Comments section differently from all other sections
        # TODO: utilize state and var_type in code.
        #  Currently not used
        if section == 'comments':
            # Persyst Comments output 5 variables "," separated
            time_sec, duration, state, var_type, text = line.split(',')
            status = 3
            key = text
            value = (time_sec, duration)
        # all other sections
        else:
            if '=' not in line:
                raise RuntimeError('The line %s does not conform '
                                   'to the standards. Please check the '
                                   '.lay file.' % line)  # noqa
            pos = line.index('=')
            status = 3

            # the line now is composed of a
            # <key>=<value>
            key = line[0:pos].lower()
            key.strip()
            if not key:
                status = -1
                key = ''
                value = ''
            else:
                value = line[pos + 1:end_idx + 1].lower()
                value.strip()
    return status, key, value
