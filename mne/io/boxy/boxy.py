# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
#
# License: BSD (3-clause)

import glob as glob
import re as re

import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ...utils import logger, verbose, fill_doc
from ...annotations import Annotations


@fill_doc
def read_raw_boxy(fname, datatype='AC', preload=False, verbose=None):
    """Reader for a BOXY optical imaging recording.

    Parameters
    ----------
    fname : str
        Path to the BOXY data folder.
    datatype : str
        Type of data to return (AC, DC, or Ph).
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawBOXY
        A Raw object containing BOXY data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawBOXY(fname, datatype, preload, verbose)


@fill_doc
class RawBOXY(BaseRaw):
    """Raw object from a BOXY optical imaging file.

    Parameters
    ----------
    fname : str
        Path to the BOXY data folder.
    datatype : str
        Type of data to return (AC, DC, or Ph).
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, datatype='AC', preload=False, verbose=None):
        logger.info('Loading %s' % fname)

        # Check if required files exist and store names for later use.
        files = dict()
        key = '*.txt'
        print(fname)
        files[key] = [glob.glob('%s/*%s' % (fname, key))]

        # Make sure filenames are in order.
        files[key][0].sort()
        if len(files[key]) != 1:
            raise RuntimeError('Expect one %s file, got %d' %
                               (key, len(files[key]),))
        files[key] = files[key][0]

        # Determine which data type to return.
        if datatype not in ['AC', 'DC', 'Ph']:
            raise RuntimeError('Expect AC, DC, or Ph, got %s' % datatype)

        # Read header file and grab some info.
        filetype = 'parsed'
        start_line = 0
        end_line = 0
        mrk_col = 0
        mrk_data = list()
        col_names = list()
        with open(files[key][0], 'r') as data:
            for line_num, i_line in enumerate(data, 1):
                if '#DATA ENDS' in i_line:
                    # Data ends just before this.
                    end_line = line_num - 1
                    break
                if 'Detector Channels' in i_line:
                    detect_num = int(i_line.rsplit(' ')[0])
                elif 'External MUX Channels' in i_line:
                    source_num = int(i_line.rsplit(' ')[0])
                elif 'Update Rate (Hz)' in i_line:
                    srate = float(i_line.rsplit(' ')[0])
                elif 'Updata Rate (Hz)' in i_line:
                    srate = float(i_line.rsplit(' ')[0])
                elif '#DATA BEGINS' in i_line:
                    # Data should start a couple lines later.
                    start_line = line_num + 2
                if start_line > 0 & end_line == 0:
                    if line_num == start_line - 1:
                        # Grab names for each column of data.
                        col_names = np.asarray(re.findall(
                            r'\w+\-\w+|\w+\-\d+|\w+', i_line.rsplit(' ')[0]))
                        if 'exmux' in col_names:
                            # Change filetype based on data organisation.
                            filetype = 'non-parsed'
                        if 'digaux' in col_names:
                            mrk_col = np.where(col_names == 'digaux')[0][0]
                    # Need to treat parsed and non-parsed files differently.
                    elif (mrk_col > 0 and line_num > start_line and
                          filetype == 'non-parsed'):
                        # Non-parsed files have different lines lengths.
                        crnt_line = i_line.rsplit(' ')[0]
                        temp_data = re.findall(r'[-+]?\d*\.?\d+', crnt_line)
                        if len(temp_data) == len(col_names):
                            mrk_data.append(float(
                                re.findall(r'[-+]?\d*\.?\d+', crnt_line)
                                [mrk_col]))
                    elif (mrk_col > 0 and line_num > start_line
                          and filetype == 'parsed'):
                        # Parsed files have the same line lengths for data.
                        crnt_line = i_line.rsplit(' ')[0]
                        mrk_data.append(float(
                            re.findall(r'[-+]?\d*\.?\d+', crnt_line)[mrk_col]))

        # Label each channel in our data.
        # Data is organised by channels x timepoint, where the first
        # 'source_num' rows correspond to the first detector, the next
        # 'source_num' rows correspond to the second detector, and so on.
        boxy_labels = list()
        for det_num in range(detect_num):
            for src_num in range(source_num):
                boxy_labels.append('S' + str(src_num + 1) +
                                   '_D' + str(det_num + 1))

        # Determine channel types.
        if datatype == 'Ph':
            chan_type = 'fnirs_fd_phase'
        else:
            chan_type = 'fnirs_cw_amplitude'

        ch_types = ([chan_type for i_chan in boxy_labels])

        # Create info structure.
        info = create_info(boxy_labels, srate, ch_types=ch_types)

        raw_extras = {'source_num': source_num,
                      'detect_num': detect_num,
                      'start_line': start_line,
                      'end_line': end_line,
                      'filetype': filetype,
                      'files': files[key][0],
                      'datatype': datatype,
                      'srate': srate,
                      }

        # Make sure data lengths are the same.
        print('Start Line: ', start_line)
        print('End Line: ', end_line)
        print('Original Difference: ', end_line - start_line)
        first_samps = start_line
        print('New first_samps: ', first_samps)
        diff = end_line - (start_line)

        # Number if rows in data file depends on data file type.
        if filetype == 'non-parsed':
            last_samps = diff // (source_num)
        elif filetype == 'parsed':
            last_samps = diff

        # First sample is technically sample 0, not the start line in the file.
        first_samps = 0

        print('New last_samps: ', last_samps)
        print('New Difference: ', last_samps - first_samps)

        super(RawBOXY, self).__init__(
            info, preload, filenames=[fname], first_samps=[first_samps],
            last_samps=[last_samps - 1],
            raw_extras=[raw_extras], verbose=verbose)

        # Now let's grab our markers, if they are present.
        if len(mrk_data) != 0:
            mrk_data = np.asarray(mrk_data)
            # We only want the first instance of each trigger.
            prev_mrk = 0
            mrk_idx = list()
            duration = list()
            tmp_dur = 0
            for i_num, i_mrk in enumerate(mrk_data):
                if i_mrk != 0 and i_mrk != prev_mrk:
                    mrk_idx.append(i_num)
                if i_mrk != 0 and i_mrk == prev_mrk:
                    tmp_dur += 1
                if i_mrk == 0 and i_mrk != prev_mrk:
                    duration.append((tmp_dur + 1) * (1.0 / srate))
                    tmp_dur = 0
                prev_mrk = i_mrk
            onset = [i_mrk * (1.0 / srate) for i_mrk in mrk_idx]
            description = [float(i_mrk)for i_mrk in mrk_data[mrk_idx]]
            annot = Annotations(onset, duration, description)
            self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        Boxy file organises data in two ways, parsed or un-parsed.
        Regardless of type, output has (n_montages x n_sources x n_detectors
        + n_marker_channels) rows, and (n_timepoints x n_blocks) columns.
        """
        source_num = self._raw_extras[fi]['source_num']
        detect_num = self._raw_extras[fi]['detect_num']
        start_line = self._raw_extras[fi]['start_line']
        end_line = self._raw_extras[fi]['end_line']
        filetype = self._raw_extras[fi]['filetype']
        datatype = self._raw_extras[fi]['datatype']
        boxy_files = self._raw_extras[fi]['files']

        # Possible detector names.
        detectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z']

        # Load our optical data.
        boxy_data = list()

        # Loop through our data.
        with open(boxy_files, 'r') as data_file:
            for line_num, i_line in enumerate(data_file, 1):
                if line_num == (start_line - 1):

                    # Grab column names.
                    col_names = np.asarray(re.findall(r'\w+\-\w+|\w+\-\d+|\w+',
                                                      i_line.rsplit(' ')[0]))
                if (line_num > start_line and line_num <= end_line):

                    # Grab actual data.
                    boxy_data.append(i_line.rsplit(' '))

        # Get number of sources.
        sources = np.arange(1, source_num + 1, 1)

        # Grab the individual data points for each column.
        boxy_data = [re.findall(r'[-+]?\d*\.?\d+', i_row[0])
                     for i_row in boxy_data]

        # Make variable to store our data as an array
        # rather than list of strings.
        boxy_length = len(col_names)
        boxy_array = np.full((len(boxy_data), boxy_length), np.nan)
        for ii, i_data in enumerate(boxy_data):

            # Need to make sure our rows are the same length.
            # This is done by padding the shorter ones.
            padding = boxy_length - len(i_data)
            boxy_array[ii] = np.pad(np.asarray(i_data, dtype=float),
                                    (0, padding), mode='empty')

        # Grab data from the other columns that aren't AC, DC, or Ph.
        meta_data = dict()
        keys = ['time', 'record', 'group', 'exmux', 'step', 'mark', 'flag',
                'aux1', 'digaux']
        for i_detect in detectors[0:detect_num]:
            keys.append('bias-' + i_detect)

        # Data that isn't in our boxy file will be an empty list.
        for key in keys:
            meta_data[key] = (boxy_array[:, np.where(col_names == key)[0][0]]
                              if key in col_names else list())

        # Make some empty variables to store our data.
        if filetype == 'non-parsed':
            all_data = np.zeros(((detect_num * source_num),
                                 int(len(boxy_data) / source_num)))
        elif filetype == 'parsed':
            all_data = np.zeros(((detect_num * source_num),
                                 int(len(boxy_data))))

        # Loop through detectors.
        for i_detect in detectors[0:detect_num]:

            # Loop through sources.
            for i_source in sources:

                # Determine where to store our data.
                index_loc = (detectors.index(i_detect) * source_num +
                             (i_source - 1))

                # Need to treat our filetypes differently.
                if filetype == 'non-parsed':

                    # Non-parsed saves timepoints in groups and
                    # this should account for that.
                    time_points = np.arange(i_source - 1,
                                            int(meta_data['record'][-1]) *
                                            source_num, source_num)

                    # Determine which channel to
                    # look for in boxy_array.
                    channel = np.where(col_names == i_detect + '-' +
                                       datatype)[0][0]

                    # Save our data based on data type.
                    all_data[index_loc, :] = boxy_array[time_points, channel]

                elif filetype == 'parsed':

                    # Which channel to look for in boxy_array.
                    channel = np.where(col_names == i_detect + '-' + datatype +
                                       str(i_source))[0][0]

                    # Save our data based on data type.
                    all_data[index_loc, :] = boxy_array[:, channel]

        print('Blank Data shape: ', data.shape)
        print('Input Data shape: ', all_data.shape)

        # Place our data into the data object in place.
        data[:] = all_data

        return data
