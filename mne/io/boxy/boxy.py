# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
#
# License: BSD (3-clause)

import glob as glob
import re as re
import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ...transforms import apply_trans, get_ras_to_neuromag_trans
from ...utils import logger, verbose, fill_doc
from ...channels.montage import make_dig_montage


@fill_doc
def read_raw_boxy(fname, datatype='AC', preload=False, verbose=None):
    """Reader for a BOXY optical imaging recording.
    Parameters
    ----------
    fname : str
        Path to the BOXY data folder.
    datatype : str
        Type of data to return (AC, DC, or Ph)
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
        Type of data to return (AC, DC, or Ph)
    %(preload)s
    %(verbose)s
    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, datatype='AC', preload=False, verbose=None):
        logger.info('Loading %s' % fname)

        # Check if required files exist and store names for later use
        files = dict()
        keys = ('mtg', 'elp', '*.[000-999]*')
        print(fname)
        for key in keys:
            if key == '*.[000-999]*':
                files[key] = [glob.glob('%s/*%s' % (fname, key))]
            else:
                files[key] = glob.glob('%s/*%s' % (fname, key))
            if len(files[key]) != 1:
                raise RuntimeError('Expect one %s file, got %d' %
                                   (key, len(files[key]),))
            files[key] = files[key][0]

        # determine which data type to return###
        if datatype in ['AC', 'DC', 'Ph']:
            data_types = [datatype]
        else:
            raise RuntimeError('Expect AC, DC, or Ph, got %s' % datatype)

        # determine how many blocks we have per montage
        blk_names = []
        mtg_names = []
        mtgs = re.findall('\w\.\d+', str(files['*.[000-999]*']))
        [mtg_names.append(i_mtg[0]) for i_mtg in mtgs
            if i_mtg[0] not in mtg_names]
        for i_mtg in mtg_names:
            temp = []
            [temp.append(ii_mtg[2:]) for ii_mtg in mtgs if ii_mtg[0] == i_mtg]
            blk_names.append(temp)

        # Read header file
        # Parse required header fields
        # this keeps track of the line we're on
        # mostly to know the start and stop of data (probably an easier way)
        # load and read data to get some meta information
        # there is alot of information at the beginning of a file
        # but this only grabs some of it

        detect_num = []
        source_num = []
        aux_num = []
        ccf_ha = []
        srate = []
        start_line = []
        end_line = []
        filetype = ['parsed' for i_file in files['*.[000-999]*']]
        for file_num, i_file in enumerate(files['*.[000-999]*'], 0):
            with open(i_file, 'r') as data:
                for line_num, i_line in enumerate(data, 1):
                    if '#DATA ENDS' in i_line:
                        end_line.append(line_num - 1)
                        break
                    if 'Detector Channels' in i_line:
                        detect_num.append(int(i_line.rsplit(' ')[0]))
                    elif 'External MUX Channels' in i_line:
                        source_num.append(int(i_line.rsplit(' ')[0]))
                    elif 'Auxiliary Channels' in i_line:
                        aux_num.append(int(i_line.rsplit(' ')[0]))
                    elif 'Waveform (CCF) Frequency (Hz)' in i_line:
                        ccf_ha.append(float(i_line.rsplit(' ')[0]))
                    elif 'Update Rate (Hz)' in i_line:
                        srate.append(float(i_line.rsplit(' ')[0]))
                    elif 'Updata Rate (Hz)' in i_line:
                        srate.append(float(i_line.rsplit(' ')[0]))
                    elif '#DATA BEGINS' in i_line:
                        start_line.append(line_num)
                    elif 'exmux' in i_line:
                        filetype[file_num] = 'non-parsed'

        # Extract source-detectors
        # set up some variables
        chan_num_1 = []
        chan_num_2 = []
        source_label = []
        detect_label = []
        chan_wavelength = []
        chan_modulation = []

        # load and read each line of the .mtg file
        with open(files['mtg'], 'r') as data:
            for line_num, i_line in enumerate(data, 1):
                if line_num == 2:
                    mtg_chan_num = [int(num) for num in i_line.split()]
                elif line_num > 2:
                    chan1, chan2, source, detector, wavelength, modulation = i_line.split()
                    chan_num_1.append(chan1)
                    chan_num_2.append(chan2)
                    source_label.append(source)
                    detect_label.append(detector)
                    chan_wavelength.append(wavelength)
                    chan_modulation.append(modulation)

        # Read information about probe/montage/optodes
        # A word on terminology used here:
        # Sources produce light
        # Detectors measure light
        # Sources and detectors are both called optodes
        # Each source - detector pair produces a channel
        # Channels are defined as the midpoint between source and detector

        # check if we are given .elp file
        all_labels = []
        all_coords = []
        fiducial_coords = []
        get_label = 0
        get_coords = 0

        # load and read .elp file
        with open(files['elp'], 'r') as data:
            for i_line in data:
                # first let's get our fiducial coordinates
                if '%F' in i_line:
                    fiducial_coords.append(i_line.split()[1:])
                # check where sensor info starts
                if '//Sensor name' in i_line:
                    get_label = 1
                elif get_label == 1:
                    # grab the part after '%N' for the label
                    label = i_line.split()[1]
                    all_labels.append(label)
                    get_label = 0
                    get_coords = 1
                elif get_coords == 1:
                    X, Y, Z = i_line.split()
                    all_coords.append([float(X), float(Y), float(Z)])
                    get_coords = 0
        for i_index in range(3):
            fiducial_coords[i_index] = np.asarray([float(x)
                                                  for x in
                                                  fiducial_coords[i_index]])

        # get coordinates for sources in .mtg file from .elp file
        source_coords = []
        for i_chan in source_label:
            if i_chan in all_labels:
                chan_index = all_labels.index(i_chan)
                source_coords.append(all_coords[chan_index])

        # get coordinates for detectors in .mtg file from .elp file
        detect_coords = []
        for i_chan in detect_label:
            if i_chan in all_labels:
                chan_index = all_labels.index(i_chan)
                detect_coords.append(all_coords[chan_index])

        # Generate meaningful channel names for each montage
        # get our unique labels for sources and detectors for each montage
        unique_source_labels = []
        unique_detect_labels = []
        for mtg_num, i_mtg in enumerate(mtg_chan_num, 0):
            mtg_source_labels = []
            mtg_detect_labels = []
            start = int(np.sum(mtg_chan_num[:mtg_num]))
            end = int(np.sum(mtg_chan_num[:mtg_num + 1]))
            [mtg_source_labels.append(label)
                for label in source_label[start:end]
                if label not in mtg_source_labels]
            [mtg_detect_labels.append(label)
                for label in detect_label[start:end]
                if label not in mtg_detect_labels]
            unique_source_labels.append(mtg_source_labels)
            unique_detect_labels.append(mtg_detect_labels)

        # swap order to have lower wavelength first
        for i_chan in range(0, len(chan_wavelength), 2):
            chan_wavelength[i_chan], chan_wavelength[i_chan + 1] = (
                chan_wavelength[i_chan + 1], chan_wavelength[i_chan])

        # now let's label each channel in our data
        # data is channels X timepoint where the first source_num rows
        # correspond to the first detector, and each row within that
        # group is a different source should note that
        # current .mtg files contain channels for multiple
        # data files going to move to have a single .mtg file
        # per participant, condition, and montage
        # combine coordinates and label our channels
        # will label them based on ac, dc, and ph data
        boxy_coords = []
        boxy_labels = []
        mrk_coords = []
        mrk_labels = []
        mtg_start = []
        mtg_end = []
        mtg_src_num = []
        mtg_det_num = []
        blk_num = [len(blk) for blk in blk_names]
        for mtg_num, i_mtg in enumerate(mtg_chan_num, 0):
            start = int(np.sum(mtg_chan_num[:mtg_num]))
            end = int(np.sum(mtg_chan_num[:mtg_num + 1]))
            # we will also organise some data for each montage
            start_blk = int(np.sum(blk_num[:mtg_num]))
            # get stop and stop lines for each montage
            mtg_start.append(start_line[start_blk])
            mtg_end.append(end_line[start_blk])
            # get source and detector numbers for each montage
            mtg_src_num.append(source_num[start_blk])
            mtg_det_num.append(detect_num[start_blk])
            for i_blk in blk_names[mtg_num]:
                for i_type in data_types:
                    for i_coord in range(start, end):
                        boxy_coords.append(np.mean(
                            np.vstack((source_coords[i_coord],
                                       detect_coords[i_coord])),
                            axis=0).tolist() + source_coords[i_coord] +
                            detect_coords[i_coord] +
                            [chan_wavelength[i_coord]] +
                            [0] + [0])
                        boxy_labels.append('S' + str(
                            unique_source_labels[mtg_num].index(
                                source_label[i_coord]) + 1) + '_D' +
                            str(unique_detect_labels[mtg_num].index(
                                detect_label[i_coord]) + 1) +
                            ' ' + chan_wavelength[i_coord] + ' ' +
                            mtg_names[mtg_num] + i_blk[1:])

                # add extra column for triggers
                mrk_labels.append('Markers' + ' ' +
                                  mtg_names[mtg_num] + i_blk[1:])
                mrk_coords.append(np.zeros((12,)))

        # add triggers to the end of our data
        boxy_labels.extend(mrk_labels)
        boxy_coords.extend(mrk_coords)

        # convert to floats
        boxy_coords = np.array(boxy_coords, float)
        all_coords = np.array(all_coords, float)

        # make our montage
        # montage only wants channel coords, so need to grab those, convert to
        # array, then make a dict with labels
        all_chan_dict = dict(zip(all_labels, all_coords))

        my_dig_montage = make_dig_montage(ch_pos=all_chan_dict,
                                          coord_frame='unknown',
                                          nasion=fiducial_coords[0],
                                          lpa=fiducial_coords[1],
                                          rpa=fiducial_coords[2])

        # create info structure
        info = create_info(boxy_labels, srate[0], ch_types='fnirs_raw')
        # add dig info

        # this also applies a transform to the data into neuromag space
        # based on fiducials
        info.set_montage(my_dig_montage)

        # Store channel, source, and detector locations
        # The channel location is stored in the first 3 entries of loc.
        # The source location is stored in the second 3 entries of loc.
        # The detector location is stored in the third 3 entries of loc.
        # Also encode the light frequency in the structure.

        # place our coordinates and wavelengths for each channel
        # These are all in actual 3d individual coordinates,
        # so let's transform them to the Neuromag head coordinate frame
        native_head_t = get_ras_to_neuromag_trans(fiducial_coords[0],
                                                  fiducial_coords[1],
                                                  fiducial_coords[2])

        for i_chan in range(len(boxy_labels)):
            temp_ch_src_det = apply_trans(native_head_t,
                                          boxy_coords[i_chan][:9].reshape(3, 3)
                                          ).ravel()
            # add wavelength and placeholders
            temp_other = np.asarray(boxy_coords[i_chan][9:], dtype=np.float64)
            info['chs'][i_chan]['loc'] = np.concatenate((temp_ch_src_det,
                                                        temp_other), axis=0)

        raw_extras = {'source_num': source_num,
                      'detect_num': detect_num,
                      'start_line': start_line,
                      'end_line': end_line,
                      'filetype': filetype,
                      'files': files,
                      'data_types': data_types}

        print('Start Line: ', start_line[0])
        print('End Line: ', end_line[0])
        print('Original Difference: ', end_line[0] - start_line[0])
        first_samps = start_line[0]
        print('New first_samps: ', first_samps)
        diff = end_line[0] - start_line[0]

        # input file has rows for each source,
        # output variable rearranges as columns and does not
        if filetype[0] == 'non-parsed':
            last_samps = ((diff - 2) // (source_num[0])) + start_line[0] - 1
        elif filetype == 'parsed':
            last_samps = (start_line[0] + diff)

        print('New last_samps: ', last_samps)
        print('New Difference: ', last_samps - first_samps)

        super(RawBOXY, self).__init__(
            info, preload, filenames=[fname], first_samps=[first_samps],
            last_samps=[last_samps],
            raw_extras=[raw_extras], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.
        """
        source_num = self._raw_extras[fi]['source_num']
        detect_num = self._raw_extras[fi]['detect_num']
        start_line = self._raw_extras[fi]['start_line']
        end_line = self._raw_extras[fi]['end_line']
        filetype = self._raw_extras[fi]['filetype']
        data_types = self._raw_extras[fi]['data_types']
        boxy_files = self._raw_extras[fi]['files']['*.[000-999]*']

        # detectors, sources, and data types
        detectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z']

        # load our data
        all_data = []
        markers = []
        for file_num, boxy_file in enumerate(boxy_files):
            boxy_data = []
            with open(boxy_file, 'r') as data_file:
                for line_num, i_line in enumerate(data_file, 1):
                    if line_num > start_line[file_num] and line_num <= end_line[file_num]:
                        boxy_data.append(i_line.rsplit(' '))

            sources = np.arange(1, source_num[file_num] + 1, 1)

            # get column names from the first row of our boxy data
            col_names = np.asarray(re.findall('\w+\-\w+|\w+\-\d+|\w+',
                                   boxy_data[0][0]))
            del boxy_data[0]

            # sometimes there is an empty line before our data starts
            # this should remove them
            while re.findall('[-+]?\d*\.?\d+', boxy_data[0][0]) == []:
                del boxy_data[0]

            # grab the individual data points for each column
            boxy_data = [re.findall('[-+]?\d*\.?\d+', i_row[0])
                         for i_row in boxy_data]

            # make variable to store our data as an array
            # rather than list of strings
            boxy_length = len(col_names)
            boxy_array = np.full((len(boxy_data), boxy_length), np.nan)
            for ii, i_data in enumerate(boxy_data):
                # need to make sure our rows are the same length
                # this is done by padding the shorter ones
                padding = boxy_length - len(i_data)
                boxy_array[ii] = np.pad(np.asarray(i_data, dtype=float),
                                        (0, padding), mode='empty')

            # grab data from the other columns
            # that don't pertain to AC, DC, or Ph
            meta_data = dict()
            keys = ['time', 'record', 'group', 'exmux', 'step', 'mark',
                    'flag', 'aux1', 'digaux']
            for i_detect in detectors[0:detect_num[file_num]]:
                keys.append('bias-' + i_detect)

            # data that isn't in our boxy file will be an empty list
            for key in keys:
                meta_data[key] = (boxy_array[:,
                                  np.where(col_names == key)[0][0]] if
                                  key in col_names else [])

            # make some empty variables to store our data
            if filetype[file_num] == 'non-parsed':
                data_ = np.zeros(((((detect_num[file_num] *
                                 source_num[file_num]) * len(data_types))),
                                 int(len(boxy_data) / source_num[file_num])))
            elif filetype[file_num] == 'parsed':
                data_ = np.zeros(((((detect_num[file_num] * 
                                 source_num[file_num]) * len(data_types))),
                                 int(len(boxy_data))))

            # loop through data types
            for i_data in data_types:

                # loop through detectors
                for i_detect in detectors[0:detect_num[file_num]]:

                    # loop through sources
                    for i_source in sources:

                        # determine where to store our data
                        index_loc = (detectors.index(i_detect) *
                                     source_num[file_num] +
                                     (i_source - 1) +
                                     (data_types.index(i_data) *
                                     (source_num[file_num] *
                                      detect_num[file_num])))

                        # need to treat our filetypes differently
                        if filetype[file_num] == 'non-parsed':

                            # non-parsed saves timepoints in groups
                            # this should account for that
                            time_points = np.arange(i_source - 1,
                                                    int(
                                                        meta_data['record'][-1]
                                                    ) * source_num[file_num],
                                                    source_num[file_num])

                            # determine which channel to look for in boxy_array
                            channel = np.where(col_names == i_detect +
                                               '-' + i_data)[0][0]

                            # save our data based on data type
                            data_[index_loc, :] = boxy_array[time_points,
                                                             channel]

                        elif filetype[file_num] == 'parsed':

                            # determine which channel to look for in boxy_array
                            channel = np.where(col_names == i_detect + '-' +
                                               i_data + str(i_source))[0][0]

                            # save our data based on data type
                            data_[index_loc, :] = boxy_array[:, channel]

            # swap channels to match new wavelength order
            for i_chan in range(0, len(data_), 2):
                data_[[i_chan, i_chan + 1]] = data_[[i_chan + 1, i_chan]]

            # Read triggers from event file
            # add our markers to the data array based on filetype###
            if type(meta_data['digaux']) is not list:
                if filetype[file_num] == 'non-parsed':
                    markers.append(meta_data['digaux'][np.arange(0,
                                   len(meta_data['digaux']),
                                   source_num[file_num])])
                elif filetype[file_num] == 'parsed':
                    markers.append(meta_data['digaux'])
            else:
                markers.append(np.zeros((len(data_[0, :]),)))

            all_data.extend(data_)

        # add markers to our data
        all_data.extend(markers)
        all_data = np.asarray(all_data)

        print('Blank Data shape: ', data.shape)
        print('Input Data shape: ', all_data.shape)
        # place our data into the data object in place
        data[:] = all_data

        return data
