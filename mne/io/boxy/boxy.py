# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
#
# License: BSD (3-clause)

import glob as glob
import re as re
import numpy as np
import os

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

        # Check if required files exist and store names for later use.
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

        # Determine which data type to return.
        if datatype in ['AC', 'DC', 'Ph']:
            data_types = [datatype]
        else:
            raise RuntimeError('Expect AC, DC, or Ph, got %s' % datatype)

        # Determine how many blocks we have per montage.
        blk_names = []
        mtg_names = []
        mtgs = re.findall(r'\w\.\d+', str(files['*.[000-999]*']))
        [mtg_names.append(i_mtg[0]) for i_mtg in mtgs
            if i_mtg[0] not in mtg_names]
        for i_mtg in mtg_names:
            temp = []
            [temp.append(ii_mtg[2:]) for ii_mtg in mtgs if ii_mtg[0] == i_mtg]
            blk_names.append(temp)

        # Read header file and grab some info.
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
                        # Data ends just before this.
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
                        # Data should start a couple lines later.
                        start_line.append(line_num + 2)
                    elif 'exmux' in i_line:
                        filetype[file_num] = 'non-parsed'

        # Extract source-detectors.
        chan_num_1 = []
        chan_num_2 = []
        source_label = []
        detect_label = []
        chan_wavelength = []
        chan_modulation = []

        # Load and read each line of the .mtg file.
        with open(files['mtg'], 'r') as data:
            for line_num, i_line in enumerate(data, 1):
                if line_num == 2:
                    mtg_chan_num = [int(num) for num in i_line.split()]
                elif line_num > 2:
                    (chan1, chan2, source, detector,
                     wavelength, modulation) = i_line.split()
                    chan_num_1.append(chan1)
                    chan_num_2.append(chan2)
                    source_label.append(source)
                    detect_label.append(detector)
                    chan_wavelength.append(wavelength)
                    chan_modulation.append(modulation)

        # Read information about probe/montage/optodes.
        # A word on terminology used here:
        # Sources produce light
        # Detectors measure light
        # Sources and detectors are both called optodes
        # Each source - detector pair produces a channel
        # Channels are defined as the midpoint between source and detector

        # Load and read .elp file.
        all_labels = []
        all_coords = []
        fiducial_coords = []
        get_label = 0
        get_coords = 0

        with open(files['elp'], 'r') as data:
            for i_line in data:
                # First let's get our fiducial coordinates.
                if '%F' in i_line:
                    fiducial_coords.append(i_line.split()[1:])
                # Check where sensor info starts.
                if '//Sensor name' in i_line:
                    get_label = 1
                elif get_label == 1:
                    # Grab the part after '%N' for the label.
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

        # Get coordinates from .elp file, for sources in .mtg file.
        source_coords = []
        for i_chan in source_label:
            if i_chan in all_labels:
                chan_index = all_labels.index(i_chan)
                source_coords.append(all_coords[chan_index])

        # get coordinates from .elp file, for detectors in .mtg file.
        detect_coords = []
        for i_chan in detect_label:
            if i_chan in all_labels:
                chan_index = all_labels.index(i_chan)
                detect_coords.append(all_coords[chan_index])

        # Generate meaningful channel names for each montage.
        unique_source_labels = []
        unique_detect_labels = []
        for mtg_num, i_mtg in enumerate(mtg_chan_num, 0):
            start = int(np.sum(mtg_chan_num[:mtg_num]))
            end = int(np.sum(mtg_chan_num[:mtg_num + 1]))
            [unique_source_labels.append(label)
                for label in source_label[start:end]
                if label not in unique_source_labels]
            [unique_detect_labels.append(label)
                for label in detect_label[start:end]
                if label not in unique_detect_labels]

        # Swap order to have lower wavelength first.
        for i_chan in range(0, len(chan_wavelength), 2):
            chan_wavelength[i_chan], chan_wavelength[i_chan + 1] = (
                chan_wavelength[i_chan + 1], chan_wavelength[i_chan])

        # Label each channel in our data.
        # Data is organised by channels x timepoint, where the first
        # 'source_num' rows correspond to the first detector, the next
        # 'source_num' rows correspond to the second detector, and so on.
        boxy_coords = []
        boxy_labels = []
        mrk_coords = []
        mrk_labels = []
        mtg_start = []
        mtg_end = []
        mtg_src_num = []
        mtg_det_num = []
        mtg_mdf = []
        blk_num = [len(blk) for blk in blk_names]
        for mtg_num, i_mtg in enumerate(mtg_chan_num, 0):
            start = int(np.sum(mtg_chan_num[:mtg_num]))
            end = int(np.sum(mtg_chan_num[:mtg_num + 1]))
            # Organise some data for each montage.
            start_blk = int(np.sum(blk_num[:mtg_num]))
            # Get stop and stop lines for each montage.
            mtg_start.append(start_line[start_blk])
            mtg_end.append(end_line[start_blk])
            # Get source and detector numbers for each montage.
            mtg_src_num.append(source_num[start_blk])
            mtg_det_num.append(detect_num[start_blk])
            # Get modulation frequency for each channel and montage.
            # Assuming modulation freq in MHz.
            mtg_mdf.append([int(chan_mdf) * 1e6 for chan_mdf
                            in chan_modulation[start:end]])
            for i_type in data_types:
                for i_coord in range(start, end):
                    boxy_coords.append(
                        np.mean(np.vstack((source_coords[i_coord],
                                           detect_coords[i_coord])),
                                axis=0).tolist() + source_coords[i_coord] +
                        detect_coords[i_coord] + [chan_wavelength[i_coord]] +
                        [0] + [0])
                    boxy_labels.append('S' + str(unique_source_labels.index(
                        source_label[i_coord]) + 1) + '_D' +
                        str(unique_detect_labels.index(detect_label[i_coord]) +
                            1) + ' ' + chan_wavelength[i_coord])

                # Add extra column for triggers.
                mrk_labels.append('Markers' + ' ' + mtg_names[mtg_num])
                mrk_coords.append(np.zeros((12,)))

        # Add triggers to the end of our data.
        boxy_labels.extend(mrk_labels)
        boxy_coords.extend(mrk_coords)

        # Convert to floats.
        boxy_coords = np.array(boxy_coords, float)
        all_coords = np.array(all_coords, float)

        # Montage only wants channel coords, so need to grab those,
        # convert to array, then make a dict with labels.
        all_chan_dict = dict(zip(all_labels, all_coords))

        my_dig_montage = make_dig_montage(ch_pos=all_chan_dict,
                                          coord_frame='unknown',
                                          nasion=fiducial_coords[0],
                                          lpa=fiducial_coords[1],
                                          rpa=fiducial_coords[2])

        # Create info structure.
        if datatype == 'Ph':
            chan_type = 'fnirs_ph'
        else:
            chan_type = 'fnirs_raw'

        ch_types = ([chan_type if i_chan < np.sum(mtg_chan_num) else 'stim'
                     for i_chan, _ in enumerate(boxy_labels)])
        info = create_info(boxy_labels, srate[0], ch_types=ch_types)

        # Add dig to info.
        info.set_montage(my_dig_montage)

        # Store channel, source, and detector locations.
        # The channel location is stored in the first 3 entries of loc.
        # The source location is stored in the second 3 entries of loc.
        # The detector location is stored in the third 3 entries of loc.
        # Also encode the light frequency in the structure.

        # These are all in actual 3d individual coordinates,
        # so let's transform them to the Neuromag head coordinate frame.
        native_head_t = get_ras_to_neuromag_trans(fiducial_coords[0],
                                                  fiducial_coords[1],
                                                  fiducial_coords[2])

        for i_chan in range(len(boxy_labels)):
            if i_chan < np.sum(mtg_chan_num):
                temp_ch_src_det = apply_trans(
                    native_head_t,
                    boxy_coords[i_chan][:9].reshape(3, 3)).ravel()
            else:
                # Don't want to transform markers.
                temp_ch_src_det = np.zeros(9,)
            # Add wavelength and placeholders.
            temp_other = np.asarray(boxy_coords[i_chan][9:], dtype=np.float64)
            info['chs'][i_chan]['loc'] = np.concatenate((temp_ch_src_det,
                                                        temp_other), axis=0)

        raw_extras = {'source_num': source_num,
                      'detect_num': detect_num,
                      'start_line': start_line,
                      'end_line': end_line,
                      'filetype': filetype,
                      'files': files,
                      'montages': mtg_names,
                      'blocks': blk_names,
                      'data_types': data_types,
                      'mtg_mdf': mtg_mdf,
                      }

        # Check data start lines.
        if len(set(start_line)) == 1:
            print('Start lines the same!')
        else:
            print('Start lines different!')

        # Check data end lines.
        if len(set(end_line)) == 1:
            print('End lines the same!')
        else:
            print('End lines different!')

        # Make sure data lengths are the same.
        data_length = ([end_line[i_line] - start_line[i_line] for i_line,
                        line_num in enumerate(start_line)])

        if len(set(data_length)) == 1:
            print('Data sizes are the same!')
        else:
            print('Data sizes are different!')

        print('Start Line: ', start_line[0])
        print('End Line: ', end_line[0])
        print('Original Difference: ', end_line[0] - start_line[0])
        first_samps = start_line[0]
        print('New first_samps: ', first_samps)
        diff = end_line[0] - (start_line[0])

        # Number if rows in data file depends on data file type.
        if filetype[0] == 'non-parsed':
            last_samps = ((diff * len(blk_names[0])) // (source_num[0]))
        elif filetype[0] == 'parsed':
            last_samps = diff * len(blk_names[0])

        # First sample is technically sample 0, not the start line in the file.
        first_samps = 0

        print('New last_samps: ', last_samps)
        print('New Difference: ', last_samps - first_samps)

        super(RawBOXY, self).__init__(
            info, preload, filenames=[fname], first_samps=[first_samps],
            last_samps=[last_samps - 1],
            raw_extras=[raw_extras], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        Boxy file organises data in two ways, parsed or un-parsed.
        Regardless of type, output has (n_montages x n_sources x n_detectors
        + n_marker_channels) rows, and (n_timepoints x n_blocks) columns.
        """
        import scipy.io as spio

        source_num = self._raw_extras[fi]['source_num']
        detect_num = self._raw_extras[fi]['detect_num']
        start_line = self._raw_extras[fi]['start_line']
        end_line = self._raw_extras[fi]['end_line']
        filetype = self._raw_extras[fi]['filetype']
        data_types = self._raw_extras[fi]['data_types']
        montages = self._raw_extras[fi]['montages']
        blocks = self._raw_extras[fi]['blocks']
        mtg_mdf = self._raw_extras[fi]['mtg_mdf']
        boxy_files = self._raw_extras[fi]['files']['*.[000-999]*']
        event_fname = os.path.join(self._filenames[fi], 'evt')

        # Check if event files are available.
        # Mostly for older boxy files since we'll be using the digaux channel
        # for markers in further recordings.
        try:
            event_files = dict()
            key = ('*.[000-999]*')
            print(event_fname)
            event_files[key] = [glob.glob('%s/*%s' % (event_fname, key))]
            event_files[key] = event_files[key][0]
            event_data = []

            for file_num, i_file in enumerate(event_files[key]):
                event_data.append(spio.loadmat(
                    event_files[key][file_num])['event'])
            if event_data != []:
                print('Event file found!')
            else:
                print('No event file found. Using digaux!')

        except Exception:
            print('No event file found. Using digaux!')
            pass

        # Possible detector names.
        detectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z']

        # Load our optical data.
        all_data = []
        all_markers = []
        for i_mtg, mtg_name in enumerate(montages):
            all_blocks = []
            block_markers = []
            for i_blk, blk_name in enumerate(blocks[i_mtg]):
                file_num = i_blk + (i_mtg * len(blocks[i_mtg]))
                boxy_file = boxy_files[file_num]
                boxy_data = []
                with open(boxy_file, 'r') as data_file:
                    for line_num, i_line in enumerate(data_file, 1):
                        if line_num == (start_line[i_blk] - 1):
                            # Grab column names.
                            col_names = np.asarray(
                                re.findall(r'\w+\-\w+|\w+\-\d+|\w+',
                                           i_line.rsplit(' ')[0]))
                        if (line_num > start_line[file_num] and
                                line_num <= end_line[file_num]):
                            boxy_data.append(i_line.rsplit(' '))

                sources = np.arange(1, source_num[file_num] + 1, 1)

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
                keys = ['time', 'record', 'group', 'exmux', 'step', 'mark',
                        'flag', 'aux1', 'digaux']
                for i_detect in detectors[0:detect_num[file_num]]:
                    keys.append('bias-' + i_detect)

                # Data that isn't in our boxy file will be an empty list.
                for key in keys:
                    meta_data[key] = (boxy_array[:,
                                      np.where(col_names == key)[0][0]] if
                                      key in col_names else [])

                # Make some empty variables to store our data.
                if filetype[file_num] == 'non-parsed':
                    data_ = np.zeros(((((detect_num[file_num] *
                                        source_num[file_num]) *
                                        len(data_types))),
                                      int(len(boxy_data) /
                                          source_num[file_num])))
                elif filetype[file_num] == 'parsed':
                    data_ = np.zeros(((((detect_num[file_num] *
                                         source_num[file_num]) *
                                        len(data_types))),
                                      int(len(boxy_data))))

                # Loop through data types.
                for i_data in data_types:

                    # Loop through detectors.
                    for i_detect in detectors[0:detect_num[file_num]]:

                        # Loop through sources.
                        for i_source in sources:

                            # Determine where to store our data.
                            index_loc = (detectors.index(i_detect) *
                                         source_num[file_num] +
                                         (i_source - 1) +
                                         (data_types.index(i_data) *
                                         (source_num[file_num] *
                                          detect_num[file_num])))

                            # Need to treat our filetypes differently.
                            if filetype[file_num] == 'non-parsed':

                                # Non-parsed saves timepoints in groups and
                                # this should account for that.
                                time_points = np.arange(
                                    i_source - 1,
                                    int(meta_data['record'][-1]) *
                                    source_num[file_num],
                                    source_num[file_num])

                                # Determine which channel to
                                # look for in boxy_array.
                                channel = np.where(col_names == i_detect +
                                                   '-' + i_data)[0][0]

                                # Save our data based on data type.
                                data_[index_loc, :] = boxy_array[time_points,
                                                                 channel]

                            elif filetype[file_num] == 'parsed':

                                # Which channel to look for in boxy_array.
                                channel = np.where(col_names == i_detect +
                                                   '-' + i_data +
                                                   str(i_source))[0][0]

                                # Save our data based on data type.
                                data_[index_loc, :] = boxy_array[:, channel]

                    # Phase unwrapping.
                    if i_data == 'Ph':
                        print('Fixing phase wrap')
                        # Accounts for sharp, sudden changes in phase
                        # such as crossing over from 0/360 degrees.
                        # Estimate mean phase of first 50 points.
                        # If a point differs more than 90 degrees from the
                        # mean, add or subtract 360 degrees from that point.
                        for i_chan in range(np.size(data_, axis=0)):
                            if np.mean(data_[i_chan, :50]) < 180:
                                wrapped_points = data_[i_chan, :] > 270
                                data_[i_chan, wrapped_points] -= 360
                            else:
                                wrapped_points = data_[i_chan, :] < 90
                                data_[i_chan, wrapped_points] += 360

                        print('Detrending phase data')
                        # Remove trends and drifts that occur over time.
                        y = np.linspace(0, np.size(data_, axis=1) - 1,
                                        np.size(data_, axis=1))
                        x = np.transpose(y)
                        for i_chan in range(np.size(data_, axis=0)):
                            poly_coeffs = np.polyfit(x, data_[i_chan, :], 3)
                            tmp_ph = (data_[i_chan, :] -
                                      np.polyval(poly_coeffs, x))
                            data_[i_chan, :] = tmp_ph

                        print('Removing phase mean')
                        # Subtract mean to better detect outliers using SD.

                        mrph = np.mean(data_, axis=1)
                        for i_chan in range(np.size(data_, axis=0)):
                            data_[i_chan, :] = (data_[i_chan, :] -
                                                mrph[i_chan])

                        print('Removing phase outliers')
                        # Remove data points that are larger than three SDs.
                        ph_out_thr = 3

                        # Set ddof to 1 to mimic matlab.
                        sdph = np.std(data_, 1, ddof=1)
                        n_ph_out = np.zeros(np.size(data_, axis=0),
                                            dtype=np.int8)

                        for i_chan in range(np.size(data_, axis=0)):
                            outliers = np.where(np.abs(data_[i_chan, :]) >
                                                (ph_out_thr * sdph[i_chan]))
                            outliers = outliers[0]
                            if len(outliers) > 0:
                                if outliers[0] == 0:
                                    outliers = outliers[1:]
                                if len(outliers) > 0:
                                    if (outliers[-1] == np.size(data_,
                                                                axis=1) - 1):
                                        outliers = outliers[:-1]
                                    n_ph_out[i_chan] = int(len(outliers))
                                    for i_pt in range(n_ph_out[i_chan]):
                                        j_pt = outliers[i_pt]
                                        data_[i_chan, j_pt] = (
                                            (data_[i_chan, j_pt - 1] +
                                             data_[i_chan, j_pt + 1]) / 2)

                        # Convert phase to pico seconds.
                        for i_chan in range(np.size(data_, axis=0)):
                            data_[i_chan, :] = ((1e12 * data_[i_chan, :]) /
                                                (360 * mtg_mdf[i_mtg][i_chan]))

                # Swap channels to match new wavelength order.
                for i_chan in range(0, len(data_), 2):
                    data_[[i_chan, i_chan + 1]] = data_[[i_chan + 1, i_chan]]

                # If there was an event file, place those events in our data.
                # If no, use digaux for our events.
                try:
                    temp_markers = np.zeros((len(data_[0, :]),))
                    for event_num, event_info in enumerate(
                            event_data[file_num]):
                        temp_markers[event_info[0] - 1] = event_info[1]
                    block_markers.append(temp_markers)
                except Exception:
                    # Add our markers to the data array based on filetype.
                    if type(meta_data['digaux']) is not list:
                        if filetype[file_num] == 'non-parsed':
                            block_markers.append(
                                meta_data['digaux']
                                [np.arange(0, len(meta_data['digaux']),
                                           source_num[file_num])])
                        elif filetype[file_num] == 'parsed':
                            block_markers.append(meta_data['digaux'])
                    else:
                        block_markers.append(np.zeros((len(data_[0, :]),)))

                # Check our markers to see if anything is actually in there.
                if (all(i_mrk == 0 for i_mrk in block_markers[i_blk]) or
                        all(i_mrk == 255 for i_mrk in block_markers[i_blk])):
                    print('No markers for montage ' + mtg_name +
                          ' and block ' + blk_name)
                else:
                    print('Found markers for montage ' + mtg_name +
                          ' and block ' + blk_name + '!')

                # Change marker for last timepoint to indicate end of block
                # We'll be using digaux to send markers, a serial port,
                # so we can send values between 1-255.
                # We'll multiply our block start/end markers by 1000 to ensure
                # we aren't within the 1-255 range.
                block_markers[i_blk][-1] = int(blk_name) * 1000

                all_blocks.append(data_)

            all_data.extend(np.hstack(all_blocks))
            all_markers.append(np.hstack(block_markers))

        # Add markers to our data.
        all_data.extend(all_markers)
        all_data = np.asarray(all_data)

        print('Blank Data shape: ', data.shape)
        print('Input Data shape: ', all_data.shape)

        # Place our data into the data object in place.
        data[:] = all_data

        return data
