# Authors: Kyle Mathewson, Jonathan Kuziek <kuziekj@ualberta.ca>
#
# License: BSD (3-clause)

from configparser import ConfigParser, RawConfigParser
import glob as glob
import re as re
import os.path as op
import pandas as pd
import numpy as np

from ..base import BaseRaw
from ..constants import FIFF
from ..meas_info import create_info, _format_dig_points, read_fiducials
from ...annotations import Annotations
from ...transforms import apply_trans, _get_trans, get_ras_to_neuromag_trans
from ...utils import logger, verbose, fill_doc
from ...channels.montage import make_dig_montage


@fill_doc
def read_raw_boxy(fname, preload=False, verbose=None):
    """Reader for a BOXY optical imaging recording.
    Parameters
    ----------
    fname : str
        Path to the BOXY data folder.
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
    return RawBOXY(fname, preload, verbose)


@fill_doc
class RawBOXY(BaseRaw):
    """Raw object from a BOXY optical imaging file.
    Parameters
    ----------
    fname : str
        Path to the BOXY data folder.
    %(preload)s
    %(verbose)s
    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        from ...externals.pymatreader import read_mat
        from ...coreg import get_mni_fiducials, coregister_fiducials  # avoid circular import prob
        logger.info('Loading %s' % fname)

        # Check if required files exist and store names for later use
        files = dict()
        keys = ('mtg', 'elp', 'tol', '001')
        for key in keys:
            files[key] = glob.glob('%s/*%s' % (fname, key))
            if len(files[key]) != 1:
                raise RuntimeError('Expect one %s file, got %d' %
                                   (key, len(files[key]),))
            files[key] = files[key][0]

        # Read header file
        # Parse required header fields
        ###this keeps track of the line we're on###
        ###mostly to know the start and stop of data (probably an easier way)###
        line_num = 0
        ###load and read data to get some meta information###
        ###there is alot of information at the beginning of a file###
        ###but this only grabs some of it###


        with open(files['001'],'r') as data:
            for i_line in data:
                line_num += 1
                if '#DATA ENDS' in i_line:
                    end_line = line_num - 1
                    break
                if 'Detector Channels' in i_line:
                    detect_num = int(i_line.rsplit(' ')[0])
                elif 'External MUX Channels' in i_line:
                    source_num = int(i_line.rsplit(' ')[0])
                elif 'Auxiliary Channels' in i_line:
                    aux_num = int(i_line.rsplit(' ')[0])
                elif 'Waveform (CCF) Frequency (Hz)' in i_line:
                    ccf_ha = float(i_line.rsplit(' ')[0])
                elif 'Update Rate (Hz)' in i_line:
                    srate = float(i_line.rsplit(' ')[0])
                elif 'Updata Rate (Hz)' in i_line:
                    srate = float(i_line.rsplit(' ')[0])
                elif '#DATA BEGINS' in i_line:
                    start_line = line_num

        # Extract source-detectors
        ###set up some variables###
        chan_num = []
        source_label = []
        detect_label = []
        chan_wavelength = []
        chan_modulation = []

        ###load and read each line of the .mtg file###
        with open(files['mtg'],'r') as data:
            for i_ignore in range(2):
                next(data)
            for i_line in data:
                chan1, chan2, source, detector, wavelength, modulation = i_line.split()
                chan_num.append(chan1)
                source_label.append(source)
                detect_label.append(detector)
                chan_wavelength.append(wavelength)
                chan_modulation.append(modulation)

       # Read information about probe/montage/optodes
        # A word on terminology used here:
        #   Sources produce light
        #   Detectors measure light
        #   Sources and detectors are both called optodes
        #   Each source - detector pair produces a channel
        #   Channels are defined as the midpoint between source and detector

        ###check if we are given .elp file###
        all_labels = []
        all_coords = []
        fiducial_coords = []
        get_label = 0
        get_coords = 0
        ###load and read .elp file###
        with open(files['elp'],'r') as data:
            for i_line in data:
                ###first let's get our fiducial coordinates###
                if '%F' in i_line:
                    fiducial_coords.append(i_line.split()[1:])
                ###check where sensor info starts###
                if '//Sensor name' in i_line:
                    get_label = 1
                elif get_label == 1:
                    ###grab the part after '%N' for the label###
                    label = i_line.split()[1]
                    all_labels.append(label)
                    get_label = 0
                    get_coords = 1
                elif get_coords == 1:
                    X, Y, Z = i_line.split()
                    all_coords.append([float(X),float(Y),float(Z)])
                    get_coords = 0
        for i_index in range(3):
            fiducial_coords[i_index] = np.asarray([float(x) for x in fiducial_coords[i_index]])

        ###get coordinates for sources###
        source_coords = []
        for i_chan in source_label:
            if i_chan in all_labels:
                chan_index = all_labels.index(i_chan)
                source_coords.append(all_coords[chan_index])
                
        ###get coordinates for detectors###
        detect_coords = []
        for i_chan in detect_label:
            if i_chan in all_labels:
                chan_index = all_labels.index(i_chan)
                detect_coords.append(all_coords[chan_index])
                

        # Generate meaningful channel names
        ###need to rename labels to make other functions happy###
        ###get our unique labels for sources and detectors###
        unique_source_labels = []
        unique_detect_labels = []
        [unique_source_labels.append(label) for label in source_label if label not in unique_source_labels]
        [unique_detect_labels.append(label) for label in detect_label if label not in unique_detect_labels]

        ###now let's label each channel in our data###
        ###data is channels X timepoint where the first source_num rows correspond to###
        ###the first detector, and each row within that group is a different source###
        ###should note that current .mtg files contain channels for multiple data files###
        ###going to move to have a single .mtg file per participant, condition, and montage###
        ###combine coordinates and label our channels###
        ###will label them based on ac, dc, and ph data###
        boxy_coords = []
        boxy_labels = []
        data_types = ['AC','DC','Ph']
        total_chans = detect_num*source_num
        for i_type in data_types:
            for i_coord in range(len(source_coords[0:total_chans])):
                boxy_coords.append(np.mean(
                    np.vstack((source_coords[i_coord], detect_coords[i_coord])),
                    axis=0).tolist() + source_coords[i_coord] + 
                    detect_coords[i_coord] + [chan_wavelength[i_coord]] + [0] + [0])
                boxy_labels.append('S' + 
                                       str(unique_source_labels.index(source_label[i_coord])+1)
                                       + '_D' + 
                                       str(unique_detect_labels.index(detect_label[i_coord])+1) 
                                       + ' ' + chan_wavelength[i_coord] + ' ' + i_type)
        
        # add extra column for triggers
        boxy_labels.append('Markers')
        # convert to floats
        boxy_coords = np.array(boxy_coords, float)
        all_coords = np.array(all_coords, float)

        ###make our montage###
        ###montage only wants channel coords, so need to grab those, convert to###
        ###array, then make a dict with labels###
        all_chan_dict = dict(zip(all_labels,all_coords))

        my_dig_montage = make_dig_montage(ch_pos=all_chan_dict,
                                        coord_frame='unknown',
                                        nasion = fiducial_coords[0],
                                        lpa = fiducial_coords[1], 
                                        rpa = fiducial_coords[2])
        
        ###create info structure###
        info = create_info(boxy_labels, srate, ch_types='fnirs_raw')
        ###add dig info###
        ## this also applies a transform to the data into neuromag space based on fiducials
        info.set_montage(my_dig_montage)

        # Store channel, source, and detector locations
        # The channel location is stored in the first 3 entries of loc.
        # The source location is stored in the second 3 entries of loc.
        # The detector location is stored in the third 3 entries of loc.
        # Also encode the light frequency in the structure.
       
        ###place our coordinates and wavelengths for each channel###
        # # These are all in actual 3d individual coordinates, so let's transform them to
        # # the Neuromag head coordinate frame
        native_head_t = get_ras_to_neuromag_trans(fiducial_coords[0], 
                                    fiducial_coords[1], 
                                    fiducial_coords[2])
        
        for i_chan in range(len(boxy_labels)-1):
            temp_ch_src_det = apply_trans(native_head_t, boxy_coords[i_chan][:9].reshape(3, 3)).ravel()
            temp_other = np.asarray(boxy_coords[i_chan][9:], dtype=np.float64) # add wavelength and placeholders
            info['chs'][i_chan]['loc'] = np.concatenate((temp_ch_src_det, temp_other), axis=0)
        info['chs'][-1]['loc'] = np.zeros((12,))   #remove last line?     
        
        raw_extras = {'source_num': source_num,
                     'detect_num': detect_num, 
                     'start_line': start_line,
                     'files': files}

        print('Start Line: ', start_line)
        print('End Line: ', end_line)
        print('Original Difference: ', end_line-start_line)
        first_samps = start_line
        print('New first_samps: ', first_samps)
        diff = end_line-start_line
        #input file has rows for each source, output variable rearranges as columns and does not
        last_samps = start_line + diff // source_num -1 
        print('New last_samps: ', last_samps)
        print('New Difference: ', last_samps-first_samps)

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

        raw_data = pd.read_csv(self._raw_extras[fi]['files']['001'], skiprows=start_line, sep='\t')
        ###detectors, sources, and data types###
        detectors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                     'Y', 'Z']
        data_types = ['AC','DC','Ph']
        sources = np.arange(1,source_num+1,1)


            ###since we can save boxy files in two different styles###
        ###this will check to see which style the data is saved###
        ###seems to also work with older boxy files###
        if 'exmux' in raw_data.columns:
            filetype = 'non-parsed'
            
            ###drop the last line as this is just '#DATA ENDS'###
            raw_data = raw_data.drop([len(raw_data)-1])
            
            ###store some extra info###
            record = raw_data['record'].to_numpy()
            exmux = raw_data['exmux'].to_numpy()
            
            ###make some empty variables to store our data###
            raw_ac = np.zeros((detect_num*source_num,int(len(raw_data)/source_num)))
            raw_dc = np.zeros((detect_num*source_num,int(len(raw_data)/source_num)))
            raw_ph = np.zeros((detect_num*source_num,int(len(raw_data)/source_num)))
        else:
            filetype = 'parsed'
            
            ###drop the last line as this is just '#DATA ENDS'###
            ###also drop the first line since this is empty###
            raw_data = raw_data.drop([0,len(raw_data)-1])
            
            ###make some empty variables to store our data###
            raw_ac = np.zeros(((detect_num*source_num),len(raw_data)))
            raw_dc = np.zeros(((detect_num*source_num),len(raw_data)))
            raw_ph = np.zeros(((detect_num*source_num),len(raw_data)))

        ###store some extra data, might not need these though###
        time = raw_data['time'].to_numpy() if 'time' in raw_data.columns else []
        time = raw_data['time'].to_numpy() if 'time' in raw_data.columns else []
        group = raw_data['group'].to_numpy() if 'group' in raw_data.columns else []
        step = raw_data['step'].to_numpy() if 'step' in raw_data.columns else []
        mark = raw_data['mark'].to_numpy() if 'mark' in raw_data.columns else []
        flag = raw_data['flag'].to_numpy() if 'flag' in raw_data.columns else []
        aux1 = raw_data['aux-1'].to_numpy() if 'aux-1' in raw_data.columns else []
        digaux = raw_data['digaux'].to_numpy() if 'digaux' in raw_data.columns else []
        bias = np.zeros((detect_num,len(raw_data)))

       ###loop through detectors###
        for i_detect in detectors[0:detect_num]:
            
            ###older boxy files don't seem to keep track of detector bias###
            ###probably due to specific boxy settings actually###
            if 'bias-A' in raw_data.columns:
                bias[detectors.index(i_detect),:] = raw_data['bias-' + i_detect].to_numpy()
                
            ###loop through data types###
            for i_data in data_types:
                ###loop through sources###
                for i_source in sources:                    
                    ###where to store our data###
                    index_loc = detectors.index(i_detect)*source_num + (i_source-1)                    
                    ###need to treat our filetypes differently###
                    if filetype == 'non-parsed':
                        
                        ###filetype saves timepoints in groups###
                        ###this should account for that###
                        time_points = np.arange(i_source-1,int(record[-1])*source_num,source_num)
                        
                        ###determine which channel to look for###
                        channel = i_detect + '-' + i_data
                        
                        ###save our data based on data type###
                        if data_types.index(i_data) == 0:
                            raw_ac[index_loc,:] = raw_data[channel][time_points].to_numpy()
                        elif data_types.index(i_data) == 1:
                            raw_dc[index_loc,:] = raw_data[channel][time_points].to_numpy()
                        elif data_types.index(i_data) == 2:
                            raw_ph[index_loc,:] = raw_data[channel][time_points].to_numpy()
                    elif filetype == 'parsed':                    
                        ###determine which channel to look for###
                        channel = i_detect + '-' + i_data + str(i_source)
                        
                        ###save our data based on data type###
                        if data_types.index(i_data) == 0:
                            raw_ac[index_loc,:] = raw_data[channel].to_numpy()
                        elif data_types.index(i_data) == 1:
                            raw_dc[index_loc,:] = raw_data[channel].to_numpy()
                        elif data_types.index(i_data) == 2:
                            raw_ph[index_loc,:] = raw_data[channel].to_numpy()
     
        ###now combine our data types into a single array with the data###
        data_ = np.append(raw_ac, np.append(raw_dc, raw_ph, axis=0),axis=0)

        # Read triggers from event file
        ###add our markers to the data array based on filetype###
        if filetype == 'non-parsed':
            if type(digaux) is list and digaux != []:
                markers = digaux[np.arange(0,len(digaux),source_num)]
            else:
                markers = np.zeros(np.size(data_,axis=1))
        elif filetype == 'parsed':
            markers = digaux  
        
        print('Blank Data shape: ', data.shape)
        temp = np.vstack((data_, markers))
        print('Input Data shape: ',temp.shape)
        # place our data into the data object in place
        data[:] = np.vstack((data_, markers))
        
        return data
