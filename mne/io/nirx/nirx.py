# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import re as re
import glob as glob
from configparser import ConfigParser, RawConfigParser
import numpy as np


from ..base import BaseRaw
from ..meas_info import create_info
from ...utils import logger, verbose, fill_doc, _check_pandas_installed
from ..constants import FIFF


@fill_doc
def read_raw_nirx(fname, preload=False, verbose=None):
    """Reader for a NIRX fNIRS recording.

    Parameters
    ----------
    fname : str
        Path to the NIRX data folder.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawNIRX
        A Raw object containing NIRX data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """
    return RawNIRX(fname, preload, verbose)


@fill_doc
class RawNIRX(BaseRaw):
    """Raw object from a NIRX fNIRS file.

    Parameters
    ----------
    fname : str
        Path to the NIRX data folder.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        logger.info('Loading %s' % fname)

        # Check if required files exist and store names for later use
        files = dict()
        keys = ('dat', 'evt', 'hdr', 'inf', 'set', 'tpl', 'wl1', 'wl2',
                'config.txt', 'probeInfo.mat')
        for key in keys:
            files[key] = glob.glob('%s/*%s' % (fname, key))
            if len(files[key]) != 1:
                raise RuntimeError('Expect one %s file, got %d' %
                                   (key, len(files[key]),))
            files[key] = files[key][0]

        # Read number of rows/samples of wavelength data
        last_sample = -1
        for line in open(files['wl1']):
            last_sample += 1

        # Read participant information file
        inf = ConfigParser(allow_no_value=True)
        inf.read(files['inf'])
        inf = inf._sections['Subject Demographics']

        # mne requires specific fields for participant info
        # https://github.com/mne-tools/mne-python/ ...
        # blob/master/mne/io/meas_info.py#L430
        # TODO: Can you put more values in subject_info than specified in link?
        #       NIRX also records "Study Type", "Experiment History",
        #       "Additional Notes", "Contact Information"
        subject_info = {}
        names = inf['name'].split()
        if len(names) > 0:
            subject_info['first_name'] = \
                inf['name'].split()[0].replace("\"", "")
        if len(names) > 1:
            subject_info['last_name'] = \
                inf['name'].split()[-1].replace("\"", "")
        if len(names) > 2:
            subject_info['middle_name'] = \
                inf['name'].split()[-2].replace("\"", "")
        subject_info['birthday'] = inf['age']
        subject_info['sex'] = inf['gender'].replace("\"", "")
        # Recode values
        if subject_info['sex'] in {'M', 'Male', '1'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_MALE
        elif subject_info['sex'] in {'F', 'Female', '2'}:
            subject_info['sex'] = FIFF.FIFFV_SUBJ_SEX_FEMALE
        # NIRStar does not record an id, or handedness by default

        # Read header file
        # This is a bit tricky as the header file isn't compliant with
        # the config specifications. So we need to remove all text
        # between comments before passing to config parser
        with open(files['hdr']) as f:
            hdr_str = f.read()
        hdr_str = re.sub('#.*?#', '', hdr_str, flags=re.DOTALL)
        hdr = RawConfigParser()
        hdr.read_string(hdr_str)

        # Check that the file format version is supported
        if hdr['GeneralInfo']['NIRStar'] != "\"15.2\"":
            raise RuntimeError('Only NIRStar version 15.2 is supported')

        # Parse required header fields

        # Extract frequencies of light used by machine
        fnirs_wavelengths = [int(s) for s in
                             re.findall(r'(\d+)',
                             hdr['ImagingParameters']['Wavelengths'])]

        # Extract source-detectors requested by user
        sources = np.asarray([int(s) for s in re.findall(r'(\d+)-\d+:\d+',
                              hdr['DataStructure']['S-D-Key'])])
        detectors = np.asarray([int(s) for s in re.findall(r'\d+-(\d+):\d+',
                                hdr['DataStructure']['S-D-Key'])])

        # Determine if short channels are present
        has_short = np.array(hdr['ImagingParameters']['ShortBundles'], int)
        short_det = [int(s) for s in
                     re.findall(r'(\d+)',
                     hdr['ImagingParameters']['ShortDetIndex'])]
        short_det = np.array(short_det, int)

        # Extract sampling rate
        samplingrate = float(hdr['ImagingParameters']['SamplingRate'])

        # Read information about probe/montage/optodes
        # A word on terminology used here:
        #   Sources produce light
        #   Detectors measure light
        #   Sources and detectors are both called optodes
        #   Each source - detector pair produces a channel
        #   Channels are defined as the midpoint between source and detector
        # Information is available about the sources, detector, and channel
        # locations. Currently I am storing the location of the channel as this
        # seems the most similar to EEG. But ideally for each channel we could
        # store the location of all three (source, detector, channel).
        # The channel info is most useful for scalp level analysis.
        # Source detector locations would be useful for photon migration
        # modelling, determining which region of the brain the fNIRS signal is
        # likely to originate from.
        from ...externals.pymatreader import read_mat
        mat_data = read_mat(files['probeInfo.mat'], uint16_codec=None)
        # Following values will be required by commented out to keep flake
        # happy until i put in locations via info['chs'][ii]['loc']
        # detector_location_labels =mat_data['probeInfo']['probes']['labels_d']
        # source_location_labels = mat_data['probeInfo']['probes']['labels_s']
        # num_sources = mat_data['probeInfo']['probes']['nSource0']
        # num_detectors = mat_data['probeInfo']['probes']['nDetector0']
        requested_channels = mat_data['probeInfo']['probes']['index_c']
        src_locs = mat_data['probeInfo']['probes']['coords_s3'] * 10
        det_locs = mat_data['probeInfo']['probes']['coords_d3'] * 10
        ch_locs = mat_data['probeInfo']['probes']['coords_c3'] * 10

        # Determine requested channel indices
        # The wl1 and wl2 files include all possible source - detector pairs
        # But most of these are not relevant. We want to extract only subset
        req_ind = np.array([])
        for req_idx in range(requested_channels.shape[0]):
            sd_idx = np.where((sources == requested_channels[req_idx][0]) &
                              (detectors == requested_channels[req_idx][1]))
            req_ind = np.concatenate((req_ind, sd_idx[0]))
        req_ind = req_ind.astype(int)

        # Generate meaningful channel names
        def prepend(list, str):
            str += '{0}'
            list = [str.format(i) for i in list]
            return(list)
        snames = prepend(sources[req_ind], 'S')
        dnames = prepend(detectors[req_ind], '-D')
        sdnames = [m + str(n) for m, n in zip(snames, dnames)]
        sd1 = [s + ' ' + str(fnirs_wavelengths[0]) + ' (nm)' for s in sdnames]
        sd2 = [s + ' ' + str(fnirs_wavelengths[1]) + ' (nm)' for s in sdnames]
        chnames = [val for pair in zip(sd1, sd2) for val in pair]

        # Create mne structure
        # TODO: ch_type is currently misc as I could not find appropriate
        #       other type, the hbo and hbr type are not relevant until the
        #       signal has been converted
        # TODO: nchan needs to be multiplied by two as we have two wavelengths
        #       per sensor should the underlying type be modified to
        #       support (wavelength x channels x data)?
        info = create_info(chnames,
                           samplingrate,
                           ch_types='misc')
        info.update({'subject_info': subject_info})

        # Store channel, source, and detector locations
        # The channel location is stored in the first 3 entries of loc
        # The source location is stored in the second 3 entries of loc
        # The detector location is stored in the third 3 entries of loc
        # TODO: pretty sure this should be done using a info.update call
        # TODO: what are the units here? What coordinate system is it in?
        for ch_idx2 in range(requested_channels.shape[0]):
            # Find source and store location
            src = int(requested_channels[ch_idx2, 0]) - 1
            info['chs'][ch_idx2 * 2]['loc'][3:6] = src_locs[src, :]
            info['chs'][ch_idx2 * 2 + 1]['loc'][3:6] = src_locs[src, :]
            # Find detector and store location
            det = int(requested_channels[ch_idx2, 1]) - 1
            info['chs'][ch_idx2 * 2]['loc'][6:9] = det_locs[det, :]
            info['chs'][ch_idx2 * 2 + 1]['loc'][6:9] = det_locs[det, :]
            # Store channel location
            # Channel locations for short channels are bodged,
            # for short channels use the source location and add small offset
            # TODO: once coord system known then make offset 8mm
            if (has_short > 0) & (len(np.where(short_det == det + 1)[0]) > 0):
                info['chs'][ch_idx2 * 2]['loc'][:3] = src_locs[src, :]
                info['chs'][ch_idx2 * 2 + 1]['loc'][:3] = src_locs[src, :]
                info['chs'][ch_idx2 * 2]['loc'][0] += 0.3
                info['chs'][ch_idx2 * 2 + 1]['loc'][0] += 0.3
            else:
                info['chs'][ch_idx2 * 2]['loc'][:3] = ch_locs[ch_idx2, :]
                info['chs'][ch_idx2 * 2 + 1]['loc'][:3] = ch_locs[ch_idx2, :]

        # Read triggers from event file
        # TODO: where in the raw structure does this go?
        # t = [re.findall(r'(\d+)', line) for line in open(files['evt'])]
        # for t_idx in range(len(t)):
            # binary_value =''.join(t[t_idx][1:])[::-1]
            # trigger_value = int(binary_value, 2) * 1.
            # trigger_frame = float(t[t_idx][0])
            # trigger_time = (trigger_frame) * (1.0 / samplingrate)

        # Store the subset of sources and detectors requested by user
        # The signals between all source-detectors are stored even if they
        # are meaningless, the user pre specifies which combinations are
        # meaningful
        # TODO: Is this style of info overloading allowed?
        #       Currently I am marking all things as fnirs_ to make easier to
        #       find and remove if there is more appropriate storage locations
        raw_extras = {"sd_index": req_ind, 'files': files}

        super(RawNIRX, self).__init__(
            info, preload, filenames=[fname], last_samps=[last_sample],
            raw_extras=[raw_extras], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        The NIRX machine records raw data as two different wavelengths.
        These are stored in two files [wl1, wl2]. This function will
        return the data as ([wl1, wl2] x samples)
        """
        # Temporary solution until I write a reader
        # TODO: Write space separated values file reader
        pd = _check_pandas_installed(strict=True)

        file_wl1 = self._raw_extras[fi]['files']['wl1']
        file_wl2 = self._raw_extras[fi]['files']['wl2']

        sdindex = self._raw_extras[fi]['sd_index']

        wl1 = pd.read_csv(file_wl1, sep=' ', header=None).values
        wl2 = pd.read_csv(file_wl2, sep=' ', header=None).values

        wl1 = wl1[start:stop, sdindex].T
        wl2 = wl2[start:stop, sdindex].T

        # Currently saving the two wavelengths in same dimension
        # this seems like a bad idea
        # TODO: Can mne return (num_wavelengths x num_channels x num_samples)?
        data[0::2, :] = wl1
        data[1::2, :] = wl2

        return data

    def _probe_distances(self):
        """Return the distance between each source-detector pair."""
        # TODO: Write my own euclidean distance function
        from scipy.spatial.distance import euclidean
        dist = [euclidean(self.info['chs'][idx]['loc'][3:6],
                self.info['chs'][idx]['loc'][6:9])
                for idx in range(len(self.info['chs']))]
        return np.array(dist, float)

    def _short_channels(self, threshold=10.0):
        """Return a vector indicating which channels are short.

        Channels with distance less than `threshold` are reported as short.
        """
        return self._probe_distances() < threshold
