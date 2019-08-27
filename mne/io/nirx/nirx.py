# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import configparser as cp
import re as re
import glob as glob
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

        # Check if required files exist

        file_dat = glob.glob(fname + '/*.dat')
        file_evt = glob.glob(fname + '/*.evt')
        file_hdr = glob.glob(fname + '/*.hdr')
        file_inf = glob.glob(fname + '/*.inf')
        file_set = glob.glob(fname + '/*.set')
        file_tpl = glob.glob(fname + '/*.tpl')
        file_wl1 = glob.glob(fname + '/*.wl1')
        file_wl2 = glob.glob(fname + '/*.wl2')
        file_cfg = glob.glob(fname + '/*config.txt')
        file_mat = glob.glob(fname + '/*probeInfo.mat')

        _assert_one(file_dat, "Should be one dat file")
        _assert_one(file_evt, "Should be one evt file")
        _assert_one(file_hdr, "Should be one hdr file")
        _assert_one(file_inf, "Should be one inf file")
        _assert_one(file_set, "Should be one set file")
        _assert_one(file_tpl, "Should be one tpl file")
        _assert_one(file_wl1, "Should be one wl1 file")
        _assert_one(file_wl2, "Should be one wl2 file")
        _assert_one(file_cfg, "Should be one config file")
        _assert_one(file_mat, "Should be one mat file")

        # Read number of rows of wavelength data which corresponds to
        # number of samples
        last_sample = -1
        for line in open(file_wl1[0]):
            last_sample += 1

        # Read participant information file

        inf = cp.ConfigParser(allow_no_value=True)
        inf.read(file_inf)
        inf = inf._sections['Subject Demographics']

        # mne requires specific fields in here
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
        with open(file_hdr[0]) as f:
            hdr_str = f.read()
        hdr_str = re.sub('#.*?#', '', hdr_str, flags=re.DOTALL)
        hdr = cp.RawConfigParser()
        hdr.read_string(hdr_str)

        # Check that the file format version is supported
        assert (hdr['GeneralInfo']['NIRStar'] == "\"15.2\""), \
            "Only NIRStar version 15.2 is supported"

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
        mat_data = read_mat(file_mat[0], uint16_codec=None)
        # Following values will be required by commented out to keep flake
        # happy until i put in locations via info['chs'][ii]['loc']
        # detector_location_labels =mat_data['probeInfo']['probes']['labels_d']
        # source_location_labels = mat_data['probeInfo']['probes']['labels_s']
        # num_sources = mat_data['probeInfo']['probes']['nSource0']
        # num_detectors = mat_data['probeInfo']['probes']['nDetector0']
        requested_channels = mat_data['probeInfo']['probes']['index_c']

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
                           hdr['ImagingParameters']['SamplingRate'],
                           ch_types='misc')
        info.update({'subject_info': subject_info})

        # Store the subset of sources and detectors requested by user
        # The signals between all source-detectors are stored even if they
        # are meaningless, the user pre specifies which combinations are
        # meaningful
        # TODO: Is this style of info overloading allowed?
        #       Currently I am marking all things as fnirs_ to make easier to
        #       find and remove if there is more appropriate storage locations
        raw_extras = {"sd_index": req_ind}

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

        file_wl1 = glob.glob(self.filenames[fi] + '/*.wl1')
        file_wl2 = glob.glob(self.filenames[fi] + '/*.wl2')

        sdindex = self._raw_extras[fi]['sd_index']

        wl1 = pd.read_csv(file_wl1[0], sep=' ', header=None).values
        wl2 = pd.read_csv(file_wl2[0], sep=' ', header=None).values

        wl1 = wl1[start:stop, sdindex].T
        wl2 = wl2[start:stop, sdindex].T

        # Currently saving the two wavelengths in same dimension
        # this seems like a bad idea
        # TODO: Can mne return (num_wavelengths x num_channels x num_samples)?
        data[0::2, :] = wl1
        data[1::2, :] = wl2

        return data


def _assert_one(x, msg):
    if len(x) != 1:
        raise RuntimeError(msg + ', got %d' % (len(x),))
