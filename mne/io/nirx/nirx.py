# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import configparser as cp
import re as re
import glob as glob
import pandas as pd
import numpy as np

from ..base import BaseRaw
from ..meas_info import create_info
from ...utils import logger, verbose, fill_doc


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

        assert_one(file_dat, "Should be one dat file")
        assert_one(file_evt, "Should be one evt file")
        assert_one(file_hdr, "Should be one hdr file")
        assert_one(file_inf, "Should be one inf file")
        assert_one(file_set, "Should be one set file")
        assert_one(file_tpl, "Should be one tpl file")
        assert_one(file_wl1, "Should be one wl1 file")
        assert_one(file_wl2, "Should be one wl2 file")
        assert_one(file_cfg, "Should be one config file")
        assert_one(file_mat, "Should be one mat file")

        # Read number of rows of wavelength data which corresponds to
        # number of samples
        last_sample = -2
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
        subject_info['last_name'] = inf['name'].split()[-1].replace("\"", "")
        subject_info['first_name'] = inf['name'].split()[0].replace("\"", "")
        subject_info['middle_name'] = inf['name'].split()[-2].replace("\"", "")
        subject_info['birthday'] = inf['age']
        subject_info['sex'] = inf['gender'].replace("\"", "")
        # Recode values
        if subject_info['sex'] in {'M', 'Male', '1'}:
            subject_info['sex'] = 1
        if subject_info['sex'] in {'F', 'Female', '2'}:
            subject_info['sex'] = 2
        # NIRStar does not record an id, or handedness by default

        # Read header file
        # This is a bit tricky as the header file isnt compliant with
        # the config specifications. So we need to remove all text
        # between comments before passing to config parser
        hdr_str = open(file_hdr[0]).read()
        hdr_str = re.sub('#.*?#', '', hdr_str, flags=re.DOTALL)
        hdr = cp.RawConfigParser()
        hdr.read_string(hdr_str)

        # Check that the file format version is supported
        assert (hdr['GeneralInfo']['NIRStar'] == "\"15.2\""), \
            "Only NIRStar version 15.2 is supported"

        # Parse required header fields

        # Extract source-detectors requested by user
        sources = [int(s) for s in re.findall(r'(\d)-\d:\d+',
                   hdr['DataStructure']['S-D-Key'])]
        detectors = [int(s) for s in re.findall(r'\d-(\d):\d+',
                     hdr['DataStructure']['S-D-Key'])]
        sdindex = [int(s) for s in re.findall(r'\d-\d:(\d+)',
                   hdr['DataStructure']['S-D-Key'])]
        assert (len(sources) == len(detectors)), \
            "Same amount of sources and detectors required"
        assert (len(sources) == len(sdindex)), \
            "Same amount of sources and keys required"

        # Create mne structure
        # TODO: ch_type is currently misc as I could not find appropriate
        #       other type, the hbo and hbr type are not relevant until the
        #       signal has been converted
        # TODO: nchan needs to be multiplied by two as we have two wavelengths
        #       per sensor should the underlying type be modified to
        #       support (wavelength x channels x data)?
        info = create_info(len(sources) * 2,
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
        info['fnirs_sources'] = np.asarray(sources)
        info['fnirs_detectors'] = np.asarray(detectors)
        info['fnirs_sdindex'] = np.asarray(sdindex)

        fnirs_wavelengths = [int(s) for s in
                             re.findall(r'(\d+)',
                             hdr['ImagingParameters']['Wavelengths'])]
        info['fnirs_wavelengths'] = np.asarray(fnirs_wavelengths)

        super(RawNIRX, self).__init__(
            info, preload, filenames=[fname], last_samps=[last_sample],
            raw_extras=[hdr], verbose=verbose)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.
        The NIRX machine records raw data as two different wavelengths.
        These are stored in two files [wl1, wl2]. This function will
        return the data as ([wl1, wl2] x samples)
        """

        file_wl1 = glob.glob(self.filenames[fi] + '/*.wl1')
        file_wl2 = glob.glob(self.filenames[fi] + '/*.wl2')

        sdindex = self.info['fnirs_sdindex'] - 1  # As idx is 1 based

        wl1 = pd.read_csv(file_wl1[0], sep=' ').values
        wl2 = pd.read_csv(file_wl2[0], sep=' ').values
        assert (wl1.shape == wl2.shape), \
            "Wavelength files should contain same amount of data"

        wl1 = wl1[start:stop, sdindex].T
        wl2 = wl2[start:stop, sdindex].T

        # Currently saving the two wavelengths in same dimension
        # this seems like a bad idea
        # TODO: Can mne return (num_wavelengths x num_channels x num_samples)?
        data[0:self.info['nchan'] // 2, :] = wl1
        data[self.info['nchan'] // 2:, :] = wl2

        return data


def assert_one(x, msg):
    if len(x) != 1:
        raise RuntimeError(msg + ', got %d' % (len(x),))
