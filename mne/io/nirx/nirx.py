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

        assert (len(file_dat) == 1), "Should be one dat file"
        assert (len(file_evt) == 1), "Should be one evt file"
        assert (len(file_hdr) == 1), "Should be one hdr file"
        assert (len(file_inf) == 1), "Should be one inf file"
        assert (len(file_set) == 1), "Should be one set file"
        assert (len(file_tpl) == 1), "Should be one tpl file"
        assert (len(file_wl1) == 1), "Should be one wl1 file"
        assert (len(file_wl2) == 1), "Should be one wl2 file"
        assert (len(file_cfg) == 1), "Should be one config file"
        assert (len(file_mat) == 1), "Should be one mat file"

        # Read number of rows of wavelength data which corresponds to
        # number of samples
        last_sample = -2
        for line in open(file_wl1[0]):
            last_sample += 1

        # Read demographic information file

        inf = cp.ConfigParser()
        inf.read(file_inf)

        # Read header file

        hdr = cp.ConfigParser(allow_no_value=True)
        hdr.read(file_hdr)

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
