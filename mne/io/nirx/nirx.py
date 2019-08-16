# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os.path as op
import glob as glob
import pandas as pd

from ..base import BaseRaw
from ..utils import _read_segments_file, _file_size
from ..meas_info import create_info
from ...utils import logger, verbose, warn, fill_doc

@fill_doc
def read_raw_nirx(fname, preload=False, verbose=None):
    """Reader for a NIRX fNIRS file.
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

        # Read wavelength data

        wl1 = pd.read_csv(file_wl1[0], sep = ' ' )
        wl2 = pd.read_csv(file_wl2[0], sep = ' ' )
        assert (wl1.shape == wl2.shape), "Wavelength files should contain same amount of data"
