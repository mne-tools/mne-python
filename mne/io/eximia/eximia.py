# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os.path as op

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _file_size, _read_segments_file
from ...utils import _check_fname, fill_doc, logger, verbose, warn
from ..base import BaseRaw


@fill_doc
def read_raw_eximia(fname, preload=False, verbose=None) -> "RawEximia":
    """Reader for an eXimia EEG file.

    Parameters
    ----------
    fname : path-like
        Path to the eXimia ``.nxe`` data file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawEximia
        A Raw object containing eXimia data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawEximia.
    """
    return RawEximia(fname, preload, verbose)


@fill_doc
class RawEximia(BaseRaw):
    """Raw object from an Eximia EEG file.

    Parameters
    ----------
    fname : path-like
        Path to the eXimia data file (.nxe).
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        fname = str(_check_fname(fname, "read", True, "fname"))
        data_name = op.basename(fname)
        logger.info(f"Loading {data_name}")
        # Create vhdr and vmrk files so that we can use mne_brain_vision2fiff
        n_chan = 64
        sfreq = 1450.0
        # data are multiplexed int16
        ch_names = ["GateIn", "Trig1", "Trig2", "EOG"]
        ch_types = ["stim", "stim", "stim", "eog"]
        cals = [
            0.0015259021896696422,
            0.0015259021896696422,
            0.0015259021896696422,
            0.3814755474174106,
        ]
        ch_names += (
            "Fp1 Fpz Fp2 AF1 AFz AF2 "
            "F7 F3 F1 Fz F2 F4 F8 "
            "FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 "
            "T7 C5 C3 C1 Cz C2 C4 C6 T8 "
            "TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10 "
            "P9 P7 P3 P1 Pz P2 P4 P8 "
            "P10 PO3 POz PO4 O1 Oz O2 Iz".split()
        )
        n_eeg = len(ch_names) - len(cals)
        cals += [0.07629510948348212] * n_eeg
        ch_types += ["eeg"] * n_eeg
        assert len(ch_names) == n_chan
        info = create_info(ch_names, sfreq, ch_types)
        n_bytes = _file_size(fname)
        n_samples, extra = divmod(n_bytes, (n_chan * 2))
        if extra != 0:
            warn(
                f"Incorrect number of samples in file ({n_samples}), the file is likely"
                " truncated"
            )
        for ch, cal in zip(info["chs"], cals):
            ch["cal"] = cal
        super().__init__(
            info,
            preload=preload,
            last_samps=(n_samples - 1,),
            filenames=[fname],
            orig_format="short",
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        _read_segments_file(self, data, idx, fi, start, stop, cals, mult, dtype="<i2")
