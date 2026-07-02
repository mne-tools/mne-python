# Author: Vittorio Pizzella <vittorio.pizzella@unich.it>
#
# License: BSD (3-clause)

import numpy as np

from ..._fiff.utils import _mult_cal_one
from ...utils import (
    _check_fname,
    fill_doc,
    verbose,
)
from ..base import BaseRaw
from .info import _mhd2info
from .mhd import _read_mhd


class RawITAB(BaseRaw):
    """Raw object from ITAB directory.

    Parameters
    ----------
    fname : str
        The raw file to load. Filename should end with *.raw
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):

        filenames = list()
        filenames.append(fname)

        fname = _check_fname(fname, overwrite="read", must_exist=True)

        fname_mhd = fname.with_name(fname.name + ".mhd")
        try:
            mhd = _read_mhd(fname_mhd)  # Read the mhd file
        except FileNotFoundError:
            raise ValueError(".mhd file not found")

        info = _mhd2info(mhd)

        orig_units = {ch["label"]: ch["unit"] for ch in mhd["ch"]}

        raw_extras = list()
        for fi, _ in enumerate(filenames):
            raw_extras.append(dict())
            for k in ["n_samp", "start_data", "units"]:
                raw_extras[fi][k] = info["temp"][k]

            raw_extras[fi]["nchan"] = info["nchan"]
            raw_extras[fi]["buffer_size_sec"] = info["temp"]["n_samp"] / info["sfreq"]
            raw_extras[fi]["data_type"] = mhd["data_type"]

        self.info = info
        info._check_consistency()

        first_samps = [0]
        last_samps = [info["temp"]["n_samp"] - 1]

        annotations = info["temp"]["annotations"]

        # Remove extras from info
        del info["temp"]

        super().__init__(
            info,
            preload,
            first_samps=first_samps,
            last_samps=last_samps,
            raw_extras=raw_extras,
            filenames=filenames,
            orig_units=orig_units,
            verbose=verbose,
        )

        self.set_annotations(annotations)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.
        Only needs to be implemented for readers that support
        ``preload=False``.

        Parameters
        ----------
        data : ndarray, shape (len(idx), stop - start + 1)
            The data array. Should be modified inplace.
        idx : ndarray | slice
            The requested channel indices.
        fi : int
            The file index that must be read from.
        start : int
            The start sample in the given file.
        stop : int
            The stop sample in the given file (inclusive).
        cals : ndarray, shape (len(idx), 1)
            Channel calibrations (already sub-indexed).
        mult : ndarray, shape (len(idx), len(info['chs']) | None
            The compensation + projection + cals matrix, if applicable.
        """
        # Initial checks
        start = int(start)
        if stop is None or stop > self._raw_extras[fi]["n_samp"]:
            stop = self._raw_extras[fi]["n_samp"]

        if start >= stop:
            raise ValueError("No data in this range")

        data_offset = self._raw_extras[fi]["start_data"]
        data_size = 4  # sizeof(int)
        n_channels = self._raw_extras[fi]["nchan"]

        data_left = (stop - start) * n_channels

        blocksize = ((int(100e6) // data_size) // n_channels) * n_channels
        blocksize = min(data_left, blocksize)

        with open(self._filenames[fi], "rb") as fid:
            # position  file pointer
            fid.seek(data_offset + data_size * start * n_channels, 0)

            for sample_start in np.arange(0, data_left, blocksize) // n_channels:
                count = min(blocksize, data_left - sample_start * n_channels)
                block = np.fromfile(fid, ">i4", count=count)

                if self._raw_extras[fi]["data_type"] == 4:
                    block = block.byteswap()
                block = block.astype(np.float32)  # convert to float32
                block = block.reshape(n_channels, -1, order="F")

                n_samples = block.shape[1]
                sample_stop = sample_start + n_samples

                data_view = data[:, sample_start:sample_stop]
                _mult_cal_one(data_view, block, idx, cals, mult)

        return data_view


@fill_doc
def read_raw_itab(fname, preload=False, verbose=None) -> RawITAB:
    """Raw object from ITAB directory

    Parameters
    ----------
    fname : str
        The raw file to load. Filename should end with *.raw
    preload : bool or str (default False)
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, preload is the
        file name of a memory-mapped file which is used to store the data
        on the hard drive (slower, requires less memory).
    %(verbose)s

    Returns
    -------
    raw : instance of RawITAB
        The raw data.

    See Also
    --------
    mne.io.Raw : Documentation of attribute and methods.

    Notes
    -----
    .. versionadded:: 0.01
    """
    return RawITAB(fname, preload=preload, verbose=verbose)
