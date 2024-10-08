# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import os.path as op
from pathlib import Path

import numpy as np

from .constants import FIFF
from .meas_info import _get_valid_units


def _check_orig_units(orig_units):
    """Check original units from a raw file.

    Units that are close to a valid_unit but not equal can be remapped to fit
    into the valid_units. All other units that are not valid will be replaced
    with "n/a".

    Parameters
    ----------
    orig_units : dict
        Dictionary mapping channel names to their units as specified in
        the header file. Example: {'FC1': 'nV'}

    Returns
    -------
    orig_units_remapped : dict
        Dictionary mapping channel names to their VALID units as specified in
        the header file. Invalid units are now labeled "n/a".
        Example: {'FC1': 'nV', 'Hfp3erz': 'n/a'}
    """
    if orig_units is None:
        return
    valid_units = _get_valid_units()
    valid_units_lowered = [unit.lower() for unit in valid_units]
    orig_units_remapped = dict(orig_units)
    for ch_name, unit in orig_units.items():
        # Be lenient: we ignore case for now.
        if unit.lower() in valid_units_lowered:
            continue

        # Common "invalid units" can be remapped to their valid equivalent
        remap_dict = dict()
        remap_dict["uv"] = "µV"
        remap_dict["μv"] = "µV"  # greek letter mu vs micro sign. use micro
        remap_dict["\x83\xeav"] = "µV"  # for shift-jis mu, use micro
        if unit.lower() in remap_dict:
            orig_units_remapped[ch_name] = remap_dict[unit.lower()]
            continue

        # Some units cannot be saved, they are invalid: assign "n/a"
        orig_units_remapped[ch_name] = "n/a"

    return orig_units_remapped


def _find_channels(ch_names, ch_type="EOG"):
    """Find EOG channel."""
    substrings = (ch_type,)
    substrings = [s.upper() for s in substrings]
    if ch_type == "EOG":
        substrings = ("EOG", "EYE")
    eog_idx = [
        idx
        for idx, ch in enumerate(ch_names)
        if any(substring in ch.upper() for substring in substrings)
    ]
    return eog_idx


def _mult_cal_one(data_view, one, idx, cals, mult):
    """Take a chunk of raw data, multiply by mult or cals, and store."""
    one = np.asarray(one, dtype=data_view.dtype)
    assert data_view.shape[1] == one.shape[1], (
        data_view.shape[1],
        one.shape[1],
    )  # noqa: E501
    if mult is not None:
        assert mult.ndim == one.ndim == 2
        data_view[:] = mult @ one[idx]
    else:
        assert cals is not None
        if isinstance(idx, slice):
            data_view[:] = one[idx]
        else:
            # faster than doing one = one[idx]
            np.take(one, idx, axis=0, out=data_view)
        data_view *= cals


def _blk_read_lims(start, stop, buf_len):
    """Deal with indexing in the middle of a data block.

    Parameters
    ----------
    start : int
        Starting index.
    stop : int
        Ending index (exclusive).
    buf_len : int
        Buffer size in samples.

    Returns
    -------
    block_start_idx : int
        The first block to start reading from.
    r_lims : list
        The read limits.
    d_lims : list
        The write limits.

    Notes
    -----
    Consider this example::

        >>> start, stop, buf_len = 2, 27, 10

                    +---------+---------+---------
    File structure: |  buf0   |   buf1  |   buf2  |
                    +---------+---------+---------
    File time:      0        10        20        30
                    +---------+---------+---------
    Requested time:   2                       27

                    |                             |
                blockstart                    blockstop
                      |                        |
                    start                    stop

    We need 27 - 2 = 25 samples (per channel) to store our data, and
    we need to read from 3 buffers (30 samples) to get all of our data.

    On all reads but the first, the data we read starts at
    the first sample of the buffer. On all reads but the last,
    the data we read ends on the last sample of the buffer.

    We call ``this_data`` the variable that stores the current buffer's data,
    and ``data`` the variable that stores the total output.

    On the first read, we need to do this::

        >>> data[0:buf_len-2] = this_data[2:buf_len]  # doctest: +SKIP

    On the second read, we need to do::

        >>> data[1*buf_len-2:2*buf_len-2] = this_data[0:buf_len]  # doctest: +SKIP

    On the final read, we need to do::

        >>> data[2*buf_len-2:3*buf_len-2-3] = this_data[0:buf_len-3]  # doctest: +SKIP

    This function encapsulates this logic to allow a loop over blocks, where
    data is stored using the following limits::

        >>> data[d_lims[ii, 0]:d_lims[ii, 1]] = this_data[r_lims[ii, 0]:r_lims[ii, 1]]  # doctest: +SKIP

    """  # noqa: E501
    # this is used to deal with indexing in the middle of a sampling period
    assert all(isinstance(x, int) for x in (start, stop, buf_len))
    block_start_idx = start // buf_len
    block_start = block_start_idx * buf_len
    last_used_samp = stop - 1
    block_stop = last_used_samp - last_used_samp % buf_len + buf_len
    read_size = block_stop - block_start
    n_blk = read_size // buf_len + (read_size % buf_len != 0)
    start_offset = start - block_start
    end_offset = block_stop - stop
    d_lims = np.empty((n_blk, 2), int)
    r_lims = np.empty((n_blk, 2), int)
    for bi in range(n_blk):
        # Triage start (sidx) and end (eidx) indices for
        # data (d) and read (r)
        if bi == 0:
            d_sidx = 0
            r_sidx = start_offset
        else:
            d_sidx = bi * buf_len - start_offset
            r_sidx = 0
        if bi == n_blk - 1:
            d_eidx = stop - start
            r_eidx = buf_len - end_offset
        else:
            d_eidx = (bi + 1) * buf_len - start_offset
            r_eidx = buf_len
        d_lims[bi] = [d_sidx, d_eidx]
        r_lims[bi] = [r_sidx, r_eidx]
    return block_start_idx, r_lims, d_lims


def _file_size(fname):
    """Get the file size in bytes."""
    with open(fname, "rb") as f:
        f.seek(0, os.SEEK_END)
        return f.tell()


def _read_segments_file(
    raw,
    data,
    idx,
    fi,
    start,
    stop,
    cals,
    mult,
    dtype,
    n_channels=None,
    offset=0,
    trigger_ch=None,
):
    """Read a chunk of raw data."""
    if n_channels is None:
        n_channels = raw._raw_extras[fi]["orig_nchan"]

    n_bytes = np.dtype(dtype).itemsize
    # data_offset and data_left count data samples (channels x time points),
    # not bytes.
    data_offset = n_channels * start * n_bytes + offset
    data_left = (stop - start) * n_channels

    # Read up to 100 MB of data at a time, block_size is in data samples
    block_size = ((int(100e6) // n_bytes) // n_channels) * n_channels
    block_size = min(data_left, block_size)
    with open(raw.filenames[fi], "rb", buffering=0) as fid:
        fid.seek(data_offset)
        # extract data in chunks
        for sample_start in np.arange(0, data_left, block_size) // n_channels:
            count = min(block_size, data_left - sample_start * n_channels)
            block = np.fromfile(fid, dtype, count)
            if block.size != count:
                raise RuntimeError(
                    f"Incorrect number of samples ({block.size} != {count}), please "
                    "report this error to MNE-Python developers"
                )
            block = block.reshape(n_channels, -1, order="F")
            n_samples = block.shape[1]  # = count // n_channels
            sample_stop = sample_start + n_samples
            if trigger_ch is not None:
                stim_ch = trigger_ch[start:stop][sample_start:sample_stop]
                block = np.vstack((block, stim_ch))
            data_view = data[:, sample_start:sample_stop]
            _mult_cal_one(data_view, block, idx, cals, mult)


def read_str(fid, count=1):
    """Read string from a binary file in a python version compatible way."""
    dtype = np.dtype(f">S{count}")
    string = fid.read(dtype.itemsize)
    data = np.frombuffer(string, dtype=dtype)[0]
    bytestr = b"".join([data[0 : data.index(b"\x00") if b"\x00" in data else count]])

    return str(bytestr.decode("ascii"))  # Return native str type for Py2/3


def _create_chs(ch_names, cals, ch_coil, ch_kind, eog, ecg, emg, misc):
    """Initialize info['chs'] for eeg channels."""
    chs = list()
    for idx, ch_name in enumerate(ch_names):
        if ch_name in eog or idx in eog:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_EOG_CH
        elif ch_name in ecg or idx in ecg:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_ECG_CH
        elif ch_name in emg or idx in emg:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_EMG_CH
        elif ch_name in misc or idx in misc:
            coil_type = FIFF.FIFFV_COIL_NONE
            kind = FIFF.FIFFV_MISC_CH
        else:
            coil_type = ch_coil
            kind = ch_kind

        chan_info = {
            "cal": cals[idx],
            "logno": idx + 1,
            "scanno": idx + 1,
            "range": 1.0,
            "unit_mul": FIFF.FIFF_UNITM_NONE,
            "ch_name": ch_name,
            "unit": FIFF.FIFF_UNIT_V,
            "coord_frame": FIFF.FIFFV_COORD_HEAD,
            "coil_type": coil_type,
            "kind": kind,
            "loc": np.zeros(12),
        }
        if coil_type == FIFF.FIFFV_COIL_EEG:
            chan_info["loc"][:3] = np.nan
        chs.append(chan_info)
    return chs


def _construct_bids_filename(base, ext, part_idx, validate=True):
    """Construct a BIDS compatible filename for split files."""
    # insert index in filename
    dirname = op.dirname(base)
    base = op.basename(base)
    deconstructed_base = base.split("_")
    if len(deconstructed_base) < 2 and validate:
        raise ValueError(
            "Filename base must end with an underscore followed "
            f"by the modality (e.g., _eeg or _meg), got {base}"
        )
    suffix = deconstructed_base[-1]
    base = "_".join(deconstructed_base[:-1])
    use_fname = f"{base}_split-{part_idx + 1:02}_{suffix}{ext}"
    if dirname:
        use_fname = op.join(dirname, use_fname)
    return use_fname


def _make_split_fnames(fname, n_splits, split_naming):
    """Make a list of split filenames."""
    if n_splits == 1:
        fname = Path(fname)
        return [fname]
    res = []
    base, ext = op.splitext(fname)
    for i in range(n_splits):
        if split_naming == "neuromag":
            path = Path(f"{base}-{i:d}{ext}" if i else fname)
            res.append(path)
        else:
            assert split_naming == "bids"
            path = Path(_construct_bids_filename(base, ext, i))
            res.append(path)
    return res
