# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np


def _mult_cal_one(data_view, one, idx, cals, mult):
    """Take a chunk of raw data, multiply by mult or cals, and store"""
    one = np.asarray(one, dtype=data_view.dtype)
    assert data_view.shape[1] == one.shape[1]
    if mult is not None:
        data_view[:] = np.dot(mult, one)
    else:
        if isinstance(idx, slice):
            data_view[:] = one[idx]
        else:
            # faster than doing one = one[idx]
            np.take(one, idx, axis=0, out=data_view)
        if cals is not None:
            data_view *= cals


def _blk_read_lims(start, stop, buf_len):
    """Helper to deal with indexing in the middle of a data block

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

    """  # noqa
    # this is used to deal with indexing in the middle of a sampling period
    assert all(isinstance(x, int) for x in (start, stop, buf_len))
    block_start_idx = (start // buf_len)
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


def _read_segments_file(raw, data, idx, fi, start, stop, cals, mult,
                        dtype='<i2', n_bytes=2):
    """Read a chunk of raw data"""
    nchan = raw.info['nchan']
    data_offset = raw.info['nchan'] * start * n_bytes
    data_left = (stop - start) * nchan
    # Read up to 100 MB of data at a time.
    blk_size = min(data_left, ((100000000 // n_bytes) // nchan) * nchan)

    with open(raw._filenames[fi], 'rb', buffering=0) as fid:
        fid.seek(data_offset)
        # extract data in chunks
        for blk_start in np.arange(0, data_left, blk_size) // nchan:
            blk_size = min(blk_size, data_left - blk_start * nchan)
            block = np.fromfile(fid, dtype, blk_size)
            block = block.reshape(nchan, -1, order='F')
            blk_stop = blk_start + block.shape[1]
            data_view = data[:, blk_start:blk_stop]
            _mult_cal_one(data_view, block, idx, cals, mult)
    return data
