import numpy as np
from .base import _mult_cal_one


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
