# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>

# License: BSD (3-clause)

import numpy as np

from ..utils import check_fname, _check_fname


def read_fine_calibration(fname):
    """Read fine calibration information from a .dat file.

    The fine calibration typically includes improved sensor locations,
    calibration coefficients, and gradiometer imbalance information.

    Parameters
    ----------
    fname : str
        The filename.

    Returns
    -------
    calibration : dict
        Fine calibration information.
    """
    # Read new sensor locations
    fname = _check_fname(fname, overwrite='read', must_exist=True)
    check_fname(fname, 'cal', ('.dat',))
    ch_names = list()
    locs = list()
    imb_cals = list()
    with open(fname, 'r') as fid:
        for line in fid:
            if line[0] in '#\n':
                continue
            vals = line.strip().split()
            if len(vals) not in [14, 16]:
                raise RuntimeError('Error parsing fine calibration file, '
                                   'should have 14 or 16 entries per line '
                                   'but found %s on line:\n%s'
                                   % (len(vals), line))
            # `vals` contains channel number
            ch_name = vals[0]
            if len(ch_name) in (3, 4):  # heuristic for Neuromag fix
                try:
                    ch_name = int(ch_name)
                except ValueError:  # something other than e.g. 113 or 2642
                    pass
                else:
                    ch_name = 'MEG' + '%04d' % ch_name
            ch_names.append(ch_name)
            # (x, y, z), x-norm 3-vec, y-norm 3-vec, z-norm 3-vec
            locs.append(np.array([float(x) for x in vals[1:13]]))
            # and 1 or 3 imbalance terms
            imb_cals.append([float(x) for x in vals[13:]])
    locs = np.array(locs)
    return dict(ch_names=ch_names, locs=locs, imb_cals=imb_cals)


def write_fine_calibration(fname, calibration):
    """Write fine calibration information to a .dat file.

    Parameters
    ----------
    fname : str
        The filename to write out.
    calibration : dict
        Fine calibration information.
    """
    fname = _check_fname(fname, overwrite=True)
    check_fname(fname, 'cal', ('.dat',))

    with open(fname, 'wb') as cal_file:
        for ci, chan in enumerate(calibration['ch_names']):
            # Write string containing 1) channel, 2) loc info, 3) calib info
            # with field widths (e.g., %.6f) chosen to match how Elekta writes
            # them out
            cal_line = np.concatenate([calibration['locs'][ci],
                                       calibration['imb_cals'][ci]]).round(6)
            cal_str = str(chan) + ' ' + ' '.join(map(lambda x: "%.6f" % x,
                                                     cal_line))

            cal_file.write((cal_str + '\n').encode('ASCII'))
