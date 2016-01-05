"""Read .eeg files
"""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ...utils import logger
from ..constants import FIFF
from .res4 import _make_ctf_name


_cardinal_dict = dict(nasion=FIFF.FIFFV_POINT_NASION,
                      lpa=FIFF.FIFFV_POINT_LPA, left=FIFF.FIFFV_POINT_LPA,
                      rpa=FIFF.FIFFV_POINT_RPA, right=FIFF.FIFFV_POINT_RPA)


def _read_eeg(directory):
    """Read the .eeg file"""
    # Missing file is ok
    fname = _make_ctf_name(directory, 'eeg', raise_error=False)
    if fname is None:
        logger.info('    Separate EEG position data file not present.')
        return
    eeg = dict(labels=list(), kinds=list(), ids=list(), rr=list(), np=0,
               assign_to_chs=True, coord_frame=FIFF.FIFFV_MNE_COORD_CTF_HEAD)
    with open(fname, 'rb') as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0:
                parts = line.decode('utf-8').split()
                if len(parts) != 5:
                    raise RuntimeError('Illegal data in EEG position file: %s'
                                       % line)
                r = np.array([float(p) for p in parts[2:]]) / 100.
                if (r * r).sum() > 1e-4:
                    label = parts[1]
                    eeg['labels'].append(label)
                    eeg['rr'].append(r)
                    id_ = _cardinal_dict.get(label.lower(), int(parts[0]))
                    if label.lower() in _cardinal_dict:
                        kind = FIFF.FIFFV_POINT_CARDINAL
                    else:
                        kind = FIFF.FIFFV_POINT_EXTRA
                    eeg['ids'].append(id_)
                    eeg['kinds'].append(kind)
                    eeg['np'] += 1
    logger.info('    Separate EEG position data file read.')
    return eeg
