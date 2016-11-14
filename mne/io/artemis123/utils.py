import numpy as np
import os.path as op
import inspect
from ...utils import logger


def _load_mne_locs():
    """Load MNE locs stucture from file (if exists) or recreate it."""
    # find input file
    FILE = inspect.getfile(inspect.currentframe())
    resource_dir = op.join(op.dirname(op.abspath(FILE)), 'resources')
    loc_fname = op.join(resource_dir, 'Artemis123_mneLoc.csv')
    if not op.exists(loc_fname):
        raise IOError('MNE locs file "%s" does not exist' % (loc_fname))

    logger.info('Loading precomputed mne loc file...')
    locs = dict()
    with open(loc_fname, 'r') as fid:
        for line in fid:
            vals = line.strip().split(',')
            locs[vals[0]] = np.array(vals[1::], np.float)

    return locs


def _generate_mne_locs_file(output_fname):
    """Generate mne coil locs and save to supplied file."""
    logger.info('Converting Tristan coil file to mne loc file...')
    FILE = inspect.getfile(inspect.currentframe())
    resource_dir = op.join(op.dirname(op.abspath(FILE)), 'resources')
    chan_fname = op.join(resource_dir, 'Artemis123_ChannelMap.csv')
    chans = _load_tristan_coil_locs(chan_fname)

    # compute a dict of loc structs
    locs = {n: _compute_mne_loc(cinfo) for n, cinfo in chans.items()}

    # write it out to loc_fname
    with open(output_fname, 'w') as fid:
        for n in sorted(locs.keys()):
            fid.write('%s,' % n)
            fid.write(','.join(locs[n].astype(str)))
            fid.write('\n')


def _load_tristan_coil_locs(coil_loc_path):
    """Load the Coil locations from Tristan CAD drawings."""
    channel_info = dict()
    with open(coil_loc_path, 'r') as fid:
        # skip 2 lines
        fid.readline()
        fid.readline()
        for line in fid:
            line = line.strip()
            vals = line.split(',')
            channel_info[vals[0]] = dict()
            if (vals[6]):
                channel_info[vals[0]]['inner_coil'] = \
                    np.array(vals[2:5], np.float)
                channel_info[vals[0]]['outer_coil'] = \
                    np.array(vals[5:8], np.float)
            else:  # nothing supplied
                channel_info[vals[0]]['inner_coil'] = np.zeros(3)
                channel_info[vals[0]]['outer_coil'] = np.zeros(3)
    return channel_info


def _compute_mne_loc(coil_loc):
    """Convert a set of coils to an mne Struct.

    Note input coil locations are in inches.
    """
    loc = np.zeros((12))
    if (np.linalg.norm(coil_loc['inner_coil']) == 0) and \
       (np.linalg.norm(coil_loc['outer_coil']) == 0):
        return loc

    # channel location is inner coil location converted to meters From inches
    loc[0:3] = coil_loc['inner_coil'] / 39.370078

    # figure out rotation
    z_axis = coil_loc['outer_coil'] - coil_loc['inner_coil']
    R = _compute_rot(z_axis)
    loc[3:13] = R.T.reshape(9)
    return loc


def _compute_rot(z_dev):
    """Compute a rotaion matrix to align channel z axis with device z axis.

    Input z_dev does not need to be normalized
    """
    z_dev = z_dev / np.linalg.norm(z_dev)
    f = 1 / (1 + z_dev[2])
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 1 * f * z_dev[0] * z_dev[0]
    R[0, 1] = -1 * f * z_dev[0] * z_dev[1]
    R[0, 2] = z_dev[0]
    R[1, 0] = -1 * f * z_dev[0] * z_dev[1]
    R[1, 1] = 1 - 1 * f * z_dev[1] * z_dev[1]
    R[1, 2] = z_dev[1]
    R[2, 0] = -z_dev[0]
    R[2, 1] = -z_dev[1]
    R[2, 2] = 1 - f * (z_dev[0] * z_dev[0] + z_dev[1] * z_dev[1])

    # this assertion could be stricter
    assert(np.linalg.norm(z_dev - R.dot([0, 0, 1])) < 1e-6)
    return R
