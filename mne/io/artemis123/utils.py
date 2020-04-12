import numpy as np
import os.path as op
from .._digitization import _artemis123_read_pos
from ...utils import logger
from ...transforms import rotation3d_align_z_axis


def _load_mne_locs(fname=None):
    """Load MNE locs structure from file (if exists) or recreate it."""
    if (not fname):
        # find input file
        resource_dir = op.join(op.dirname(op.abspath(__file__)), 'resources')
        fname = op.join(resource_dir, 'Artemis123_mneLoc.csv')

    if not op.exists(fname):
        raise IOError('MNE locs file "%s" does not exist' % (fname))

    logger.info('Loading mne loc file {}'.format(fname))
    locs = dict()
    with open(fname, 'r') as fid:
        for line in fid:
            vals = line.strip().split(',')
            locs[vals[0]] = np.array(vals[1::], np.float)

    return locs


def _generate_mne_locs_file(output_fname):
    """Generate mne coil locs and save to supplied file."""
    logger.info('Converting Tristan coil file to mne loc file...')
    resource_dir = op.join(op.dirname(op.abspath(__file__)), 'resources')
    chan_fname = op.join(resource_dir, 'Artemis123_ChannelMap.csv')
    chans = _load_tristan_coil_locs(chan_fname)

    # compute a dict of loc structs
    locs = {n: _compute_mne_loc(cinfo) for n, cinfo in chans.items()}

    # write it out to output_fname
    with open(output_fname, 'w') as fid:
        for n in sorted(locs.keys()):
            fid.write('%s,' % n)
            fid.write(','.join(locs[n].astype(str)))
            fid.write('\n')


def _load_tristan_coil_locs(coil_loc_path):
    """Load the Coil locations from Tristan CAD drawings."""
    channel_info = dict()
    with open(coil_loc_path, 'r') as fid:
        # skip 2 Header lines
        fid.readline()
        fid.readline()
        for line in fid:
            line = line.strip()
            vals = line.split(',')
            channel_info[vals[0]] = dict()
            if vals[6]:
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
    R = rotation3d_align_z_axis(z_axis)
    loc[3:13] = R.T.reshape(9)
    return loc


def _read_pos(fname):
    """Read the .pos file and return positions as dig points."""
    nas, lpa, rpa, hpi, extra = None, None, None, None, None
    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0:
                parts = line.split()
                # The lines can have 4 or 5 parts. First part is for the id,
                # which can be an int or a string. The last three are for xyz
                # coordinates. The extra part is for additional info
                # (e.g. 'Pz', 'Cz') which is ignored.
                if len(parts) not in [4, 5]:
                    continue

                if parts[0].lower() == 'nasion':
                    nas = np.array([float(p) for p in parts[-3:]]) / 100.
                elif parts[0].lower() == 'left':
                    lpa = np.array([float(p) for p in parts[-3:]]) / 100.
                elif parts[0].lower() == 'right':
                    rpa = np.array([float(p) for p in parts[-3:]]) / 100.
                elif 'hpi' in parts[0].lower():
                    if hpi is None:
                        hpi = list()
                    hpi.append(np.array([float(p) for p in parts[-3:]]) / 100.)
                else:
                    if extra is None:
                        extra = list()
                    extra.append(np.array([float(p)
                                           for p in parts[-3:]]) / 100.)

    return _artemis123_read_pos(nas, lpa, rpa, hpi, extra)
