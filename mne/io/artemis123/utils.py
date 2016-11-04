import numpy as np
import os.path as op
import inspect
from ...utils import logger

def _load_mne_locs():
    #find input file
    FILE = inspect.getfile(inspect.currentframe())
    resource_dir = op.join(op.dirname(op.abspath(FILE)), 'resources')
    locFname = op.join(resource_dir, 'Artemis123_mneLoc.csv')
    if op.exists(locFname):
        logger.info('Loading precomputed mne loc file...')
        locs = dict()
        with open(locFname, 'r') as fid:
            for line in fid:
                vals = line.strip().split(',')
                locs[vals[0]] = np.array(vals[1::],np.float)

    else:
        logger.info('Converting Tristan coil file to mne loc file...')
        chanFname = op.join(resource_dir, 'Artemis123_ChannelMap.csv')
        chans = _load_tristanCoilLocs(chanFname)
        #compute a dict of loc structs
        locs = {n:_compute_mne_loc(cinfo) for n,cinfo in chans.items()}

        #write it out to locFname
        with open(locFname, 'w') as fid:
            for n in sorted(locs.keys()):
                fid.write('%s,'%n)
                fid.write(','.join(locs[n].astype(str)))
                fid.write('\n')
    return locs

def _load_tristanCoilLocs(coilLocPath):
    """Load the Coil locations from Tristan CAD drawings."""
    channelInfo = dict()
    with open(coilLocPath, 'r') as fid:
        #skip 2 lines
        fid.readline()
        fid.readline()
        for line in fid:
            line = line.strip()
            vals = line.split(',')
            channelInfo[vals[0]] = dict();
            if (vals[6]):
                channelInfo[vals[0]]['innerCoil'] = np.array(vals[1:4],np.float)
                channelInfo[vals[0]]['outerCoil'] = np.array(vals[4:7],np.float)
            else: #nothing supplied
                channelInfo[vals[0]]['innerCoil'] = np.zeros(3)
                channelInfo[vals[0]]['outerCoil'] = np.zeros(3)
    return channelInfo

# double(12)	The channel location. The first three numbers indicate the location [m], followed by the three unit vectors of the channel-specific coordinate frame.
def _compute_mne_loc(coilLoc):
    """Convert a set of coils to an mne Struct
    Note input coil locations are in inches."""
    loc = np.zeros((12))
    if  (np.linalg.norm(coilLoc['innerCoil']) == 0) and \
        (np.linalg.norm(coilLoc['outerCoil']) == 0):
        return loc

    #channel location is the inner coil location converted to meters From inches
    loc[0:3] = coilLoc['innerCoil'] / 39.370078

    #figure out rotation
    zAxis = coilLoc['outerCoil']  - coilLoc['innerCoil']
    R = _compute_rot(zAxis)
    loc[3:13] = R.T.reshape(9)
    return loc

def _compute_rot(zAxis_dev):
    """Compute a rotaion matrix to align channel coord zAxis (0,0,1)
    with the supplied device coordinate zAxis
    Input zAxis Does not need to be normalized"""

    zAxis_dev = zAxis_dev / np.linalg.norm(zAxis_dev)
    f = 1 / (1+zAxis_dev[2])
    R = np.zeros((3,3))
    R[0,0] = 1 -1 * f * zAxis_dev[0] * zAxis_dev[0]
    R[0,1] =   -1 * f * zAxis_dev[0] * zAxis_dev[1]
    R[0,2] =   zAxis_dev[0]
    R[1,0] =   -1 * f * zAxis_dev[0] * zAxis_dev[1]
    R[1,1] = 1 -1 * f * zAxis_dev[1] * zAxis_dev[1]
    R[1,2] =   zAxis_dev[1]
    R[2,0] = -zAxis_dev[0]
    R[2,1] = -zAxis_dev[1]
    R[2,2] = 1 -f*(zAxis_dev[0] * zAxis_dev[0] + zAxis_dev[1] * zAxis_dev[1])
    #this assertion could be stricter
    assert(np.linalg.norm(zAxis_dev - R.dot([0,0,1])) < 1e-6)
    return R
