from ..preprocessing import maxfilter
from ..fiff.constants import FIFF
from ..fiff import pick_types
import numpy as np

def make_eeg_layout(info, output_file, radius=20, width=5, height=4):
    """Create .lout file from EEG electrode digitization

    Parameters
    ----------
    info : dict
        Measurement info (e.g., raw.info)
    output_file : string
        Filename to save the layout file to.
    radius: float
        Viewport radius
    width: float
        Viewport width
    height: float
        Viewport height
    """
    radius_head, origin_head, origin_device = maxfilter.fit_sphere_to_headshape(info)
    inds = pick_types(info, meg=False, eeg=True)
    hsp = [info['chs'][ii]['eeg_loc'][:,0] for ii in inds]
    names = [info['chs'][ii]['ch_name'] for ii in inds]
    if len(hsp) <= 0:
        raise ValueError('No EEG digitization points found')

    if not len(hsp) == len(names):
        raise ValueError('Channel names don\'t match digitization values')
    hsp = np.array(hsp)

    # Move points to origin
    hsp -= origin_head/1e3 # convert to millimeters

    # Calculate angles
    r = np.sqrt(np.sum(hsp ** 2,axis=-1))
    theta = np.arccos(hsp[:,2]/r)
    phi = np.arctan2(hsp[:,1], hsp[:,0])

    # Mark the points that might have caused bad angle estimates
    iffy = np.nonzero(np.sum(hsp[:,:2]**2,axis=-1)**(1./2) < np.finfo(np.float).eps*10)
    theta[iffy] = 0
    phi[iffy] = 0

    # Do the azimuthal equidistant projection
    x = radius*(2.0*theta/np.pi)*np.cos(phi)
    y = radius*(2.0*theta/np.pi)*np.sin(phi)

    outStr = '%8.2f %8.2f %8.2f %8.2f\n' % (x.min() - 0.1*width, x.max() + 1.1*width, y.min() - 0.1*width, y.max() + 1.1*height)
    for ii in range(hsp.shape[0]):
        outStr = outStr + '%03d %8.2f %8.2f %8.2f %8.2f %s\n' % (ii+1, x[ii], y[ii], width, height, names[ii])

    f = open(output_file, 'w')
    f.write(outStr)
    f.close();
