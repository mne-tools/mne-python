from ..preprocessing import maxfilter
from ..fiff.constants import FIFF
import numpy as np
from math import pi

def make_eeg_layout(info, output_file, prad=20, width=5, height=4):
    """
    Parameters
    ----------
    info : (e.g., from raw.info)
        Has info['ch_names'] and info['dig'] including EEG channels
    output_file : string
        Filename to save the layout file to.
    prad: float
        Viewport radius
    width: float
        Viewport width
    height: float
        Viewport height
    """
    radius, origin_head, origin_device = maxfilter.fit_sphere_to_headshape(info)
    hsp = [p['r'] for p in info['dig'] if p['kind'] == FIFF.FIFFV_POINT_EEG]
    names = [p for p in info['ch_names'] if 'EEG' in p]
    if len(hsp) <= 1:
      raise ValueError('No non-reference EEG digitization points found')

    # Get rid of reference
    if not len(hsp) == len(names)+1:
      raise ValueError('Channel names don\'t match digitization values')
    hsp = np.array(hsp)
    hsp = hsp[1:,:]

    # Move points to origin
    hsp = hsp - origin_head/1e3

    # Calculate angles
    r = np.sum(hsp**2,axis=-1)**(1./2)
    theta = np.arccos(hsp[:,2]/r)
    phi = np.arctan2(hsp[:,1], hsp[:,0])

    # Mark the points that might have caused bad angle estimates
    iffy = np.nonzero(np.sum(hsp[:,:2]**2,axis=-1)**(1./2) < np.finfo(np.float).eps*10)
    theta[iffy] = 0
    phi[iffy] = 0

    # Do the azimuthal equidistant projection
    x = prad*(2.0*theta/pi)*np.cos(phi);
    y = prad*(2.0*theta/pi)*np.sin(phi);

    outStr = '%8.2f %8.2f %8.2f %8.2f\n' % (x.min() - 0.1*width, x.max() + 1.1*width, y.min() - 0.1*width, y.max() + 1.1*height)
    for ii in range(hsp.shape[0]):
      outStr = outStr + '%03d %8.2f %8.2f %8.2f %8.2f %s\n' % (ii+1, x[ii], y[ii], width, height, names[ii])

    f = open(output_file, 'w')
    f.write(outStr)
    f.close();
