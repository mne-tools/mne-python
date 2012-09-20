from ..preprocessing import maxfilter
from ..fiff.constants import FIFF
import numpy as np
from math import pi

def make_eeg_layout(info, output_file):
    """
    Parameters
    ----------
    info : (e.g., from raw.info)
        Has info['ch_names'] and info['dig'] including EEG channels
    output_file : string
        Filename to save the layout file to.
    """
    radius, origin_head, origin_device = maxfilter.fit_sphere_to_headshape(info)
    hsp = [p['r'] for p in info['dig'] if p['kind'] == FIFF.FIFFV_POINT_EEG]
    names = [p for p in info['ch_names'] if 'EEG' in p]
    if len(hsp) <= 1:
      raise ValueError('No non-reference EEG digitization points found')

    # Get rid of reference
    if not len(hsp) == len(names)+1:
      raise ValueError('Channel names don\'t match digitization values')

    # Project onto unit sphere
    hsp = np.array(hsp)
    hsp = hsp[1:,:]
    hsp = hsp - origin_head/1e3
    hsp = hsp / (np.sum(hsp**2,axis=-1)**(1./2)).reshape(-1,1)

    lat = np.arccos(hsp[:,1])
    lon = np.arctan(hsp[:,0] / hsp[:,2])

#    # View from above (can't get it to work properly)
#    lat0 = pi/2
#    lon0 = 90
#    c = np.arccos(np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon - lon0))
#    kp = c / np.sin(c)
#    x = 20*kp*np.cos(lat)*np.sin(lon - lon0)
#    y = 20*kp*(np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon - lon0))

    # Uncomment these lines if we want a naive projection
    x = 20*hsp[:,0]
    y = 20*hsp[:,1]

    outStr = '%8.02f %8.02f %8.02f %8.02f\n' % (x.min(), x.max(), y.min(), y.max())
    for ii in range(hsp.shape[0]):
      outStr = outStr + '%03.0f %8.02f %8.02f %8.02f %8.02f %s\n' % (ii, x[ii], y[ii], 5., 4., names[ii])

    f = open(output_file, 'w')
    f.write(outStr)
    f.close();
