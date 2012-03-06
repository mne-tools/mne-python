# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy.optimize import fmin_powell

from .fiff.constants import FIFF


def fit_sphere_to_headshape(info):

    # get head digization points, excluding some frontal points (nose etc.)
    hsp = [p['r'] for p in info['dig'] if p['kind'] == FIFF.FIFFV_POINT_EXTRA
           and not (p['r'][2] < 0 and p['r'][1] > 0)]

    if len(hsp) == 0:
        raise ValueError('No head digitization points found')

    hsp = 1e3 * np.array(hsp)

    # initial guess for center and radius
    xradius = (np.max(hsp[:, 0]) - np.min(hsp[:, 0])) / 2
    yradius = (np.max(hsp[:, 1]) - np.min(hsp[:, 1])) / 2

    radius_init = (xradius + yradius) / 2
    center_init = np.array([0.0, 0.0, np.max(hsp[:, 2]) - radius_init])

    # optimization
    x0 = np.r_[center_init, radius_init]
    cost_fun = lambda x, hsp:\
        np.sum((np.sqrt(np.sum((hsp - x[:3]) ** 2, axis=1)) - x[3]) ** 2)

    x_opt = fmin_powell(cost_fun, x0, args=(hsp,))

    center = x_opt[:3]
    radius = x_opt[3]

    print ('Fitted sphere: r = %0.1f mm, center = %0.1f %0.1f %0.1f mm'
           % (radius, center[0], center[1], center[2]))

    return center, radius



