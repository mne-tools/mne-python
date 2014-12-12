# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from scipy import optimize


def _fit_sphere(points):
    """Aux function to fit points to a sphere"""
    # initial guess for center and radius
    xradius = (np.max(points[:, 0]) - np.min(points[:, 0])) / 2
    yradius = (np.max(points[:, 1]) - np.min(points[:, 1])) / 2

    radius_init = (xradius + yradius) / 2
    center_init = np.array([0.0, 0.0, np.max(points[:, 2]) - radius_init])

    # optimization
    x0 = np.r_[center_init, radius_init]

    def cost_fun(x, points):
        return np.sum((np.sqrt(np.sum((points - x[:3]) ** 2, axis=1)) -
                      x[3]) ** 2)

    x_opt = optimize.fmin_powell(cost_fun, x0, args=(points,),
                                 disp=True)

    origin = x_opt[:3]
    radius = x_opt[3]
    return radius, origin
