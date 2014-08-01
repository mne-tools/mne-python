# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import numpy as np

from .utils import logger, verbose


@verbose
def read_dip(fname, verbose=None):
    """Read .dip file from Neuromag/xfit or MNE

    Parameters
    ----------
    fname : str
        The name of the .dip file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    time : array, shape=(n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape=(n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape=(n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape=(n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape=(n_dipoles,)
        The goodness of fit
    """
    try:
        data = np.loadtxt(fname, comments='%')
    except:
        data = np.loadtxt(fname, comments='#')  # handle 2 types of comments...
    if data.ndim == 1:
        data = data[None, :]
    logger.info("%d dipole(s) found" % len(data))
    time = data[:, 0]
    pos = 1e-3 * data[:, 2:5]  # put data in meters
    amplitude = data[:, 5]
    ori = data[:, 6:9]
    gof = data[:, 9]
    return time, pos, amplitude, ori, gof
