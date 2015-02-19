# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import re

from .utils import logger, verbose, deprecated


@deprecated("'read_dip' will be removed in version 0.10, please use "
            "'read_dipoles' instead")
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
    dipoles = read_dipoles(fname)
    return (dipoles['time'], dipoles['pos'], dipoles['amplitude'],
            dipoles['ori'], dipoles['gof'])


@verbose
def read_dipoles(fname, verbose=None):
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
    name = None
    with open(fname, 'r') as fid:
        for line in fid.readlines():
            if line.startswith('##') or line.startswith('%%'):
                m = re.search('Name "(.*) dipoles"', line)
                if m:
                    name = m.group(1)
                    break
    if data.ndim == 1:
        data = data[None, :]
    logger.info("%d dipole(s) found" % len(data))
    time = data[:, 0]
    pos = 1e-3 * data[:, 2:5]  # put data in meters
    amplitude = data[:, 5]
    ori = data[:, 6:9]
    gof = data[:, 9]
    dipoles = dict(time=time, pos=pos, amplitude=amplitude, ori=ori,
                   gof=gof, name=name)
    return dipoles


def write_dipoles(fname, dipoles, verbose=None):
    """Write a .dip file from a set of dipole definitions

    Parameters
    ----------
    fname : str
        The name of the .dip file.
    dipoles : dict
        The dipole definitions.
    """
    fmt = "  %7.1f %7.1f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %6.1f"
    with open(fname, 'w') as fid:
        fid.write('# CoordinateSystem "Head"\n')
        fid.write('#   begin     end   X (mm)   Y (mm)   Z (mm)'
                  '   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%\n')
        t = dipoles['time'][:, np.newaxis]
        gof = dipoles['gof'][:, np.newaxis]
        amp = dipoles['amplitude'][:, np.newaxis]
        out = np.concatenate((t, t, dipoles['pos'] / 1e-3, amp,
                              dipoles['ori'], gof), axis=-1)
        np.savetxt(fid, out, fmt=fmt)
        if dipoles.get('name') is not None:
            fid.write('## Name "%s dipoles" Style "Dipoles"' % dipoles['name'])
