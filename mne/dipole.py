# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import re

from .utils import logger, verbose, deprecated


class Dipole(dict):
    """Dipole class

    Used to store positions, orientations, amplitudes, times, goodness of fit
    of dipoles, typically obtained with Neuromag/xfit, mne_dipole_fit
    or certain inverse solvers.

    Parameters
    ----------
    times : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
        The goodness of fit
    """
    def __init__(self, times, pos, amplitude, ori, gof, name=None):
        self['times'] = times
        self['pos'] = pos
        self['amplitude'] = amplitude
        self['ori'] = ori
        self['gof'] = gof
        if name is not None:
            self['name'] = name

    def __repr__(self):
        s = "n_times : %s" % len(self['times'])
        s += ", tmin : %s" % np.min(self['times'])
        s += ", tmax : %s" % np.max(self['times'])
        return "<Dipole  |  %s>" % s

    def save(self, fname):
        """Save dipole in a .dip file

        Parameters
        ----------
        fname : str
            The name of the .dip file.
        """
        fmt = "  %7.1f %7.1f %8.2f %8.2f %8.2f %8.3f %8.3f %8.3f %8.3f %6.1f"
        with open(fname, 'wb') as fid:
            fid.write('# CoordinateSystem "Head"\n'.encode('utf-8'))
            fid.write('#   begin     end   X (mm)   Y (mm)   Z (mm)'
                      '   Q(nAm)  Qx(nAm)  Qy(nAm)  Qz(nAm)    g/%\n'
                      .encode('utf-8'))
            t = self['times'][:, np.newaxis]
            gof = self['gof'][:, np.newaxis]
            amp = self['amplitude'][:, np.newaxis]
            out = np.concatenate((t, t, self['pos'] / 1e-3, amp,
                                  self['ori'], gof), axis=-1)
            np.savetxt(fid, out, fmt=fmt)
            if self.get('name') is not None:
                fid.write(('## Name "%s dipoles" Style "Dipoles"'
                           % self['name']).encode('utf-8'))


@deprecated("'read_dip' will be removed in version 0.10, please use "
            "'read_dipole' instead")
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
    time : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
        The goodness of fit
    """
    dipole = read_dipole(fname)
    return (dipole['times'], dipole['pos'], dipole['amplitude'],
            dipole['ori'], dipole['gof'])


@verbose
def read_dipole(fname, verbose=None):
    """Read .dip file from Neuromag/xfit or MNE

    Parameters
    ----------
    fname : str
        The name of the .dip file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    time : array, shape (n_dipoles,)
        The time instants at which each dipole was fitted.
    pos : array, shape (n_dipoles, 3)
        The dipoles positions in meters
    amplitude : array, shape (n_dipoles,)
        The amplitude of the dipoles in nAm
    ori : array, shape (n_dipoles, 3)
        The dipolar moments. Amplitude of the moment is in nAm.
    gof : array, shape (n_dipoles,)
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
    times = data[:, 0]
    pos = 1e-3 * data[:, 2:5]  # put data in meters
    amplitude = data[:, 5]
    ori = data[:, 6:9]
    gof = data[:, 9]
    dipole = Dipole(times=times, pos=pos, amplitude=amplitude, ori=ori,
                    gof=gof, name=name)
    return dipole
