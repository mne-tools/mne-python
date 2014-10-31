# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os
import os.path as op
import numpy as np

from ..externals.six import BytesIO
from ..channels import _contains_ch_type
from ..viz import plot_montage


class Montage(object):
    """Montage for EEG cap

    Montages are typically loaded from a file using read_montage. Only use this
    class directly if you're constructing a new montage.

    Parameters
    ----------
    pos : array, shape (n_channels, 3)
        The positions of the channels in 3d.
    ch_names : list
        The channel names.
    kind : str
        The type of montage (e.g. 'standard_1005').
    selection : array of int
        The indices of the selected channels in the montage file.
    """
    def __init__(self, pos, ch_names, kind, selection):
        self.pos = pos
        self.ch_names = ch_names
        self.kind = kind
        self.selection = selection

    def __repr__(self):
        s = '<Montage | %s - %d Channels: %s ...>'
        s %= self.kind, len(self.ch_names), ', '.join(self.ch_names[:3])
        return s

    def plot(self, scale_factor=1.5, show_names=False):
        """Plot EEG sensor montage

        Parameters
        ----------
        scale_factor : float
            Determines the size of the points. Defaults to 1.5
        show_names : bool
            Whether to show the channel names. Defaults to False

        Returns
        -------
        fig : isntance of mayavi.Scene
            The malab scene object.
        """
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names)


def read_montage(kind, ch_names=None, path=None, scale=True):
    """Read montage from a file

    Parameters
    ----------
    kind : str
        The name of the montage file (e.g. kind='easycap-M10' for
        'easycap-M10.txt'). Files with extensions '.elc', '.txt', '.csd'
        or '.sfp' are supported.
    ch_names : list of str
        The names to read. If None, all names are returned.
    path : str | None
        The path of the folder containing the montage file
    scale : bool
        Apply useful scaling for out the box plotting using montage.pos

    Returns
    -------
    montage : instance of Montage
        The montage.
    """
    if path is None:
        path = op.dirname(__file__)
    if not op.isabs(kind):
        supported = ('.elc', '.txt', '.csd', '.sfp')
        montages = [op.splitext(f) for f in os.listdir(path)]
        montages = [m for m in montages if m[1] in supported and kind == m[0]]
        if len(montages) != 1:
            raise ValueError('Could not find the montage. Please provide the '
                             'full path.')
        kind, ext = montages[0]
        fname = op.join(path, kind + ext)
    else:
        kind, ext = op.splitext(kind)
        fname = op.join(path, kind + ext)

    if ext == '.sfp':
        # EGI geodesic
        dtype = np.dtype('S4, f8, f8, f8')
        data = np.loadtxt(fname, dtype=dtype)
        pos = np.c_[data['f1'], data['f2'], data['f3']]
        ch_names_ = data['f0']
    elif ext == '.elc':
        # 10-5 system
        ch_names_ = []
        pos = []
        with open(fname) as fid:
            for line in fid:
                if 'Positions\n' in line:
                    break
            for line in fid:
                if 'Labels\n' in line:
                    break
                pos.append(line)
            for line in fid:
                if not line or not set(line) - set([' ']):
                    break
                ch_names_.append(line.strip(' ').strip('\n'))
        pos = np.loadtxt(BytesIO(''.join(pos)))
    elif ext == '.txt':
        # easycap
        dtype = np.dtype('S4, f8, f8')
        data = np.loadtxt(fname, dtype=dtype, skiprows=1)
        theta, phi = data['f1'], data['f2']
        x = 85. * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
        y = 85. * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = 85. * np.cos(np.deg2rad(theta))
        pos = np.c_[x, y, z]
        ch_names_ = data['f0']
    elif ext == '.csd':
        # CSD toolbox
        dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                 ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                 ('off_sph', 'f8')]
        table = np.loadtxt(fname, skiprows=2, dtype=dtype)
        pos = np.c_[table['x'], table['y'], table['z']]
        ch_names_ = table['label']
    else:
        raise ValueError('Currently the "%s" template is not supported.' %
                         kind)
    selection = np.arange(len(pos))
    if ch_names is not None:
        sel, ch_names_ = zip(*[(i, e) for i, e in enumerate(ch_names_)
                            if e in ch_names])
        sel = list(sel)
        pos = pos[sel]
        selection = selection[sel]
    else:
        ch_names_ = list(ch_names_)
    kind = op.split(kind)[-1]
    return Montage(pos=pos, ch_names=ch_names_, kind=kind, selection=selection)


def apply_montage(info, montage):
    """Apply montage to EEG data.

    This function will replace the EEG channel names and locations with
    the values specified for the particular montage.

    Note: You have to rename your object to correclty map
    the montage names.

    Note: This function will change the info variable in place.

    Parameters
    ----------
    info : instance of Info
        The measurement info to update.
    montage : instance of Montage
        The montage to apply.
    """
    if not _contains_ch_type(info, 'eeg'):
        raise ValueError('No EEG channels found.')

    sensors_found = False
    for pos, ch_name in zip(montage.pos, montage.ch_names):
        if ch_name not in info['ch_names']:
            continue

        ch_idx = info['ch_names'].index(ch_name)
        info['ch_names'][ch_idx] = ch_name
        info['chs'][ch_idx]['eeg_loc'] = np.c_[pos, [0.] * 3]
        info['chs'][ch_idx]['loc'] = np.r_[pos, [0.] * 9]
        sensors_found = True

    if not sensors_found:
        raise ValueError('None of the sensors defined in the montage were '
                         'found in the info structure. Check the channel '
                         'names.')
