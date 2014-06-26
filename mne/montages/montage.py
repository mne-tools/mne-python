# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: Simplified BSD

import os
import os.path as op
import numpy as np

from ..externals.six import BytesIO
from ..channels import _contains_ch_type
from ..viz import plot_montage


class Montage(object):
    """Sensor layouts

    Montages are typically loaded from a file using read_montage. Only use this
    class directly if you're constructing a new layout.

    Parameters
    ----------
    pos : array, shape (n_channels, 3)
        The positions of the channels in 3d.
    names : list
        The channel names
    kind : str
        The type of Layout (e.g. 'standard_1005')
    """
    def __init__(self, pos, names, kind, ids):
        self.pos = pos
        self.names = names
        self.kind = kind
        self.ids = ids

    def __repr__(self):
        s = '<Montage | %s - Channels: %s ...>' % (self.kind,
                                                   ', '.join(self.names[:3]))
        return s

    def plot(self, scale_factor=1.5):
        """Plot EEG sensor montage

        Parameters
        ----------
        scale_factor : float
            Detemrines the size of the points. defaults to 1.5

        Returns
        -------
        fig : isntance of mayavi.Scene
            The malab scene object.
        """
        return plot_montage(self, scale_factor=scale_factor)


def read_montage(kind, names=None, path=None, scale=True):
    """Read layout from a file

    Parameters
    ----------
    kind : str
        The name of the .lout file (e.g. kind='Vectorview-all' for
        'Vectorview-all.lout'
    names : list of str
        The names to read. If None, all names are returned.
    path : str | None
        The path of the folder containing the Layout file
    scale : bool
        Apply useful scaling for out the box plotting using layout.pos

    Returns
    -------
    layout : instance of Layout
        The layout
    """
    if path is None:
        path = op.dirname(__file__)
    if not op.isabs(kind):
        supported = ('.elc', '.txt', '.csd', '.sfp')
        montages = [f for f in os.listdir(path) if f[-4:] in supported]
        kind = [f for f in montages if kind in f]
        if len(kind) != 1:
            raise ValueError('Could not find the montage. Please provide the'
                             'full path')
        kind = kind[0]

    if kind.endswith('.sfp'):
        # EGI geodesic
        dtype = np.dtype('S4, f8, f8, f8')
        fname = op.join(path, kind)
        data = np.loadtxt(fname, dtype=dtype)
        pos = np.c_[data['f1'], data['f2'], data['f3']]
        names_ = data['f0']
    elif kind.endswith('.elc'):
        # 10-5 system
        fname = op.join(path, kind)
        names_ = []
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
                names_.append(line.strip(' ').strip('\n'))
        pos = np.loadtxt(BytesIO(''.join(pos)))
    elif kind.endswith('.txt'):
        # easycap
        dtype = np.dtype('S4, f8, f8')
        fname = op.join(path, kind)
        data = np.loadtxt(fname, dtype=dtype, skiprows=1)
        theta, phi = data['f1'], data['f2']
        x = 85. * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
        y = 85. * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = 85. * np.cos(np.deg2rad(theta))
        pos = np.c_[x, y, z]
        names_ = data['f0']
    elif kind.endswith('.csd'):
        # CSD toolbox
        dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                 ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                 ('off_sph', 'f8')]
        fname = op.join(path, kind)
        table = np.loadtxt(fname, skiprows=2, dtype=dtype)
        pos = np.c_[table['x'], table['y'], table['z']]
        names_ = table['label']
    else:
        raise ValueError('Currently the "%s" template is not supported.' %
                         kind)
    ids = np.arange(len(pos))
    if names is not None:
        sel, names_ = zip(*[(i, e) for i, e in enumerate(names_)
                            if e in names])
        sel = list(sel)
        pos = pos[sel]
        ids = ids[sel]
    else:
        names_ = list(names_)
    kind = op.split(kind)[-1]
    return Montage(pos=pos, names=names_, kind=kind, ids=ids)


def apply_montage(info, montage):
    """Apply montage to EEG data.

    This function will replace the eeg channel names and locations with
    the values specified for the particular montage.
    Note. You have to rename your object to correclty map
    the montage names.
    Note. This function will change the info in place.

    Parameters
    ----------
    inst : instance of Info
        The info to update.
    montage : instance of Montage
        The montage to apply.
    """
    if not _contains_ch_type(info, 'eeg'):
        raise ValueError('No eeg channels found')
    for pos, name in zip(montage.pos, montage.names):
        ch_idx = info['ch_names'].index(name)
        info['ch_names'][ch_idx] = name
        info['chs'][ch_idx]['eeg_loc'] = np.c_[pos, [0] * 3]
        info['chs'][ch_idx]['loc'] = np.r_[pos, [0] * 9]
