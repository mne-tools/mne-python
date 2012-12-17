import os.path as op
import numpy as np
from ..preprocessing.maxfilter import fit_sphere_to_headshape
from ..fiff import pick_types


class Layout(object):
    """Sensor layouts

    Parameters
    ----------
    kind : 'Vectorview-all' | 'CTF-275' | 'Vectorview-grad' | 'Vectorview-mag'
        Type of layout (can also be custom for EEG)
    path : string
        Path to folder where to find the layout file.

    Attributes
    ----------
    box : tuple of length 4
        The box dimension (x_min, x_max, y_min, y_max)
    pos : array, shape=(n_channels, 4)
        The positions of the channels in 2d (x, y, width, height)
    names : list
        The channel names
    ids : list
        The channel ids
    kind : str
        The type of Layout (e.g. 'Vectorview-all')
    """
    def __init__(self, box, pos, names, ids, kind):
        self.box = box
        self.pos = pos
        self.names = names
        self.ids = ids
        self.kind = kind

    def save(self, fname):
        """Save Layout to disk

        Parameters
        ----------
        fname : str
            The file name (e.g. 'my_layout.lout')
        """
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        width = self.pos[:, 2]
        height = self.pos[:, 3]
        out_str = '%8.2f %8.2f %8.2f %8.2f\n' % self.box
        for ii in range(x.shape[0]):
            out_str += ('%03d %8.2f %8.2f %8.2f %8.2f %s\n' % (self.ids[ii],
                        x[ii], y[ii], width[ii], height[ii], self.names[ii]))

        f = open(fname, 'w')
        f.write(out_str)
        f.close()


def read_layout(kind, path=None, scale=True):
    """Read layout from a file

    Parameters
    ----------
    kind : str
        The name of the .lout file (e.g. kind='Vectorview-all' for
        'Vectorview-all.lout')

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

    if kind.endswith('.lout'):
        kind = kind[:-5]
    lout_fname = op.join(path, kind + '.lout')

    f = open(lout_fname)
    box_line = f.readline()  # first line contains box dimension
    box = tuple(map(float, box_line.split()))

    names = []
    pos = []
    ids = []

    for line in f:
        splits = line.split()
        if len(splits) == 7:
            cid, x, y, dx, dy, chkind, nb = splits
            name = chkind + ' ' + nb
        else:
            cid, x, y, dx, dy, name = splits
        pos.append(np.array([x, y, dx, dy], dtype=np.float))
        names.append(name)
        ids.append(int(cid))

    pos = np.array(pos)

    if scale:
        pos[:, 0] -= np.min(pos[:, 0])
        pos[:, 1] -= np.min(pos[:, 1])
        scaling = max(np.max(pos[:, 0]), np.max(pos[:, 1])) + pos[0, 2]
        pos /= scaling
        pos[:, :2] += 0.03
        pos[:, :2] *= 0.97 / 1.03
        pos[:, 2:] *= 0.94

    f.close()
    return Layout(box=box, pos=pos, names=names, kind=kind, ids=ids)


def make_eeg_layout(info, radius=20, width=5, height=4):
    """Create .lout file from EEG electrode digitization

    Parameters
    ----------
    info : dict
        Measurement info (e.g., raw.info)
    radius : float
        Viewport radius
    width : float
        Viewport width
    height : float
        Viewport height

    Returns
    -------
    layout : Layout
        The generated Layout
    """
    radius_head, origin_head, origin_device = fit_sphere_to_headshape(info)
    inds = pick_types(info, meg=False, eeg=True)
    hsp = [info['chs'][ii]['eeg_loc'][:, 0] for ii in inds]
    names = [info['chs'][ii]['ch_name'] for ii in inds]
    if len(hsp) <= 0:
        raise ValueError('No EEG digitization points found')

    if not len(hsp) == len(names):
        raise ValueError('Channel names don\'t match digitization values')
    hsp = np.array(hsp)

    # Move points to origin
    hsp -= origin_head / 1e3  # convert to millimeters

    # Calculate angles
    r = np.sqrt(np.sum(hsp ** 2, axis=-1))
    theta = np.arccos(hsp[:, 2] / r)
    phi = np.arctan2(hsp[:, 1], hsp[:, 0])

    # Mark the points that might have caused bad angle estimates
    iffy = np.nonzero(
        np.sum(hsp[:, :2] ** 2, axis=-1) ** (1. / 2) < np.finfo(np.float).eps * 10)
    theta[iffy] = 0
    phi[iffy] = 0

    # Do the azimuthal equidistant projection
    x = radius * (2.0 * theta / np.pi) * np.cos(phi)
    y = radius * (2.0 * theta / np.pi) * np.sin(phi)

    n_channels = len(x)
    pos = np.c_[x, y, width * np.ones(n_channels), height * np.ones(n_channels)]

    box = (x.min() - 0.1 * width, x.max() + 1.1 * width,
           y.min() - 0.1 * width, y.max() + 1.1 * height)
    ids = 1 + np.arange(n_channels)
    layout = Layout(box=box, pos=pos, names=names, kind='EEG', ids=ids)
    return layout


def make_grid_layout(info, picks=None):
    """ Generate .lout file for custom data, i.e., ICA sources

    Parameters
    ----------
    info : dict
        Measurement info (e.g., raw.info). If None, default names will be
        employed.
    picks : array-like | None
        The indices of the channels to be included. If None, al misc channels will
        be included.

    Returns
    -------
    layout : Layout
        The generated layout.

    """
    if picks is None:
        picks = pick_types(info, misc=True)

    names = [info['chs'][k]['ch_name'] for k in picks]

    if not names:
        raise ValueError('No misc data channels found.')

    ids = range(len(picks))
    size = len(picks)

    # prepare square-like layout
    ht = wd = np.sqrt(size)  # try square
    if wd % 1:
        wd, ht = int(wd + 1), int(ht)  # try n * (n-1) rectangle

    if wd * ht < size:  # jump to the next full square
        ht += 1

    # setup position grid and fill up
    x, y = np.meshgrid(np.linspace(0, 1, wd), np.linspace(0, 1, ht))

    # scale boxes depending on size such that square is always filled
    width = size * .15  # value depends on mne default full-view size
    spacing = (width * ht)

    # XXX : width and height are here assumed to be equal. Could be improved.
    x, y = (x.ravel()[:size] * spacing, y.ravel()[:size] * spacing)

    # calculate pos
    pos = np.c_[x, y, width * np.ones(size), width * np.ones(size)]

    # calculate box
    box = (x.min() - 0.1 * width, x.max() + 1.1 * width,
           y.min() - 0.1 * width, y.max() + 1.1 * width)

    layout = Layout(box=box, pos=pos, names=names, kind='grid-misc', ids=ids)
    return layout
