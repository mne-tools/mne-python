from collections import defaultdict
import os.path as op
import numpy as np
from scipy.optimize import leastsq
from ..preprocessing.maxfilter import fit_sphere_to_headshape
from ..fiff import FIFF, pick_types


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
        if fname.endswith('.lout'):
            out_str = '%8.2f %8.2f %8.2f %8.2f\n' % self.box
        elif fname.endswith('.lay'):
            out_str = ''
        else:
            raise ValueError('Unknown layout type. Should be of type '
                             '.lout or .lay.')

        for ii in range(x.shape[0]):
            out_str += ('%03d %8.2f %8.2f %8.2f %8.2f %s\n' % (self.ids[ii],
                        x[ii], y[ii], width[ii], height[ii], self.names[ii]))

        f = open(fname, 'w')
        f.write(out_str)
        f.close()


def _read_lout(fname):
    """Aux function"""
    with open(fname) as f:
        box_line = f.readline()  # first line contains box dimension
        box = tuple(map(float, box_line.split()))
        names, pos, ids = [], [], []
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

    return box, pos, names, ids


def _read_lay(fname):
    """Aux function"""
    with open(fname) as f:
        box = None
        names, pos, ids = [], [], []
        for line in f:
            splits = line.split()
            cid, x, y, dx, dy, name = splits
            pos.append(np.array([x, y, dx, dy], dtype=np.float))
            names.append(name)
            ids.append(int(cid))

    pos = np.array(pos)

    return box, pos, names, ids


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

    if not kind.endswith('.lout') and op.exists(op.join(path, kind + '.lout')):
        kind += '.lout'
    elif not kind.endswith('.lay') and op.exists(op.join(path, kind + '.lay')):
        kind += '.lay'

    if kind.endswith('.lout'):
        fname = op.join(path, kind)
        kind = kind[:-5]
        box, pos, names, ids = _read_lout(fname)
    elif kind.endswith('.lay'):
        fname = op.join(path, kind)
        kind = kind[:-4]
        box, pos, names, ids = _read_lay(fname)
        kind.endswith('.lay')
    else:
        raise ValueError('Unknown layout type. Should be of type '
                         '.lout or .lay.')

    if scale:
        pos[:, 0] -= np.min(pos[:, 0])
        pos[:, 1] -= np.min(pos[:, 1])
        scaling = max(np.max(pos[:, 0]), np.max(pos[:, 1])) + pos[0, 2]
        pos /= scaling
        pos[:, :2] += 0.03
        pos[:, :2] *= 0.97 / 1.03
        pos[:, 2:] *= 0.94

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
    inds = pick_types(info, meg=False, eeg=True, exclude='bads')
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
    iffy = np.nonzero(np.sum(hsp[:, :2] ** 2, axis=-1) ** (1. / 2)
                      < np.finfo(np.float).eps * 10)
    theta[iffy] = 0
    phi[iffy] = 0

    # Do the azimuthal equidistant projection
    x = radius * (2.0 * theta / np.pi) * np.cos(phi)
    y = radius * (2.0 * theta / np.pi) * np.sin(phi)

    n_channels = len(x)
    pos = np.c_[x, y, width * np.ones(n_channels),
                height * np.ones(n_channels)]

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
        The indices of the channels to be included. If None, al misc channels
        will be included.

    Returns
    -------
    layout : Layout
        The generated layout.
    """
    if picks is None:
        picks = pick_types(info, misc=True, exclude='bads')

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


def find_layout(chs):
    """Choose a layout based on the channels in the chs parameter

    Parameters
    ----------
    chs : list
        A list of channels as contained in the info['chs'] entry.

    Returns
    -------
    layout : Layout instance | None
        None if layout not found.
    """

    coil_types = np.unique([ch['coil_type'] for ch in chs])
    has_vv_mag = FIFF.FIFFV_COIL_VV_MAG_T3 in coil_types
    has_vv_grad = FIFF.FIFFV_COIL_VV_PLANAR_T1 in coil_types
    has_4D_mag = FIFF.FIFFV_COIL_MAGNES_MAG in coil_types
    if has_vv_mag and has_vv_grad:
        layout_name = 'Vectorview-all'
    elif has_vv_mag:
        layout_name = 'Vectorview-mag'
    elif has_vv_grad:
        layout_name = 'Vectorview-grad'
    elif has_4D_mag:
        layout_name = 'magnesWH3600'
    else:
        return None
    
    return read_layout(layout_name)

def _find_topomap_coords(chs, layout=None):
    """Try to guess the MEG system and return appropriate topomap coordinates

    Parameters
    ----------
    chs : list
        A list of channels as contained in the info['chs'] entry.
    layout : None | instance of Layout
        Enforce using a specific layout. With None, a new map is generated.
        With None, a layout is chosen based on the channels in the chs
        parameter.

    Returns
    -------
    coords : array, shape = (n_chs, 2)
        2 dimensional coordinates for each sensor for a topomap plot.
    """
    if len(chs) == 0:
        raise ValueError("Need more than 0 channels.")

    if layout is not None:
        pos = [layout.pos[layout.names.index(ch['ch_name'])] for ch in chs]
        pos = np.asarray(pos)
    else:
        pos = _auto_topomap_coords(chs)

    return pos


def _auto_topomap_coords(chs):
    """Make a 2 dimensional sensor map from sensor positions in an info dict

    Parameters
    ----------
    chs : list
        A list of channels as contained in the info['chs'] entry.

    Returns
    -------
    locs : array, shape = (n_sensors, 2)
        An array of positions of the 2 dimensional map.
    """
    locs3d = np.array([ch['loc'][:3] for ch in chs])

    # fit the 3d sensor locations to a sphere with center (cx, cy, cz)
    # and radius r

    # error function
    def err(params):
        r, cx, cy, cz = params
        return   np.sum((locs3d - [cx, cy, cz]) ** 2, 1) - r ** 2

    (r, cx, cy, cz), _ = leastsq(err, (1, 0, 0, 0))

    # center the sensor locations based on the sphere and scale to
    # radius 1
    sphere_center = np.array((cx, cy, cz))
    locs3d -= sphere_center
    locs3d /= r

    # implement projection
    locs2d = np.copy(locs3d[:, :2])
    z = max(locs3d[:, 2]) - locs3d[:, 2]  # distance form top
    r = np.sqrt(z)  # desired 2d radius
    r_xy = np.sqrt(np.sum(locs3d[:, :2] ** 2, 1))  # current radius in xy
    idx = (r_xy != 0)  # avoid zero division
    F = r[idx] / r_xy[idx]  # stretching factor accounting for current r
    locs2d[idx, :] *= F[:, None]

    return locs2d


def _pair_grad_sensors(info, layout=None, topomap_coords=True, exclude='bads'):
    """Find the picks for pairing grad channels

    Parameters
    ----------
    info : dict
        An info dictionary containing channel information.
    layout : Layout
        The layout if available.
    topomap_coords : bool
        Return the coordinates for a topomap plot along with the picks. If
        False, only picks are returned.
    exclude : list of str | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in info['bads'].

    Returns
    -------
    picks : list of int
        Picks for the grad channels, ordered in pairs.
    coords : array, shape = (n_grad_channels, 3)
        Coordinates for a topomap plot (optional, only returned if
        topomap_coords == True).
    """
    # find all complete pairs of grad channels
    pairs = defaultdict(list)
    grad_picks = pick_types(info, meg='grad', exclude=exclude)
    for i in grad_picks:
        ch = info['chs'][i]
        name = ch['ch_name']
        if name.startswith('MEG'):
            if name.endswith(('2', '3')):
                key = name[-4:-1]
                pairs[key].append(ch)
    pairs = [p for p in pairs.values() if len(p) == 2]
    if len(pairs) == 0:
        raise ValueError("No 'grad' channel pairs found.")

    # find the picks corresponding to the grad channels
    grad_chs = sum(pairs, [])
    ch_names = info['ch_names']
    picks = [ch_names.index(ch['ch_name']) for ch in grad_chs]

    if topomap_coords:
        shape = (len(pairs), 2, -1)
        coords = _find_topomap_coords(grad_chs, layout).reshape(shape).mean(axis=1)
        return picks, coords
    else:
        return picks


def _pair_grad_sensors_from_ch_names(ch_names):
    """Find the indexes for pairing grad channels

    Parameters
    ----------
    ch_names : list of str
        A list of channel names.

    Returns
    -------
    indexes : list of int
        Indexes of the grad channels, ordered in pairs.
    """
    pairs = defaultdict(list)
    for i, name in enumerate(ch_names):
        if name.startswith('MEG'):
            if name.endswith(('2', '3')):
                key = name[-4:-1]
                pairs[key].append(i)

    pairs = [p for p in pairs.values() if len(p) == 2]

    grad_chs = sum(pairs, [])
    return grad_chs


def _merge_grad_data(data):
    """Merge data from channel pairs using the RMS

    Parameters
    ----------
    data : array, shape = (n_channels, n_times)
        Data for channels, ordered in pairs.

    Returns
    -------
    data : array, shape = (n_channels / 2, n_times)
        The root mean square for each pair.
    """
    data = data.reshape((len(data) // 2, 2, -1))
    data = np.sqrt(np.sum(data ** 2, axis=1) / 2)
    return data
