# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: Simplified BSD

import logging
from collections import defaultdict
from itertools import combinations
import os
import os.path as op

import numpy as np
from scipy.spatial.distance import pdist

from .channels import _contains_ch_type
from ..io.pick import pick_types
from ..io.constants import FIFF
from ..utils import _clean_names
from ..externals.six.moves import map
from ..viz import plot_montage
from ..transforms import (_sphere_to_cartesian, _polar_to_cartesian,
                          _cartesian_to_sphere, apply_trans,
                          get_ras_to_neuromag_trans)


class Layout(object):
    """Sensor layouts

    Layouts are typically loaded from a file using read_layout. Only use this
    class directly if you're constructing a new layout.

    Parameters
    ----------
    box : tuple of length 4
        The box dimension (x_min, x_max, y_min, y_max).
    pos : array, shape=(n_channels, 4)
        The positions of the channels in 2d (x, y, width, height).
    names : list
        The channel names.
    ids : list
        The channel ids.
    kind : str
        The type of Layout (e.g. 'Vectorview-all').
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
            The file name (e.g. 'my_layout.lout').
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

    def __repr__(self):
        return '<Layout | %s - Channels: %s ...>' % (self.kind,
                                                     ', '.join(self.names[:3]))


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
        'Vectorview-all.lout').

    path : str | None
        The path of the folder containing the Layout file. Defaults to the
        mne/channels/data/layouts folder inside your mne-python installation.

    scale : bool
        Apply useful scaling for out the box plotting using layout.pos.
        Defaults to True.

    Returns
    -------
    layout : instance of Layout
        The layout.
    """
    if path is None:
        path = op.join(op.dirname(__file__), 'data', 'layouts')

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


def make_eeg_layout(info, radius=0.5, width=None, height=None, exclude='bads'):
    """Create .lout file from EEG electrode digitization

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        Measurement info (e.g., raw.info).
    radius : float
        Viewport radius as a fraction of main figure height. Defaults to 0.5.
    width : float | None
        Width of sensor axes as a fraction of main figure height. By default,
        this will be the maximum width possible without axes overlapping.
    height : float | None
        Height of sensor axes as a fraction of main figure height. By default,
        this will be the maximum height possible withough axes overlapping.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in info['bads'].
    Returns
    -------
    layout : Layout
        The generated Layout.
    """
    if not (0 <= radius <= 0.5):
        raise ValueError('The radius parameter should be between 0 and 0.5.')
    if width is not None and not (0 <= width <= 1.0):
        raise ValueError('The width parameter should be between 0 and 1.')
    if height is not None and not (0 <= height <= 1.0):
        raise ValueError('The height parameter should be between 0 and 1.')

    picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                       exclude=exclude)
    loc2d = _auto_topomap_coords(info, picks)
    names = [info['chs'][i]['ch_name'] for i in picks]

    # Scale [x, y] to [-0.5, 0.5]
    loc2d_min = np.min(loc2d, axis=0)
    loc2d_max = np.max(loc2d, axis=0)
    loc2d = (loc2d - (loc2d_max + loc2d_min) / 2.) / (loc2d_max - loc2d_min)

    # If no width or height specified, calculate the maximum value possible
    # without axes overlapping.
    if width is None or height is None:
        width, height = _box_size(loc2d, width, height, padding=0.1)

    # Scale to viewport radius
    loc2d *= 2 * radius

    # Some subplot centers will be at the figure edge. Shrink everything so it
    # fits in the figure.
    scaling = min(1 / (1. + width), 1 / (1. + height))
    loc2d *= scaling
    width *= scaling
    height *= scaling

    # Shift to center
    loc2d += 0.5

    n_channels = loc2d.shape[0]
    pos = np.c_[loc2d[:, 0] - 0.5 * width,
                loc2d[:, 1] - 0.5 * height,
                width * np.ones(n_channels),
                height * np.ones(n_channels)]

    box = (0, 1, 0, 1)
    ids = 1 + np.arange(n_channels)
    layout = Layout(box=box, pos=pos, names=names, kind='EEG', ids=ids)
    return layout


def make_grid_layout(info, picks=None, n_col=None):
    """ Generate .lout file for custom data, i.e., ICA sources

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info | None
        Measurement info (e.g., raw.info). If None, default names will be
        employed.
    picks : array-like of int | None
        The indices of the channels to be included. If None, al misc channels
        will be included.
    n_col : int | None
        Number of columns to generate. If None, a square grid will be produced.

    Returns
    -------
    layout : Layout
        The generated layout.
    """
    if picks is None:
        picks = pick_types(info, misc=True, ref_meg=False, exclude='bads')

    names = [info['chs'][k]['ch_name'] for k in picks]

    if not names:
        raise ValueError('No misc data channels found.')

    ids = list(range(len(picks)))
    size = len(picks)

    if n_col is None:
        # prepare square-like layout
        n_row = n_col = np.sqrt(size)  # try square
        if n_col % 1:
            # try n * (n-1) rectangle
            n_col, n_row = int(n_col + 1), int(n_row)

        if n_col * n_row < size:  # jump to the next full square
            n_row += 1
    else:
        n_row = np.ceil(size / float(n_col))

    # setup position grid
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, n_col),
                       np.linspace(-0.5, 0.5, n_row))
    x, y = x.ravel()[:size], y.ravel()[:size]
    width, height = _box_size(np.c_[x, y], padding=0.1)

    # Some axes will be at the figure edge. Shrink everything so it fits in the
    # figure. Add 0.01 border around everything
    border_x, border_y = (0.01, 0.01)
    x_scaling = 1 / (1. + width + border_x)
    y_scaling = 1 / (1. + height + border_y)
    x = x * x_scaling
    y = y * y_scaling
    width *= x_scaling
    height *= y_scaling

    # Shift to center
    x += 0.5
    y += 0.5

    # calculate pos
    pos = np.c_[x - 0.5 * width, y - 0.5 * height,
                width * np.ones(size), height * np.ones(size)]
    box = (0, 1, 0, 1)

    layout = Layout(box=box, pos=pos, names=names, kind='grid-misc', ids=ids)
    return layout


def find_layout(info, ch_type=None, exclude='bads'):
    """Choose a layout based on the channels in the info 'chs' field

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        The measurement info.
    ch_type : {'mag', 'grad', 'meg', 'eeg'} | None
        The channel type for selecting single channel layouts.
        Defaults to None. Note, this argument will only be considered for
        VectorView type layout. Use `meg` to force using the full layout
        in situations where the info does only contain one sensor type.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in info['bads'].

    Returns
    -------
    layout : Layout instance | None
        None if layout not found.
    """
    our_types = ' or '.join(['`None`', '`mag`', '`grad`', '`meg`'])
    if ch_type not in (None, 'meg', 'mag', 'grad', 'eeg'):
        raise ValueError('Invalid channel type (%s) requested '
                         '`ch_type` must be %s' % (ch_type, our_types))

    chs = info['chs']
    coil_types = set([ch['coil_type'] for ch in chs])
    channel_types = set([ch['kind'] for ch in chs])

    has_vv_mag = any([k in coil_types for k in [FIFF.FIFFV_COIL_VV_MAG_T1,
                                                FIFF.FIFFV_COIL_VV_MAG_T2,
                                                FIFF.FIFFV_COIL_VV_MAG_T3]])
    has_vv_grad = any([k in coil_types for k in [FIFF.FIFFV_COIL_VV_PLANAR_T1,
                                                 FIFF.FIFFV_COIL_VV_PLANAR_T2,
                                                 FIFF.FIFFV_COIL_VV_PLANAR_T3]]
                      )
    has_vv_meg = has_vv_mag and has_vv_grad
    has_vv_only_mag = has_vv_mag and not has_vv_grad
    has_vv_only_grad = has_vv_grad and not has_vv_mag
    is_old_vv = ' ' in chs[0]['ch_name']

    has_4D_mag = FIFF.FIFFV_COIL_MAGNES_MAG in coil_types
    ctf_other_types = (FIFF.FIFFV_COIL_CTF_REF_MAG,
                       FIFF.FIFFV_COIL_CTF_REF_GRAD,
                       FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD)
    has_CTF_grad = (FIFF.FIFFV_COIL_CTF_GRAD in coil_types or
                    (FIFF.FIFFV_MEG_CH in channel_types and
                     any([k in ctf_other_types for k in coil_types])))
    # hack due to MNE-C bug in IO of CTF
    n_kit_grads = len([ch for ch in chs
                       if ch['coil_type'] == FIFF.FIFFV_COIL_KIT_GRAD])

    has_any_meg = any([has_vv_mag, has_vv_grad, has_4D_mag, has_CTF_grad,
                       n_kit_grads])
    has_eeg_coils = (FIFF.FIFFV_COIL_EEG in coil_types and
                     FIFF.FIFFV_EEG_CH in channel_types)
    has_eeg_coils_and_meg = has_eeg_coils and has_any_meg
    has_eeg_coils_only = has_eeg_coils and not has_any_meg

    if ch_type == "meg" and not has_any_meg:
        raise RuntimeError('No MEG channels present. Cannot find MEG layout.')

    if ch_type == "eeg" and not has_eeg_coils:
        raise RuntimeError('No EEG channels present. Cannot find EEG layout.')

    if ((has_vv_meg and ch_type is None) or
            (any([has_vv_mag, has_vv_grad]) and ch_type == 'meg')):
        layout_name = 'Vectorview-all'
    elif has_vv_only_mag or (has_vv_meg and ch_type == 'mag'):
        layout_name = 'Vectorview-mag'
    elif has_vv_only_grad or (has_vv_meg and ch_type == 'grad'):
        layout_name = 'Vectorview-grad'
    elif ((has_eeg_coils_only and ch_type in [None, 'eeg']) or
          (has_eeg_coils_and_meg and ch_type == 'eeg')):
        if not isinstance(info, dict):
            raise RuntimeError('Cannot make EEG layout, no measurement info '
                               'was passed to `find_layout`')
        return make_eeg_layout(info, exclude=exclude)
    elif has_4D_mag:
        layout_name = 'magnesWH3600'
    elif has_CTF_grad:
        layout_name = 'CTF-275'
    elif n_kit_grads == 157:
        layout_name = 'KIT-157'
    else:
        return None

    layout = read_layout(layout_name)
    if not is_old_vv:
        layout.names = _clean_names(layout.names, remove_whitespace=True)
    if has_CTF_grad:
        layout.names = _clean_names(layout.names, before_dash=True)

    return layout


def _box_size(points, width=None, height=None, padding=0.0):
    """ Given a series of points, calculate an appropriate box size.

    Parameters
    ----------
    points : array, shape (n_points, 2)
        The centers of the axes as a list of (x, y) coordinate pairs. Normally
        these are points in the range [0, 1] centered at 0.5.
    width : float | None
        An optional box width to enforce. When set, only the box height will be
        calculated by the function.
    height : float | None
        An optional box height to enforce. When set, only the box width will be
        calculated by the function.
    padding : float
        Portion of the box to reserve for padding. The value can range between
        0.0 (boxes will touch, default) to 1.0 (boxes consist of only padding).

    Returns
    -------
    width : float
        Width of the box
    height : float
        Height of the box
    """
    def xdiff(a, b):
        return np.abs(a[0] - b[0])

    def ydiff(a, b):
        return np.abs(a[1] - b[1])

    points = np.asarray(points)
    all_combinations = list(combinations(points, 2))

    if width is None and height is None:
        if len(points) <= 1:
            # Trivial case first
            width = 1.0
            height = 1.0
        else:
            # Find the closest two points A and B.
            a, b = all_combinations[np.argmin(pdist(points))]

            # The closest points define either the max width or max height.
            w, h = xdiff(a, b), ydiff(a, b)
            if w > h:
                width = w
            else:
                height = h

    # At this point, either width or height is known, or both are known.
    if height is None:
        # Find all axes that could potentially overlap horizontally.
        hdist = pdist(points, xdiff)
        candidates = [all_combinations[i] for i, d in enumerate(hdist)
                      if d < width]

        if len(candidates) == 0:
            # No axes overlap, take all the height you want.
            height = 1.0
        else:
            # Find an appropriate height so all none of the found axes will
            # overlap.
            height = np.min([ydiff(*c) for c in candidates])

    elif width is None:
        # Find all axes that could potentially overlap vertically.
        vdist = pdist(points, ydiff)
        candidates = [all_combinations[i] for i, d in enumerate(vdist)
                      if d < height]

        if len(candidates) == 0:
            # No axes overlap, take all the width you want.
            width = 1.0
        else:
            # Find an appropriate width so all none of the found axes will
            # overlap.
            width = np.min([xdiff(*c) for c in candidates])

    # Add a bit of padding between boxes
    width *= 1 - padding
    height *= 1 - padding

    return width, height


def _find_topomap_coords(info, picks, layout=None):
    """Try to guess the E/MEG layout and return appropriate topomap coordinates

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        Measurement info.
    picks : list of int
        Channel indices to generate topomap coords for.
    layout : None | instance of Layout
        Enforce using a specific layout. With None, a new map is generated.
        With None, a layout is chosen based on the channels in the chs
        parameter.

    Returns
    -------
    coords : array, shape = (n_chs, 2)
        2 dimensional coordinates for each sensor for a topomap plot.
    """
    if len(picks) == 0:
        raise ValueError("Need more than 0 channels.")

    if layout is not None:
        chs = [info['chs'][i] for i in picks]
        pos = [layout.pos[layout.names.index(ch['ch_name'])] for ch in chs]
        pos = np.asarray(pos)
    else:
        pos = _auto_topomap_coords(info, picks)

    return pos


def _auto_topomap_coords(info, picks):
    """Make a 2 dimensional sensor map from sensor positions in an info dict.
    The default is to use the electrode locations. The fallback option is to
    attempt using digitization points of kind FIFFV_POINT_EEG. This only works
    with EEG and requires an equal number of digitization points and sensors.

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        The measurement info.
    picks : list of int
        The channel indices to generate topomap coords for.

    Returns
    -------
    locs : array, shape = (n_sensors, 2)
        An array of positions of the 2 dimensional map.
    """
    chs = [info['chs'][i] for i in picks]

    # Use channel locations if available
    locs3d = np.array([ch['loc'][:3] for ch in chs])

    # If electrode locations are not available, use digization points
    if len(locs3d) == 0 or np.allclose(locs3d, 0):
        logging.warning('Did not find any electrode locations the info, '
                        'will attempt to use digitization points instead. '
                        'However, if digitization points do not correspond to '
                        'the EEG electrodes, this will lead to bad results. '
                        'Please verify that the sensor locations in the plot '
                        'are accurate.')

        # MEG/EOG/ECG sensors don't have digitization points; all requested
        # channels must be EEG
        for ch in chs:
            if ch['kind'] != FIFF.FIFFV_EEG_CH:
                raise ValueError("Cannot determine location of MEG/EOG/ECG "
                                 "channels using digitization points.")
                break

        eeg_ch_names = [ch['ch_name'] for ch in info['chs']
                        if ch['kind'] == FIFF.FIFFV_EEG_CH]

        # Get EEG digitization points
        if info['dig'] is None or len(info['dig']) == 0:
            raise RuntimeError('No digitization points found.')

        locs3d = np.array([point['r'] for point in info['dig']
                           if point['kind'] == FIFF.FIFFV_POINT_EEG])

        if len(locs3d) == 0:
            raise RuntimeError('Did not find any digitization points of '
                               'kind FIFFV_POINT_EEG (%d) in the info.'
                               % FIFF.FIFFV_POINT_EEG)

        if len(locs3d) != len(eeg_ch_names):
            raise ValueError("Number of EEG digitization points (%d) "
                             "doesn't match the number of EEG channels "
                             "(%d)" % (len(locs3d), len(eeg_ch_names)))

        # Center digitization points on head origin
        dig_kinds = (FIFF.FIFFV_POINT_CARDINAL,
                     FIFF.FIFFV_POINT_EEG,
                     FIFF.FIFFV_POINT_EXTRA)
        from ..preprocessing.maxfilter import fit_sphere_to_headshape
        _, origin_head, _ = fit_sphere_to_headshape(info, dig_kinds)
        origin_head /= 1000.  # to meters
        locs3d -= origin_head

        # Match the digitization points with the requested
        # channels.
        eeg_ch_locs = dict(zip(eeg_ch_names, locs3d))
        locs3d = np.array([eeg_ch_locs[ch['ch_name']] for ch in chs])

    # Duplicate points cause all kinds of trouble during visualization
    if np.min(pdist(locs3d)) < 1e-10:
        raise ValueError('Electrode positions must be unique.')

    x, y, z = locs3d.T
    az, el, r = _cartesian_to_sphere(x, y, z)
    locs2d = np.c_[_polar_to_cartesian(az, np.pi / 2 - el)]
    return locs2d


def _pair_grad_sensors(info, layout=None, topomap_coords=True, exclude='bads'):
    """Find the picks for pairing grad channels

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info
        An info dictionary containing channel information.
    layout : Layout | None
        The layout if available. Defaults to None.
    topomap_coords : bool
        Return the coordinates for a topomap plot along with the picks. If
        False, only picks are returned. Defaults to True.
    exclude : list of str | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in info['bads']. Defaults to 'bads'.

    Returns
    -------
    picks : array of int
        Picks for the grad channels, ordered in pairs.
    coords : array, shape = (n_grad_channels, 3)
        Coordinates for a topomap plot (optional, only returned if
        topomap_coords == True).
    """
    # find all complete pairs of grad channels
    pairs = defaultdict(list)
    grad_picks = pick_types(info, meg='grad', ref_meg=False, exclude=exclude)
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
    picks = [ch_names.index(c['ch_name']) for c in grad_chs]

    if topomap_coords:
        shape = (len(pairs), 2, -1)
        coords = (_find_topomap_coords(info, picks, layout)
                  .reshape(shape).mean(axis=1))
        return picks, coords
    else:
        return picks


# this function is used to pair grad when info is not present
# it is the case of Projection that don't have the info.
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
        fig : Instance of matplotlib.figure.Figure
            The figure object.
        """
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names)


def read_montage(kind, ch_names=None, path=None, unit='m', transform=False):
    """Read montage from a file

    Note: built-in montages are not scaled or transformed by default.

    Parameters
    ----------
    kind : str
        The name of the montage file (e.g. kind='easycap-M10' for
        'easycap-M10.txt'). Files with extensions '.elc', '.txt', '.csd',
        '.elp', '.hpts' or '.sfp' are supported.
    ch_names : list of str | None
        The names to read. If None, all names are returned.
    path : str | None
        The path of the folder containing the montage file. Defaults to the
        mne/channels/data/montages folder in your mne-python installation.
    unit : 'm' | 'cm' | 'mm'
        Unit of the input file. If not 'm', coordinates will be rescaled
        to 'm'.
    transform : bool
        If True, points will be transformed to Neuromag space.
        The fidicuals, 'nasion', 'lpa', 'rpa' must be specified in
        the montage file. Useful for points captured using Polhemus FastSCAN.
        Default is False.

    Returns
    -------
    montage : instance of Montage
        The montage.
    """
    if path is None:
        path = op.join(op.dirname(__file__), 'data', 'montages')
    if not op.isabs(kind):
        supported = ('.elc', '.txt', '.csd', '.sfp', '.elp', '.hpts')
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
        ch_names_ = data['f0'].astype(np.str)
    elif ext == '.elc':
        # 10-5 system
        ch_names_ = []
        pos = []
        with open(fname) as fid:
            for line in fid:
                if 'Positions\n' in line:
                    break
            pos = []
            for line in fid:
                if 'Labels\n' in line:
                    break
                pos.append(list(map(float, line.split())))
            for line in fid:
                if not line or not set(line) - set([' ']):
                    break
                ch_names_.append(line.strip(' ').strip('\n'))
        pos = np.array(pos)
    elif ext == '.txt':
        # easycap
        try:  # newer version
            data = np.genfromtxt(fname, dtype='str', skip_header=1)
        except TypeError:
            data = np.genfromtxt(fname, dtype='str', skiprows=1)
        ch_names_ = list(data[:, 0])
        theta, phi = data[:, 1].astype(float), data[:, 2].astype(float)
        x = 85. * np.cos(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
        y = 85. * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        z = 85. * np.cos(np.deg2rad(theta))
        pos = np.c_[x, y, z]
    elif ext == '.csd':
        # CSD toolbox
        dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                 ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                 ('off_sph', 'f8')]
        try:  # newer version
            table = np.loadtxt(fname, skip_header=2, dtype=dtype)
        except TypeError:
            table = np.loadtxt(fname, skiprows=2, dtype=dtype)
        ch_names_ = table['label']
        theta = (2 * np.pi * table['theta']) / 360.
        phi = (2 * np.pi * table['phi']) / 360.
        pos = _sphere_to_cartesian(theta, phi, r=1.0)
        pos = np.asarray(pos).T
    elif ext == '.elp':
        # standard BESA spherical
        dtype = np.dtype('S8, S8, f8, f8, f8')
        try:
            data = np.loadtxt(fname, dtype=dtype, skip_header=1)
        except TypeError:
            data = np.loadtxt(fname, dtype=dtype, skiprows=1)

        az = data['f2']
        horiz = data['f3']

        radius = np.abs(az / 180.)
        angles = np.array([90. - h if a >= 0. else -90. - h
                           for h, a in zip(horiz, az)])

        sph_phi = (0.5 - radius) * 180.
        sph_theta = angles

        azimuth = sph_theta / 180.0 * np.pi
        elevation = sph_phi / 180.0 * np.pi
        r = 85.

        y, x, z = _sphere_to_cartesian(azimuth, elevation, r)

        pos = np.c_[x, y, z]
        ch_names_ = data['f1'].astype(np.str)
    elif ext == '.hpts':
        # MNE-C specified format for generic digitizer data
        dtype = [('type', 'S8'), ('name', 'S8'),
                 ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        data = np.loadtxt(fname, dtype=dtype)
        pos = np.vstack((data['x'], data['y'], data['z'])).T
        ch_names_ = data['name'].astype(np.str)
    else:
        raise ValueError('Currently the "%s" template is not supported.' %
                         kind)
    selection = np.arange(len(pos))

    if unit == 'mm':
        pos /= 1e3
    elif unit == 'cm':
        pos /= 1e2
    elif unit != 'm':
        raise ValueError("'unit' should be either 'm', 'cm', or 'mm'.")
    if transform:
        names_lower = [name.lower() for name in list(ch_names_)]
        if ext == '.hpts':
            fids = ('2', '1', '3')  # Alternate cardinal point names
        else:
            fids = ('nasion', 'lpa', 'rpa')

        missing = [name for name in fids
                   if name not in names_lower]
        if missing:
            raise ValueError("The points %s are missing, but are needed "
                             "to transform the points to the MNE coordinate "
                             "system. Either add the points, or read the "
                             "montage with transform=False. " % missing)
        nasion = pos[names_lower.index(fids[0])]
        lpa = pos[names_lower.index(fids[1])]
        rpa = pos[names_lower.index(fids[2])]

        neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        pos = apply_trans(neuromag_trans, pos)

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
