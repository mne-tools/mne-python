# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: Simplified BSD

import logging
from collections import defaultdict
from itertools import combinations
import os.path as op

import numpy as np

from ..transforms import _pol_to_cart, _cart_to_sph
from ..bem import fit_sphere_to_headshape
from ..io.pick import pick_types
from ..io.constants import FIFF
from ..io.meas_info import Info
from ..utils import _clean_names, warn
from ..externals.six.moves import map
from .channels import _get_ch_info


class Layout(object):
    """Sensor layouts.

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

    def __init__(self, box, pos, names, ids, kind):  # noqa: D102
        self.box = box
        self.pos = pos
        self.names = names
        self.ids = ids
        self.kind = kind

    def save(self, fname):
        """Save Layout to disk.

        Parameters
        ----------
        fname : str
            The file name (e.g. 'my_layout.lout').

        See Also
        --------
        read_layout
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
        """Return the string representation."""
        return '<Layout | %s - Channels: %s ...>' % (self.kind,
                                                     ', '.join(self.names[:3]))

    def plot(self, picks=None, show=True):
        """Plot the sensor positions.

        Parameters
        ----------
        picks : array-like
            Indices of the channels to show. If None (default), all the
            channels are shown.
        show : bool
            Show figure if True. Defaults to True.

        Returns
        -------
        fig : instance of matplotlib figure
            Figure containing the sensor topography.

        Notes
        -----
        .. versionadded:: 0.12.0
        """
        from ..viz.topomap import plot_layout
        return plot_layout(self, picks=picks, show=show)


def _read_lout(fname):
    """Aux function."""
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
    """Aux function."""
    with open(fname) as f:
        box = None
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


def read_layout(kind, path=None, scale=True):
    """Read layout from a file.

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

    See Also
    --------
    Layout.save
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
    """Create .lout file from EEG electrode digitization.

    Parameters
    ----------
    info : instance of Info
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
        List of channels to exclude. If empty do not exclude any.
        If 'bads', exclude channels in info['bads'] (default).

    Returns
    -------
    layout : Layout
        The generated Layout.

    See Also
    --------
    make_grid_layout, generate_2d_layout
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
    """Generate .lout file for custom data, i.e., ICA sources.

    Parameters
    ----------
    info : instance of Info | None
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

    See Also
    --------
    make_eeg_layout, generate_2d_layout
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
        n_row = int(np.ceil(size / float(n_col)))

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
    """Choose a layout based on the channels in the info 'chs' field.

    Parameters
    ----------
    info : instance of Info
        The measurement info.
    ch_type : {'mag', 'grad', 'meg', 'eeg'} | None
        The channel type for selecting single channel layouts.
        Defaults to None. Note, this argument will only be considered for
        VectorView type layout. Use `meg` to force using the full layout
        in situations where the info does only contain one sensor type.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any.
        If 'bads', exclude channels in info['bads'] (default).

    Returns
    -------
    layout : Layout instance | None
        None if layout not found.
    """
    our_types = ' or '.join(['`None`', '`mag`', '`grad`', '`meg`'])
    if ch_type not in (None, 'meg', 'mag', 'grad', 'eeg'):
        raise ValueError('Invalid channel type (%s) requested '
                         '`ch_type` must be %s' % (ch_type, our_types))

    (has_vv_mag, has_vv_grad, is_old_vv, has_4D_mag, ctf_other_types,
     has_CTF_grad, n_kit_grads, has_any_meg, has_eeg_coils,
     has_eeg_coils_and_meg, has_eeg_coils_only) = _get_ch_info(info)
    has_vv_meg = has_vv_mag and has_vv_grad
    has_vv_only_mag = has_vv_mag and not has_vv_grad
    has_vv_only_grad = has_vv_grad and not has_vv_mag
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
        if info['ch_names'][0].endswith('X'):
            layout_name = 'Vectorview-grad_norm'
        else:
            layout_name = 'Vectorview-grad'
    elif ((has_eeg_coils_only and ch_type in [None, 'eeg']) or
          (has_eeg_coils_and_meg and ch_type == 'eeg')):
        if not isinstance(info, (dict, Info)):
            raise RuntimeError('Cannot make EEG layout, no measurement info '
                               'was passed to `find_layout`')
        return make_eeg_layout(info, exclude=exclude)
    elif has_4D_mag:
        layout_name = 'magnesWH3600'
    elif has_CTF_grad:
        layout_name = 'CTF-275'
    elif n_kit_grads > 0:
        layout_name = _find_kit_layout(info, n_kit_grads)
    else:
        xy = _auto_topomap_coords(info, picks=range(info['nchan']),
                                  ignore_overlap=True, to_sphere=False)
        return generate_2d_layout(xy, ch_names=info['ch_names'], name='custom',
                                  normalize=False)

    layout = read_layout(layout_name)
    if not is_old_vv:
        layout.names = _clean_names(layout.names, remove_whitespace=True)
    if has_CTF_grad:
        layout.names = _clean_names(layout.names, before_dash=True)

    # Apply mask for excluded channels.
    if exclude == 'bads':
        exclude = info['bads']
    idx = [ii for ii, name in enumerate(layout.names) if name not in exclude]
    layout.names = [layout.names[ii] for ii in idx]
    layout.pos = layout.pos[idx]
    layout.ids = [layout.ids[ii] for ii in idx]

    return layout


def _find_kit_layout(info, n_grads):
    """Determine the KIT layout.

    Parameters
    ----------
    info : Info
        Info object.
    n_grads : int
        Number of KIT-gradiometers in the info.

    Returns
    -------
    kit_layout : str
        One of 'KIT-AD', 'KIT-157' or 'KIT-UMD'.
    """
    if info['kit_system_id'] is not None:
        # avoid circular import
        from ..io.kit.constants import KIT_LAYOUT

        if info['kit_system_id'] in KIT_LAYOUT:
            kit_layout = KIT_LAYOUT[info['kit_system_id']]
            if kit_layout is not None:
                return kit_layout
        raise NotImplementedError("The layout for the KIT system with ID %i "
                                  "is missing. Please contact the developers "
                                  "about adding it." % info['kit_system_id'])
    elif n_grads > 157:
        return 'KIT-AD'

    # channels which are on the left hemisphere for NY and right for UMD
    test_chs = ('MEG  13', 'MEG  14', 'MEG  15', 'MEG  16', 'MEG  25',
                'MEG  26', 'MEG  27', 'MEG  28', 'MEG  29', 'MEG  30',
                'MEG  31', 'MEG  32', 'MEG  57', 'MEG  60', 'MEG  61',
                'MEG  62', 'MEG  63', 'MEG  64', 'MEG  73', 'MEG  90',
                'MEG  93', 'MEG  95', 'MEG  96', 'MEG 105', 'MEG 112',
                'MEG 120', 'MEG 121', 'MEG 122', 'MEG 123', 'MEG 124',
                'MEG 125', 'MEG 126', 'MEG 142', 'MEG 144', 'MEG 153',
                'MEG 154', 'MEG 155', 'MEG 156')
    x = [ch['loc'][0] < 0 for ch in info['chs'] if ch['ch_name'] in test_chs]
    if np.all(x):
        return 'KIT-157'  # KIT-NY
    elif np.all(np.invert(x)):
        raise NotImplementedError("Guessing sensor layout for legacy UMD "
                                  "files is not implemented. Please convert "
                                  "your files using MNE-Python 0.13 or "
                                  "higher.")
    else:
        raise RuntimeError("KIT system could not be determined for data")


def _box_size(points, width=None, height=None, padding=0.0):
    """Given a series of points, calculate an appropriate box size.

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
    from scipy.spatial.distance import pdist

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
    """Guess the E/MEG layout and return appropriate topomap coordinates.

    Parameters
    ----------
    info : instance of Info
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


def _auto_topomap_coords(info, picks, ignore_overlap=False, to_sphere=True):
    """Make a 2 dimensional sensor map from sensor positions in an info dict.

    The default is to use the electrode locations. The fallback option is to
    attempt using digitization points of kind FIFFV_POINT_EEG. This only works
    with EEG and requires an equal number of digitization points and sensors.

    Parameters
    ----------
    info : instance of Info
        The measurement info.
    picks : list of int
        The channel indices to generate topomap coords for.
    ignore_overlap : bool
        Whether to ignore overlapping positions in the layout. If False and
        positions overlap, an error is thrown.
    to_sphere : bool
        If True, the radial distance of spherical coordinates is ignored, in
        effect fitting the xyz-coordinates to a sphere. Defaults to True.

    Returns
    -------
    locs : array, shape = (n_sensors, 2)
        An array of positions of the 2 dimensional map.
    """
    from scipy.spatial.distance import pdist, squareform

    chs = [info['chs'][i] for i in picks]

    # Use channel locations if available
    locs3d = np.array([ch['loc'][:3] for ch in chs])

    # If electrode locations are not available, use digization points
    if len(locs3d) == 0 or (~np.isfinite(locs3d)).all() or \
            np.allclose(locs3d, 0.):
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
        _, origin_head, _ = fit_sphere_to_headshape(info, dig_kinds, units='m')
        locs3d -= origin_head

        # Match the digitization points with the requested
        # channels.
        eeg_ch_locs = dict(zip(eeg_ch_names, locs3d))
        locs3d = np.array([eeg_ch_locs[ch['ch_name']] for ch in chs])

    # Duplicate points cause all kinds of trouble during visualization
    dist = pdist(locs3d)
    if len(locs3d) > 1 and np.min(dist) < 1e-10 and not ignore_overlap:
        problematic_electrodes = [
            chs[elec_i]['ch_name']
            for elec_i in squareform(dist < 1e-10).any(axis=0).nonzero()[0]
        ]

        raise ValueError('The following electrodes have overlapping positions,'
                         ' which causes problems during visualization:\n' +
                         ', '.join(problematic_electrodes))

    if to_sphere:
        # use spherical (theta, pol) as (r, theta) for polar->cartesian
        return _pol_to_cart(_cart_to_sph(locs3d)[:, 1:][:, ::-1])
    return _pol_to_cart(_cart_to_sph(locs3d))


def _topo_to_sphere(pos, eegs):
    """Transform xy-coordinates to sphere.

    Parameters
    ----------
    pos : array-like, shape (n_channels, 2)
        xy-oordinates to transform.
    eegs : list of int
        Indices of eeg channels that are included when calculating the sphere.

    Returns
    -------
    coords : array, shape (n_channels, 3)
        xyz-coordinates.
    """
    xs, ys = np.array(pos).T

    sqs = np.max(np.sqrt((xs[eegs] ** 2) + (ys[eegs] ** 2)))
    xs /= sqs  # Shape to a sphere and normalize
    ys /= sqs

    xs += 0.5 - np.mean(xs[eegs])  # Center the points
    ys += 0.5 - np.mean(ys[eegs])

    xs = xs * 2. - 1.  # Values ranging from -1 to 1
    ys = ys * 2. - 1.

    rs = np.clip(np.sqrt(xs ** 2 + ys ** 2), 0., 1.)
    alphas = np.arccos(rs)
    zs = np.sin(alphas)
    return np.column_stack([xs, ys, zs])


def _pair_grad_sensors(info, layout=None, topomap_coords=True, exclude='bads',
                       raise_error=True):
    """Find the picks for pairing grad channels.

    Parameters
    ----------
    info : instance of Info
        An info dictionary containing channel information.
    layout : Layout | None
        The layout if available. Defaults to None.
    topomap_coords : bool
        Return the coordinates for a topomap plot along with the picks. If
        False, only picks are returned. Defaults to True.
    exclude : list of str | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in info['bads']. Defaults to 'bads'.
    raise_error : bool
        Whether to raise an error when no pairs are found. If False, raises a
        warning.

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
        if raise_error:
            raise ValueError("No 'grad' channel pairs found.")
        else:
            warn("No 'grad' channel pairs found.")
            return list()

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
    """Find the indexes for pairing grad channels.

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


def _merge_grad_data(data, method='rms'):
    """Merge data from channel pairs using the RMS or mean.

    Parameters
    ----------
    data : array, shape = (n_channels, n_times)
        Data for channels, ordered in pairs.
    method : str
        Can be 'rms' or 'mean'.

    Returns
    -------
    data : array, shape = (n_channels / 2, n_times)
        The root mean square or mean for each pair.
    """
    data = data.reshape((len(data) // 2, 2, -1))
    if method == 'mean':
        data = np.mean(data, axis=1)
    elif method == 'rms':
        data = np.sqrt(np.sum(data ** 2, axis=1) / 2)
    else:
        raise ValueError('method must be "rms" or "mean, got %s.' % method)
    return data


def generate_2d_layout(xy, w=.07, h=.05, pad=.02, ch_names=None,
                       ch_indices=None, name='ecog', bg_image=None,
                       normalize=True):
    """Generate a custom 2D layout from xy points.

    Generates a 2-D layout for plotting with plot_topo methods and
    functions. XY points will be normalized between 0 and 1, where
    normalization extremes will be either the min/max of xy, or
    the width/height of bg_image.

    Parameters
    ----------
    xy : ndarray (N x 2)
        The xy coordinates of sensor locations.
    w : float
        The width of each sensor's axis (between 0 and 1)
    h : float
        The height of each sensor's axis (between 0 and 1)
    pad : float
        Portion of the box to reserve for padding. The value can range between
        0.0 (boxes will touch, default) to 1.0 (boxes consist of only padding).
    ch_names : list
        The names of each channel. Must be a list of strings, with one
        string per channel.
    ch_indices : list
        Index of each channel - must be a collection of unique integers,
        one index per channel.
    name : string
        The name of this layout type.
    bg_image : str | ndarray
        The image over which sensor axes will be plotted. Either a path to an
        image file, or an array that can be plotted with plt.imshow. If
        provided, xy points will be normalized by the width/height of this
        image. If not, xy points will be normalized by their own min/max.
    normalize : bool
        Whether to normalize the coordinates to run from 0 to 1. Defaults to
        True.

    Returns
    -------
    layout : Layout
        A Layout object that can be plotted with plot_topo
        functions and methods.

    See Also
    --------
    make_eeg_layout, make_grid_layout

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    import matplotlib.pyplot as plt
    if ch_indices is None:
        ch_indices = np.arange(xy.shape[0])
    if ch_names is None:
        ch_names = ['{0}'.format(i) for i in ch_indices]

    if len(ch_names) != len(ch_indices):
        raise ValueError('# ch names and indices must be equal')
    if len(ch_names) != len(xy):
        raise ValueError('# ch names and xy vals must be equal')

    x, y = xy.copy().astype(float).T

    # Normalize xy to 0-1
    if bg_image is not None:
        # Normalize by image dimensions
        img = plt.imread(bg_image) if isinstance(bg_image, str) else bg_image
        x /= img.shape[1]
        y /= img.shape[0]
    elif normalize:
        # Normalize x and y by their maxes
        for i_dim in [x, y]:
            i_dim -= i_dim.min(0)
            i_dim /= (i_dim.max(0) - i_dim.min(0))

    # Create box and pos variable
    box = _box_size(np.vstack([x, y]).T, padding=pad)
    box = (0, 0, box[0], box[1])
    w, h = [np.array([i] * x.shape[0]) for i in [w, h]]
    loc_params = np.vstack([x, y, w, h]).T

    layout = Layout(box, loc_params, ch_names, ch_indices, name)
    return layout
