# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

from collections.abc import Iterable
import os
import os.path as op

import numpy as np
import xml.etree.ElementTree as ElementTree

from ..viz import plot_montage
from .channels import _contains_ch_type
from ..transforms import (apply_trans, get_ras_to_neuromag_trans, _sph_to_cart,
                          _topo_to_sph, _str_to_frame, _frame_to_str)
from .._digitization import Digitization
from .._digitization._utils import (_make_dig_points, _read_dig_points,
                                    write_dig)
from ..io.pick import pick_types
from ..io.constants import FIFF
from ..utils import (warn, copy_function_doc_to_method_doc,
                     _check_option, Bunch, deprecated, _validate_type)

from .layout import _pol_to_cart, _cart_to_sph
from ._dig_montage_utils import _transform_to_head_call, _read_dig_montage_fif
from ._dig_montage_utils import _read_dig_montage_egi, _read_dig_montage_bvct
from ._dig_montage_utils import _foo_get_data_from_dig
from ._dig_montage_utils import _fix_data_fiducials

DEPRECATED_PARAM = object()


def _check_get_coord_frame(dig):
    _MSG = 'Only single coordinate frame in dig is supported'
    dig_coord_frames = set([d['coord_frame'] for d in dig])
    assert len(dig_coord_frames) == 1, _MSG
    return _frame_to_str[dig_coord_frames.pop()]


class Montage(object):
    """Montage for standard EEG electrode locations.

    .. warning:: Montages should typically be loaded from a file using
                 :func:`mne.channels.read_montage` instead of
                 instantiating this class directly.

    Parameters
    ----------
    pos : array, shape (n_channels, 3)
        The positions of the channels in 3d given in meters.
    ch_names : list
        The channel names.
    kind : str
        The type of montage (e.g. 'standard_1005').
    selection : array of int
        The indices of the selected channels in the montage file.

    See Also
    --------
    DigMontage
    read_montage
    read_dig_montage

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    def __init__(self, pos, ch_names, kind, selection,
                 nasion=None, lpa=None, rpa=None):  # noqa: D102
        self.pos = pos
        self.ch_names = ch_names
        self.kind = kind
        self.selection = selection
        self.lpa = lpa
        self.nasion = nasion
        self.rpa = rpa

    def __repr__(self):
        """Return string representation."""
        s = ('<Montage | %s - %d channels: %s ...>'
             % (self.kind, len(self.ch_names), ', '.join(self.ch_names[:3])))
        return s

    def get_pos2d(self):
        """Return positions converted to 2D."""
        return _pol_to_cart(_cart_to_sph(self.pos)[:, 1:][:, ::-1])

    @copy_function_doc_to_method_doc(plot_montage)
    def plot(self, scale_factor=20, show_names=True, kind='topomap',
             show=True):
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names, kind=kind, show=show)


def get_builtin_montages():
    """Get a list of all builtin montages.

    Returns
    -------
    montages : list
        Names of all builtin montages that can be loaded with
        :func:`read_montage`.
    """
    path = op.join(op.dirname(__file__), 'data', 'montages')
    supported = ('.elc', '.txt', '.csd', '.sfp', '.elp', '.hpts', '.loc',
                 '.locs', '.eloc', '.bvef')
    files = [op.splitext(f) for f in os.listdir(path)]
    return sorted([f for f, ext in files if ext in supported])


def read_montage(kind, ch_names=None, path=None, unit='m', transform=False):
    """Read a generic (built-in) montage.

    Individualized (digitized) electrode positions should be read in using
    :func:`read_dig_montage`.

    In most cases, you should only need to set the `kind` parameter to load one
    of the built-in montages (see Notes).

    Parameters
    ----------
    kind : str
        The name of the montage file without the file extension (e.g.
        kind='easycap-M10' for 'easycap-M10.txt'). Files with extensions
        '.elc', '.txt', '.csd', '.elp', '.hpts', '.sfp', '.loc' ('.locs' and
        '.eloc') or .bvef are supported.
    ch_names : list of str | None
        If not all electrodes defined in the montage are present in the EEG
        data, use this parameter to select a subset of electrode positions to
        load. If None (default), all defined electrode positions are returned.

        .. note:: ``ch_names`` are compared to channel names in the montage
                  file after converting them both to upper case. If a match is
                  found, the letter case in the original ``ch_names`` is used
                  in the returned montage.

    path : str | None
        The path of the folder containing the montage file. Defaults to the
        mne/channels/data/montages folder in your mne-python installation.
    unit : 'm' | 'cm' | 'mm' | 'auto'
        Unit of the input file. When 'auto' the montage is normalized to
        a sphere with a radius capturing the average head size (8.5cm).
        Defaults to 'auto'.
    transform : bool
        If True, points will be transformed to Neuromag space. The fiducials,
        'nasion', 'lpa', 'rpa' must be specified in the montage file. Useful
        for points captured using Polhemus FastSCAN. Default is False.

    Returns
    -------
    montage : instance of Montage
        The montage.

    See Also
    --------
    DigMontage
    Montage
    read_dig_montage

    Notes
    -----
    Built-in montages are not scaled or transformed by default.

    Montages can contain fiducial points in addition to electrode channels,
    e.g. ``biosemi64`` contains 67 locations. In the following table, the
    number of channels and fiducials is given in parentheses in the description
    column (e.g. 64+3 means 64 channels and 3 fiducials).

    Valid ``kind`` arguments are:

    ===================   =====================================================
    Kind                  Description
    ===================   =====================================================
    standard_1005         Electrodes are named and positioned according to the
                          international 10-05 system (343+3 locations)
    standard_1020         Electrodes are named and positioned according to the
                          international 10-20 system (94+3 locations)
    standard_alphabetic   Electrodes are named with LETTER-NUMBER combinations
                          (A1, B2, F4, ...) (65+3 locations)
    standard_postfixed    Electrodes are named according to the international
                          10-20 system using postfixes for intermediate
                          positions (100+3 locations)
    standard_prefixed     Electrodes are named according to the international
                          10-20 system using prefixes for intermediate
                          positions (74+3 locations)
    standard_primed       Electrodes are named according to the international
                          10-20 system using prime marks (' and '') for
                          intermediate positions (100+3 locations)

    biosemi16             BioSemi cap with 16 electrodes (16+3 locations)
    biosemi32             BioSemi cap with 32 electrodes (32+3 locations)
    biosemi64             BioSemi cap with 64 electrodes (64+3 locations)
    biosemi128            BioSemi cap with 128 electrodes (128+3 locations)
    biosemi160            BioSemi cap with 160 electrodes (160+3 locations)
    biosemi256            BioSemi cap with 256 electrodes (256+3 locations)

    easycap-M1            EasyCap with 10-05 electrode names (74 locations)
    easycap-M10           EasyCap with numbered electrodes (61 locations)

    EGI_256               Geodesic Sensor Net (256 locations)

    GSN-HydroCel-32       HydroCel Geodesic Sensor Net and Cz (33+3 locations)
    GSN-HydroCel-64_1.0   HydroCel Geodesic Sensor Net (64+3 locations)
    GSN-HydroCel-65_1.0   HydroCel Geodesic Sensor Net and Cz (65+3 locations)
    GSN-HydroCel-128      HydroCel Geodesic Sensor Net (128+3 locations)
    GSN-HydroCel-129      HydroCel Geodesic Sensor Net and Cz (129+3 locations)
    GSN-HydroCel-256      HydroCel Geodesic Sensor Net (256+3 locations)
    GSN-HydroCel-257      HydroCel Geodesic Sensor Net and Cz (257+3 locations)

    mgh60                 The (older) 60-channel cap used at
                          MGH (60+3 locations)
    mgh70                 The (newer) 70-channel BrainVision cap used at
                          MGH (70+3 locations)
    ===================   =====================================================

    .. versionadded:: 0.9.0
    """
    _check_option('unit', unit, ['mm', 'cm', 'm', 'auto'])

    if path is None:
        path = op.join(op.dirname(__file__), 'data', 'montages')
    if not op.isabs(kind):
        supported = ('.elc', '.txt', '.csd', '.sfp', '.elp', '.hpts', '.loc',
                     '.locs', '.eloc', '.bvef')
        montages = [op.splitext(f) for f in os.listdir(path)]
        montages = [m for m in montages if m[1] in supported and kind == m[0]]
        if len(montages) != 1:
            raise ValueError('Could not find the montage. Please provide the '
                             'full path.')
        kind, ext = montages[0]
    else:
        kind, ext = op.splitext(kind)
    fname = op.join(path, kind + ext)

    fid_names = ['lpa', 'nz', 'rpa']
    if ext == '.sfp':
        # EGI geodesic
        fid_names = ['fidt9', 'fidnz', 'fidt10']
        with open(fname, 'r') as f:
            lines = f.read().replace('\t', ' ').splitlines()

        ch_names_, pos = [], []
        for ii, line in enumerate(lines):
            line = line.strip().split()
            if len(line) > 0:  # skip empty lines
                if len(line) != 4:  # name, x, y, z
                    raise ValueError("Malformed .sfp file in line " + str(ii))
                this_name, x, y, z = line
                ch_names_.append(this_name)
                pos.append([float(cord) for cord in (x, y, z)])
        pos = np.asarray(pos)
    elif ext == '.elc':
        # 10-5 system
        ch_names_ = []
        pos = []
        with open(fname) as fid:
            # Default units are meters
            for line in fid:
                if 'UnitPosition' in line:
                    units = line.split()[1]
                    scale_factor = dict(m=1., mm=1e-3)[units]
                    break
            else:
                raise RuntimeError('Could not detect units in file %s' % fname)
            for line in fid:
                if 'Positions\n' in line:
                    break
            pos = []
            for line in fid:
                if 'Labels\n' in line:
                    break
                pos.append(list(map(float, line.split())))
            for line in fid:
                if not line or not set(line) - {' '}:
                    break
                ch_names_.append(line.strip(' ').strip('\n'))
        pos = np.array(pos) * scale_factor
    elif ext == '.txt':
        # easycap
        try:  # newer version
            data = np.genfromtxt(fname, dtype='str', skip_header=1)
        except TypeError:
            data = np.genfromtxt(fname, dtype='str', skiprows=1)
        ch_names_ = data[:, 0].tolist()
        az = np.deg2rad(data[:, 2].astype(float))
        pol = np.deg2rad(data[:, 1].astype(float))
        rad = np.ones(len(az))  # spherical head model
        rad *= 85.  # scale up to realistic head radius (8.5cm == 85mm)
        pos = _sph_to_cart(np.array([rad, az, pol]).T)
    elif ext == '.csd':
        # CSD toolbox
        try:  # newer version
            data = np.genfromtxt(fname, dtype='str', skip_header=2)
        except TypeError:
            data = np.genfromtxt(fname, dtype='str', skiprows=2)

        ch_names_ = data[:, 0].tolist()
        az = np.deg2rad(data[:, 1].astype(float))
        pol = np.deg2rad(90. - data[:, 2].astype(float))
        pos = _sph_to_cart(np.array([np.ones(len(az)), az, pol]).T)
    elif ext == '.elp':
        # standard BESA spherical
        dtype = np.dtype('S8, S8, f8, f8, f8')
        try:
            data = np.loadtxt(fname, dtype=dtype, skip_header=1)
        except TypeError:
            data = np.loadtxt(fname, dtype=dtype, skiprows=1)

        ch_names_ = data['f1'].astype(str).tolist()
        az = data['f2']
        horiz = data['f3']
        radius = np.abs(az / 180.)
        az = np.deg2rad(np.array([h if a >= 0. else 180 + h
                                  for h, a in zip(horiz, az)]))
        pol = radius * np.pi
        rad = np.ones(len(az))  # spherical head model
        rad *= 85.  # scale up to realistic head radius (8.5cm == 85mm)
        pos = _sph_to_cart(np.array([rad, az, pol]).T)
    elif ext == '.hpts':
        # MNE-C specified format for generic digitizer data
        fid_names = ['1', '2', '3']
        dtype = [('type', 'S8'), ('name', 'S8'),
                 ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        data = np.loadtxt(fname, dtype=dtype)
        ch_names_ = data['name'].astype(str).tolist()
        pos = np.vstack((data['x'], data['y'], data['z'])).T
    elif ext in ('.loc', '.locs', '.eloc'):
        ch_names_ = np.genfromtxt(fname, dtype=str, usecols=3).tolist()
        topo = np.loadtxt(fname, dtype=float, usecols=[1, 2])
        sph = _topo_to_sph(topo)
        pos = _sph_to_cart(sph)
        pos[:, [0, 1]] = pos[:, [1, 0]] * [-1, 1]
    elif ext == '.bvef':
        # 'BrainVision Electrodes File' format
        # Based on BrainVision Analyzer coordinate system: Defined between
        # standard electrode positions: X-axis from T7 to T8, Y-axis from Oz to
        # Fpz, Z-axis orthogonal from XY-plane through Cz, fit to a sphere if
        # idealized (when radius=1), specified in millimeters
        if unit not in ['auto', 'mm']:
            raise ValueError('`unit` must be "auto" or "mm" for .bvef files.')
        root = ElementTree.parse(fname).getroot()
        ch_names_ = [s.text for s in root.findall("./Electrode/Name")]
        theta = [float(s.text) for s in root.findall("./Electrode/Theta")]
        pol = np.deg2rad(np.array(theta))
        phi = [float(s.text) for s in root.findall("./Electrode/Phi")]
        az = np.deg2rad(np.array(phi))
        rad = [float(s.text) for s in root.findall("./Electrode/Radius")]
        rad = np.array(rad)  # specified in mm
        if set(rad) == set([1]):
            # idealized montage (spherical head model), scale up to realistic
            # head radius (8.5cm == 85mm)
            rad = np.array(rad) * 85.
        pos = _sph_to_cart(np.array([rad, az, pol]).T)
    else:
        raise ValueError('Currently the "%s" template is not supported.' %
                         kind)
    selection = np.arange(len(pos))

    if unit == 'auto':  # rescale to realistic head radius in meters: 0.085
        pos -= np.mean(pos, axis=0)
        pos = 0.085 * (pos / np.linalg.norm(pos, axis=1).mean())
    elif unit == 'mm':
        pos /= 1e3
    elif unit == 'cm':
        pos /= 1e2
    elif unit == 'm':  # montage is supposed to be in m
        pass

    names_lower = [name.lower() for name in list(ch_names_)]
    fids = {key: pos[names_lower.index(fid_names[ii])]
            if fid_names[ii] in names_lower else None
            for ii, key in enumerate(['lpa', 'nasion', 'rpa'])}
    if transform:
        missing = [name for name, val in fids.items() if val is None]
        if missing:
            raise ValueError("The points %s are missing, but are needed "
                             "to transform the points to the MNE coordinate "
                             "system. Either add the points, or read the "
                             "montage with transform=False. " % missing)
        neuromag_trans = get_ras_to_neuromag_trans(
            fids['nasion'], fids['lpa'], fids['rpa'])
        pos = apply_trans(neuromag_trans, pos)
    fids = {key: pos[names_lower.index(fid_names[ii])]
            if fid_names[ii] in names_lower else None
            for ii, key in enumerate(['lpa', 'nasion', 'rpa'])}

    if ch_names is not None:
        # Ensure channels with differing case are found.
        upper_names = [ch_name.upper() for ch_name in ch_names]
        sel, ch_names_ = zip(*[(i, ch_names[upper_names.index(e)]) for i, e in
                               enumerate([n.upper() for n in ch_names_])
                               if e in upper_names])
        sel = list(sel)
        pos = pos[sel]
        selection = selection[sel]
    kind = op.split(kind)[-1]
    return Montage(pos=pos, ch_names=ch_names_, kind=kind, selection=selection,
                   lpa=fids['lpa'], nasion=fids['nasion'], rpa=fids['rpa'])


def make_dig_montage(ch_pos=None, nasion=None, lpa=None, rpa=None,
                     hsp=None, hpi=None, hpi_dev=None, coord_frame='unknown',
                     transform_to_head=False, compute_dev_head_t=False):
    r"""Make montage from arrays.

    Parameters
    ----------
    ch_pos : dict
        Dictionary of channel positions. Keys are channel names and values
        are 3D coordinates - array of shape (3,) - in native digitizer space
        in m.
    nasion : None | array, shape (3,)
        The position of the nasion fiducial point.
        This point is assumed to be in the native digitizer space in m.
    lpa : None | array, shape (3,)
        The position of the left periauricular fiducial point.
        This point is assumed to be in the native digitizer space in m.
    rpa : None | array, shape (3,)
        The position of the right periauricular fiducial point.
        This point is assumed to be in the native digitizer space in m.
    hsp : None | array, shape (n_points, 3)
        This corresponds to an array of positions of the headshape points in
        3d. These points are assumed to be in the native digitizer space in m.
    hpi : None | array, shape (n_hpi, 3)
        This corresponds to an array of HPI points in the native digitizer
        space. They only necessary if computation of a ``compute_dev_head_t``
        is True.
    hpi_dev : None | array, shape (n_hpi, 3)
        This corresponds to an array of HPI points. These points are in device
        space, and are only necessary if computation of a
        ``compute_dev_head_t`` is True.
    coord_frame : str
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space.
    transform_to_head : bool
        If True (default), points will be transformed to Neuromag head space.
        The fiducials (nasion, lpa, and rpa) must be specified. This is useful
        for points captured using a device that does not automatically convert
        points to Neuromag head coordinates
        (e.g., Polhemus FastSCAN).
    compute_dev_head_t : bool
        If True, a Dev-to-Head transformation matrix will be added to the
        montage. To get a proper `dev_head_t`, the hpi and the hpi_dev points
        must be in the same order. If False (default), no transformation will
        be added to the montage.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    Montage
    read_montage
    DigMontage
    read_dig_montage

    """
    # XXX: hpi was historically elp
    # XXX: hpi_dev was historically hpi
    assert coord_frame in ('unknown', 'head')
    from ..coreg import fit_matched_points
    data = Bunch(
        nasion=nasion, lpa=lpa, rpa=rpa,
        elp=hpi, dig_ch_pos=ch_pos, hsp=hsp,
        coord_frame=coord_frame,
    )
    if transform_to_head:
        data = _transform_to_head_call(data)

    if compute_dev_head_t:
        if data.elp is None or hpi_dev is None:
            raise RuntimeError('must have both elp and hpi to compute the '
                               'device to head transform')
        else:
            # here is hpi
            dev_head_t = fit_matched_points(
                tgt_pts=data.elp, src_pts=hpi_dev, out='trans'
            )  # XXX: shall we make it a Transform? rather than np.array
    else:
        dev_head_t = None

    ch_names = list() if ch_pos is None else list(sorted(ch_pos.keys()))
    dig = _make_dig_points(
        nasion=data.nasion, lpa=data.lpa, rpa=data.rpa, hpi=data.elp,
        extra_points=data.hsp, dig_ch_pos=data.dig_ch_pos,
        coord_frame=data.coord_frame,
    )
    return DigMontage(dig=dig, ch_names=ch_names, dev_head_t=dev_head_t)


class DigMontage(object):
    """Montage for digitized electrode and headshape position data.

    .. warning:: Montages are typically loaded from a file using
                 :func:`read_dig_montage` instead of instantiating
                 this class.

    Parameters
    ----------
    hsp : array, shape (n_points, 3)
        The positions of the headshape points in 3d.
        These points are in the native digitizer space.
        Deprecated, will be removed in 0.20.
    hpi : array, shape (n_hpi, 3)
        The positions of the head-position indicator coils in 3d.
        These points are in the MEG device space.
        Deprecated, will be removed in 0.20.
    elp : array, shape (n_hpi, 3)
        The positions of the head-position indicator coils in 3d.
        This is typically in the native digitizer space.
        Deprecated, will be removed in 0.20.
    point_names : list, shape (n_elp)
        The names of the digitized points for hpi and elp.
        Deprecated, will be removed in 0.20.
    nasion : array, shape (1, 3)
        The position of the nasion fiducial point.
        Deprecated, will be removed in 0.20.
    lpa : array, shape (1, 3)
        The position of the left periauricular fiducial point.
        Deprecated, will be removed in 0.20.
    rpa : array, shape (1, 3)
        The position of the right periauricular fiducial point.
        Deprecated, will be removed in 0.20.
    dev_head_t : array, shape (4, 4)
        A Device-to-Head transformation matrix.
    dig_ch_pos : dict
        Dictionary of channel positions given in meters.
        Deprecated, will be removed in 0.20.

        .. versionadded:: 0.12

    coord_frame : str
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space.

        .. versionadded:: 0.19

    dig : list of dict
        The object containing all the dig points.
    ch_names : list of str
        The names of the EEG channels.

    See Also
    --------
    Montage
    read_dig_montage
    read_montage

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    def __init__(self,
        hsp=DEPRECATED_PARAM, hpi=DEPRECATED_PARAM, elp=DEPRECATED_PARAM,
        point_names=DEPRECATED_PARAM, nasion=DEPRECATED_PARAM,
        lpa=DEPRECATED_PARAM, rpa=DEPRECATED_PARAM,
        dev_head_t=None, dig_ch_pos=DEPRECATED_PARAM,
        coord_frame=DEPRECATED_PARAM,
        dig=None, ch_names=None,
    ):  # noqa: D102
        # XXX: dev_head_t now is np.array, we should add dev_head_transform
        #      (being instance of Transformation) and move the parameter to the
        #      end of the call.
        _non_deprecated_kwargs = [
            key for key, val in dict(
                hsp=hsp, hpi=hpi, elp=elp, point_names=point_names,
                nasion=nasion, lpa=lpa, rpa=rpa,
                dig_ch_pos=dig_ch_pos, coord_frame=coord_frame,
            ).items() if val is not DEPRECATED_PARAM
        ]
        if not _non_deprecated_kwargs:
            _validate_type(item=dig, types=Digitization,
                           item_name='dig', type_name='Digitization')
            ch_names = list() if ch_names is None else ch_names
            n_eeg = sum([1 for d in dig if d['kind'] == FIFF.FIFFV_POINT_EEG])
            if n_eeg != len(ch_names):
                raise ValueError(
                    'The number of EEG channels (%d) does not match the number'
                    ' of channel names provided (%d)' % (n_eeg, len(ch_names))
                )

            self.dev_head_t = dev_head_t
            self.dig = dig
            self.ch_names = ch_names
            self._coord_frame = _check_get_coord_frame(self.dig)
        else:
            # Deprecated
            _msg = (
                "Using {params} in DigMontage constructor is deprecated."
                " Use 'dig', and 'ch_names' instead."
            ).format(params=", ".join(
                ["'{}'".format(k) for k in _non_deprecated_kwargs]
            ))
            warn(_msg, DeprecationWarning)

            # Restore old defaults
            hsp = None if hsp is DEPRECATED_PARAM else hsp
            hpi = None if hpi is DEPRECATED_PARAM else hpi
            elp = None if elp is DEPRECATED_PARAM else elp
            nasion = None if nasion is DEPRECATED_PARAM else nasion
            lpa = None if lpa is DEPRECATED_PARAM else lpa
            rpa = None if rpa is DEPRECATED_PARAM else rpa
            dig_ch_pos = None if dig_ch_pos is DEPRECATED_PARAM else dig_ch_pos
            coord_frame = \
                'unknown' if coord_frame is DEPRECATED_PARAM else coord_frame
            point_names = \
                None if point_names is DEPRECATED_PARAM else point_names

            # Old behavior
            if elp is not None:
                if not isinstance(point_names, Iterable):
                    raise TypeError('If elp is specified, point_names must'
                                    ' provide a list of str with one entry per'
                                    ' ELP point.')
                point_names = list(point_names)
                if len(point_names) != len(elp):
                    raise ValueError('elp contains %i points but %i '
                                     'point_names were specified.' %
                                     (len(elp), len(point_names)))

            self.dev_head_t = dev_head_t
            self._point_names = point_names
            self.ch_names = \
                [] if dig_ch_pos is None else list(sorted(dig_ch_pos.keys()))
            self._hpi = hpi
            self._coord_frame = coord_frame
            self.dig = _make_dig_points(
                nasion=nasion, lpa=lpa, rpa=rpa, hpi=elp,
                extra_points=hsp, dig_ch_pos=dig_ch_pos,
                coord_frame=self._coord_frame,
            )

    @property
    def point_names(self):
        warn('"point_names" attribute is deprecated and will be removed'
             ' in v0.20', DeprecationWarning)
        return self._point_names

    @property
    def coord_frame(self):
        warn('"coord_frame" attribute is deprecated and will be removed'
             ' in v0.20', DeprecationWarning)
        return self._coord_frame

    def __repr__(self):
        """Return string representation."""
        _data = _foo_get_data_from_dig(self.dig)
        s = ('<DigMontage | %d extras (headshape), %d HPIs, %d fiducials, %d '
             'channels>' %
             (len(_data.hsp) if _data.hsp is not None else 0,
              len(_data.hpi) if _data.hpi is not None else 0,
              sum(x is not None for x in (_data.lpa, _data.rpa, _data.nasion)),
              len(_data.dig_ch_pos_location) if _data.dig_ch_pos_location is not None else 0,))  # noqa
        return s

    @copy_function_doc_to_method_doc(plot_montage)
    def plot(self, scale_factor=20, show_names=False, kind='3d', show=True):
        # XXX: plot_montage takes an empty info and sets 'self'
        #      Therefore it should not be a representation problem.
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names, kind=kind, show=show)

    def transform_to_head(self):
        """Transform digitizer points to Neuromag head coordinates."""
        raise RuntimeError('The transform_to_head method has been removed to '
                           'enforce that DigMontage are constructed already '
                           'in the correct coordinate system. This method '
                           'will disappear in version 0.20.')

    @deprecated(
        'compute_dev_head_t is deprecated and will be removed in 0.20.'
    )
    def compute_dev_head_t(self):
        """Compute the Neuromag dev_head_t from matched points."""
        if not hasattr(self, '_hpi'):
            raise RuntimeError(
                'Cannot compute dev_head_t if DigMontage was not created'
                ' from arrays')

        from ..coreg import fit_matched_points
        data = _foo_get_data_from_dig(self.dig)
        if data.elp is None or self._hpi is None:
            raise RuntimeError('must have both elp and hpi to compute the '
                               'device to head transform')
        self.dev_head_t = fit_matched_points(tgt_pts=data.elp,
                                             src_pts=self._hpi, out='trans')

    def save(self, fname):
        """Save digitization points to FIF.

        Parameters
        ----------
        fname : str
            The filename to use. Should end in .fif or .fif.gz.
        """
        if self._coord_frame != 'head':
            raise RuntimeError('Can only write out digitization points in '
                               'head coordinates.')
        write_dig(fname, self.dig)

    @property
    def dig_ch_pos(self):
        warn('"dig_ch_pos" attribute is deprecated and will be removed in '
             'v0.20', DeprecationWarning)
        return self._ch_pos()

    def _get_ch_pos(self):
        return dict(zip(self.ch_names,
                        _foo_get_data_from_dig(self.dig).dig_ch_pos_location))

    @property
    def elp(self):
        warn('"elp" attribute is deprecated and will be removed in v0.20',
             DeprecationWarning)
        return _foo_get_data_from_dig(self.dig).elp

    @property
    def hpi(self):
        warn('"hpi" attribute is deprecated and will be removed in v0.20',
             DeprecationWarning)
        return getattr(self, '_hpi', None)

    @property
    def hsp(self):
        warn('"hsp" attribute is deprecated and will be removed in v0.20',
             DeprecationWarning)
        return _foo_get_data_from_dig(self.dig).hsp

    @property
    def lpa(self):
        warn('"lpa" attribute is deprecated and will be removed in v0.20',
             DeprecationWarning)
        return _foo_get_data_from_dig(self.dig).lpa

    @property
    def rpa(self):
        warn('"rpa" attribute is deprecated and will be removed in v0.20',
             DeprecationWarning)
        return _foo_get_data_from_dig(self.dig).rpa

    @property
    def nasion(self):
        warn('"nasion" attribute is deprecated and will be removed in v0.20',
             DeprecationWarning)
        return _foo_get_data_from_dig(self.dig).nasion


def _check_frame(d, frame_str):
    """Check coordinate frames."""
    if d['coord_frame'] != _str_to_frame[frame_str]:
        raise RuntimeError('dig point must be in %s coordinate frame, got %s'
                           % (frame_str, _frame_to_str[d['coord_frame']]))


def _get_scaling(unit, scale):
    if unit not in scale:
        raise ValueError("Unit needs to be one of %s, not %r" %
                         (sorted(scale.keys()), unit))
    else:
        return scale[unit]


def read_dig_montage(hsp=None, hpi=None, elp=None,
                     point_names=None, unit='auto', fif=None, egi=None,
                     bvct=None, transform=True, dev_head_t=False, ):
    r"""Read subject-specific digitization montage from a file.

    Parameters
    ----------
    hsp : None | str | array, shape (n_points, 3)
        If str, this corresponds to the filename of the headshape points.
        This is typically used with the Polhemus FastSCAN system.
        If numpy.array, this corresponds to an array of positions of the
        headshape points in 3d. These points are assumed to be in the native
        digitizer space and will be rescaled according to the unit parameter.
    hpi : None | str | array, shape (n_hpi, 3)
        If str, this corresponds to the filename of Head Position Indicator
        (HPI) points. If numpy.array, this corresponds to an array
        of HPI points. These points are in device space, and are only
        necessary if computation of a ``dev_head_t`` by the
        :class:`DigMontage` is required.
    elp : None | str | array, shape (n_fids + n_hpi, 3)
        If str, this corresponds to the filename of electrode position
        points. This is typically used with the Polhemus FastSCAN system.
        If numpy.array, this corresponds to an array of digitizer points in
        the same order. These points are assumed to be in the native digitizer
        space and will be rescaled according to the unit parameter.
    point_names : None | list
        A list of point names for elp (required if elp is defined).
        Typically this would be like::

            ('nasion', 'lpa', 'rpa', 'CHPI001', 'CHPI002', 'CHPI003')

    unit : 'auto' | 'm' | 'cm' | 'mm'
        Unit of the digitizer files (hsp and elp). If not 'm', coordinates will
        be rescaled to 'm'. Default is 'auto', which assumes 'm' for \*.hsp and
        \*.elp files and 'mm' for \*.txt files, corresponding to the known
        Polhemus export formats.
    fif : str | None
        FIF file from which to read digitization locations.
        If str (filename), all other arguments are ignored.

        .. versionadded:: 0.12

    egi : str | None
        EGI MFF XML coordinates file from which to read digitization locations.
        If str (filename), all other arguments are ignored.

        .. versionadded:: 0.14

    bvct : None | str
        BrainVision CapTrak coordinates file from which to read digitization
        locations. This is typically in XML format. If str (filename), all
        other arguments are ignored.

        .. versionadded:: 0.19
    transform : bool
        If True (default), points will be transformed to Neuromag space
        using :meth:`DigMontage.transform_to_head`.
        The fiducials (nasion, lpa, and rpa) must be specified.
        This is useful for points captured using a device that does
        not automatically convert points to Neuromag head coordinates
        (e.g., Polhemus FastSCAN).
    dev_head_t : bool
        If True, a Dev-to-Head transformation matrix will be added to the
        montage using :meth:`DigMontage.compute_dev_head_t`.
        To get a proper `dev_head_t`, the hpi and the elp points
        must be in the same order. If False (default), no transformation
        will be added to the montage.

    Returns
    -------
    montage : instance of DigMontage
        The digitizer montage.

    See Also
    --------
    DigMontage
    Montage
    read_montage

    Notes
    -----
    All digitized points will be transformed to head-based coordinate system
    if transform is True and fiducials are present.

    .. versionadded:: 0.9.0
    """
    # XXX: This scaling business seems really dangerous to me.
    EGI_SCALE = dict(mm=1e-3, cm=1e-2, auto=1e-2, m=1)
    NUMPY_DATA_SCALE = dict(mm=1e-3, cm=1e-2, auto=1e-3, m=1)

    if fif is not None:
        _raise_transform_err = True if dev_head_t or not transform else False
        data = _read_dig_montage_fif(
            fname=fif,
            _raise_transform_err=_raise_transform_err,
            _all_data_kwargs_are_none=all(
                x is None for x in (hsp, hpi, elp, point_names, egi, bvct))
        )

    elif egi is not None:
        data = _read_dig_montage_egi(
            fname=egi,
            _scaling=_get_scaling(unit, EGI_SCALE),
            _all_data_kwargs_are_none=all(
                x is None for x in (hsp, hpi, elp, point_names, fif, bvct))
        )

    elif bvct is not None:
        data = _read_dig_montage_bvct(
            fname=bvct,
            unit=unit,  # XXX: this should change
            _all_data_kwargs_are_none=all(
                x is None for x in (hsp, hpi, elp, point_names, fif, egi))
        )

    else:
        # XXX: This should also become a function
        _scaling = _get_scaling(unit, NUMPY_DATA_SCALE),
        # HSP
        if isinstance(hsp, str):
            hsp = _read_dig_points(hsp, unit=unit)
        elif hsp is not None:
            hsp *= _scaling

        # HPI
        if isinstance(hpi, str):
            ext = op.splitext(hpi)[-1]
            if ext in ('.txt', '.mat'):
                hpi = _read_dig_points(hpi, unit='m')
            elif ext in ('.sqd', '.mrk'):
                from ..io.kit import read_mrk
                hpi = read_mrk(hpi)
            else:
                raise ValueError('HPI file with extension *%s is not '
                                 'supported. Only *.txt, *.sqd and *.mrk are '
                                 'supported.' % ext)

        # ELP
        if isinstance(elp, str):
            elp = _read_dig_points(elp, unit=unit)
        elif elp is not None:
            elp *= _scaling

        data = Bunch(
            nasion=None, lpa=None, rpa=None,
            hsp=hsp, elp=elp, coord_frame='unknown',
            dig_ch_pos=None, hpi=hpi, point_names=point_names,
        )

    if any(x is None for x in (data.nasion, data.rpa, data.lpa)) and transform:
        data = _fix_data_fiducials(data)

    point_names = data.pop('point_names')
    data['hpi_dev'] = data['hpi']
    data['hpi'] = data.pop('elp')
    data['ch_pos'] = data.pop('dig_ch_pos')
    montage = make_dig_montage(
        **data, transform_to_head=transform,
        compute_dev_head_t=dev_head_t,
    )

    montage._point_names = point_names  # XXX: hack this should go!!
    return montage


def _set_montage(info, montage, update_ch_names=False, set_dig=True):
    """Apply montage to data.

    With a Montage, this function will replace the EEG channel names and
    locations with the values specified for the particular montage.

    With a DigMontage, this function will replace the digitizer info with
    the values specified for the particular montage.

    Usually, a montage is expected to contain the positions of all EEG
    electrodes and a warning is raised when this is not the case.

    Parameters
    ----------
    info : instance of Info
        The measurement info to update.
    montage : instance of Montage | instance of DigMontage | str | None
        The montage to apply (None removes any location information). If
        montage is a string, a builtin montage with that name will be used.
    update_ch_names : bool
        If True, overwrite the info channel names with the ones from montage.
        Defaults to False.

    Notes
    -----
    This function will change the info variable in place.
    """
    if isinstance(montage, str):  # load builtin montage
        montage = read_montage(montage)

    if isinstance(montage, Montage):
        if update_ch_names:
            for ii, (ch, ch_name) in \
                    enumerate(zip(info['chs'], montage.ch_names)):
                ch_info = {'cal': 1., 'logno': ii + 1, 'scanno': ii + 1,
                           'range': 1.0, 'unit_mul': 0, 'ch_name': ch_name,
                           'unit': FIFF.FIFF_UNIT_V, 'kind': FIFF.FIFFV_EEG_CH,
                           'coord_frame': FIFF.FIFFV_COORD_HEAD,
                           'coil_type': FIFF.FIFFV_COIL_EEG}
                ch.update(ch_info)
            info._update_redundant()

        if not _contains_ch_type(info, 'eeg'):
            raise ValueError('No EEG channels found.')

        # If there are no name collisions, match channel names in a case
        # insensitive manner.
        montage_lower = [ch_name.lower() for ch_name in montage.ch_names]
        info_lower = [ch_name.lower() for ch_name in info['ch_names']]
        if (len(set(montage_lower)) == len(montage_lower) and
                len(set(info_lower)) == len(info_lower)):
            montage_ch_names = montage_lower
            info_ch_names = info_lower
        else:
            montage_ch_names = montage.ch_names
            info_ch_names = info['ch_names']

        info_ch_names = [name.replace(' ', '') for name in info_ch_names]
        montage_ch_names = [name.replace(' ', '') for name in montage_ch_names]

        dig = dict()
        for pos, ch_name in zip(montage.pos, montage_ch_names):
            if ch_name not in info_ch_names:
                continue

            ch_idx = info_ch_names.index(ch_name)
            info['chs'][ch_idx]['loc'] = np.r_[pos, [0.] * 9]
            info['chs'][ch_idx]['coord_frame'] = FIFF.FIFFV_COORD_HEAD
            dig[ch_idx] = pos
        if set_dig:
            info['dig'] = _make_dig_points(
                nasion=montage.nasion, lpa=montage.lpa, rpa=montage.rpa,
                dig_ch_pos=dig)
        if len(dig) == 0:
            raise ValueError('None of the sensors defined in the montage were '
                             'found in the info structure. Check the channel '
                             'names.')

        eeg_sensors = pick_types(info, meg=False, ref_meg=False, eeg=True,
                                 exclude=[])
        not_found = np.setdiff1d(eeg_sensors, list(dig.keys()))
        if len(not_found) > 0:
            not_found_names = [info['ch_names'][ch] for ch in not_found]
            warn('The following EEG sensors did not have a position '
                 'specified in the selected montage: ' +
                 str(not_found_names) + '. Their position has been '
                 'left untouched.')

    elif isinstance(montage, DigMontage):

        if set_dig:
            info['dig'] = montage.dig

        if montage.dev_head_t is not None:
            info['dev_head_t']['trans'] = montage.dev_head_t

        if montage.ch_names:  # update channel positions, too
            dig_ch_pos = dict(zip(montage.ch_names, [
                d['r'] for d in montage.dig
                if d['kind'] == FIFF.FIFFV_POINT_EEG
            ]))
            eeg_ref_pos = dig_ch_pos.get('EEG000', np.zeros(3))
            did_set = np.zeros(len(info['ch_names']), bool)
            is_eeg = np.zeros(len(info['ch_names']), bool)
            is_eeg[pick_types(info, meg=False, eeg=True, exclude=())] = True

            for ch_name, ch_pos in dig_ch_pos.items():
                if ch_name == 'EEG000':  # what if eeg ref. has different name?
                    continue
                if ch_name not in info['ch_names']:
                    raise RuntimeError('Montage channel %s not found in info'
                                       % ch_name)
                idx = info['ch_names'].index(ch_name)
                did_set[idx] = True
                this_loc = np.concatenate((ch_pos, eeg_ref_pos))
                info['chs'][idx]['loc'][:6] = this_loc

            did_not_set = [info['chs'][ii]['ch_name']
                           for ii in np.where(is_eeg & ~did_set)[0]]

            if len(did_not_set) > 0:
                warn('Did not set %s channel positions:\n%s'
                     % (len(did_not_set), ', '.join(did_not_set)))

    elif montage is None:
        for ch in info['chs']:
            ch['loc'] = np.full(12, np.nan)
        if set_dig:
            info['dig'] = None
    else:
        raise TypeError("Montage must be a 'Montage', 'DigMontage', 'str' or "
                        "'None' instead of '%s'." % type(montage))
