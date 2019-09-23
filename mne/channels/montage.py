# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
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

from collections import OrderedDict
from collections.abc import Iterable
import os
import os.path as op
import re
from copy import deepcopy
from itertools import takewhile, chain
from functools import partial

import numpy as np
import xml.etree.ElementTree as ElementTree

from ..viz import plot_montage
from .channels import _contains_ch_type
from ..transforms import (apply_trans, get_ras_to_neuromag_trans, _sph_to_cart,
                          _topo_to_sph, _frame_to_str, _str_to_frame,
                          Transform)
from .._digitization import Digitization
from .._digitization.base import _count_points_by_type
from .._digitization.base import _get_dig_eeg
from .._digitization._utils import (_make_dig_points, _read_dig_points,
                                    write_dig, _read_dig_fif,
                                    _format_dig_points)
from ..io.pick import pick_types
from ..io.open import fiff_open
from ..io.constants import FIFF
from ..utils import (warn, logger, copy_function_doc_to_method_doc,
                     _check_option, Bunch, deprecated, _validate_type,
                     _check_fname)
from .._digitization._utils import _get_fid_coords, _foo_get_data_from_dig

from .layout import _pol_to_cart, _cart_to_sph
from ._dig_montage_utils import _transform_to_head_call
from ._dig_montage_utils import _read_dig_montage_egi, _read_dig_montage_bvct
from ._dig_montage_utils import _fix_data_fiducials
from ._dig_montage_utils import _parse_brainvision_dig_montage

from .channels import DEPRECATED_PARAM

HEAD_SIZE_DEFAULT = 0.095  # in [m]

_BUILT_IN_MONTAGES = [
    'EGI_256',
    'GSN-HydroCel-128', 'GSN-HydroCel-129', 'GSN-HydroCel-256',
    'GSN-HydroCel-257', 'GSN-HydroCel-32', 'GSN-HydroCel-64_1.0',
    'GSN-HydroCel-65_1.0',
    'biosemi128', 'biosemi16', 'biosemi160', 'biosemi256',
    'biosemi32', 'biosemi64',
    'easycap-M1', 'easycap-M10',
    'mgh60', 'mgh70',
    'standard_1005', 'standard_1020', 'standard_alphabetic',
    'standard_postfixed', 'standard_prefixed', 'standard_primed'
]


def _check_get_coord_frame(dig):
    _MSG = 'Only single coordinate frame in dig is supported'
    dig_coord_frames = set([d['coord_frame'] for d in dig])
    assert len(dig_coord_frames) <= 1, _MSG
    return _frame_to_str[dig_coord_frames.pop()] if dig_coord_frames else None


def _check_ch_names_are_compatible(info_names, montage_names, raise_if_subset):
    assert isinstance(info_names, set) and isinstance(montage_names, set)

    match_set = info_names & montage_names
    not_in_montage = info_names - montage_names
    not_in_info = montage_names - info_names

    if len(not_in_montage):  # Montage is subset of info
        if raise_if_subset:
            raise ValueError((
                'DigMontage is a only a subset of info.'
                ' There are {n_ch} channel positions not present it the'
                ' DigMontage. The required channels are: {ch_names}.'

                # XXX: the rest of the message is deprecated. to remove in 0.20
                '\nYou can use `raise_if_subset=False` in `set_montage` to'
                ' avoid this ValueError and get a DeprecationWarning instead.'
            ).format(n_ch=len(not_in_montage), ch_names=not_in_montage))
        else:
            # XXX: deprecated. to remove in 0.20 (raise_if_subset, too)
            warn('DigMontage is a only a subset of info.'
                 ' Did not set %s channel positions:\n%s'
                 % (len(not_in_montage), ', '.join(not_in_montage)),
                 RuntimeWarning)
    else:
        pass  # noqa

    if len(not_in_info):  # Montage is superset of info
        logger.info((
            'DigMontage is a superset of info. {n_ch} in DigMontage will be'
            ' ignored. The ignored channels are: {ch_names}'
        ).format(n_ch=len(not_in_info), ch_names=not_in_info))

    else:
        pass  # noqa

    return match_set


@deprecated(
    'Montage class is deprecated and will be removed in v0.20.'
    ' Please use DigMontage instead.'
)
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
    return _BUILT_IN_MONTAGES


@deprecated(
    '``read_montage`` is deprecated and will be removed in v0.20. Please use'
    ' ``read_dig_fif``, ``read_dig_egi``, ``read_custom_montage``,'
    ' or ``read_dig_captrack``'
    ' to read a digitization based on your needs instead;'
    ' or ``make_standard_montage`` to create ``DigMontage`` based on template;'
    ' or ``make_dig_montage`` to create a ``DigMontage`` out of np.arrays'
)
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
        data = np.genfromtxt(fname, dtype='str', skip_header=1)
        ch_names_ = data[:, 0].tolist()
        az = np.deg2rad(data[:, 2].astype(float))
        pol = np.deg2rad(data[:, 1].astype(float))
        rad = np.ones(len(az))  # spherical head model
        rad *= 85.  # scale up to realistic head radius (8.5cm == 85mm)
        pos = _sph_to_cart(np.array([rad, az, pol]).T)
    elif ext == '.csd':
        # CSD toolbox
        data = np.genfromtxt(fname, dtype='str', skip_header=2)
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
    # XXX: This code was duplicated !!
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
                     hsp=None, hpi=None, coord_frame='unknown'):
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
    coord_frame : str
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    read_dig_captrack
    read_dig_egi
    read_dig_fif
    read_dig_polhemus_isotrak
    """
    if ch_pos is None:
        ch_names = None
    else:
        ch_names = list(ch_pos.keys())
        if not isinstance(ch_pos, OrderedDict):
            ch_names = sorted(ch_names)
    dig = _make_dig_points(
        nasion=nasion, lpa=lpa, rpa=rpa, hpi=hpi, extra_points=hsp,
        dig_ch_pos=ch_pos, coord_frame=coord_frame
    )

    return DigMontage(dig=dig, ch_names=ch_names)


# XXX : should be kill one read_dig_montage is removed in 0.20
def _make_dig_montage(ch_pos=None, nasion=None, lpa=None, rpa=None,
                      hsp=None, hpi=None, hpi_dev=None, coord_frame='unknown',
                      transform_to_head=False, compute_dev_head_t=False):
    # XXX: hpi was historically elp
    # XXX: hpi_dev was historically hpi
    assert coord_frame in _str_to_frame
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
        dig=None, ch_names=None,
    ):  # noqa: D102
        # XXX: dev_head_t now is np.array, we should add dev_head_transform
        #      (being instance of Transformation) and move the parameter to the
        #      end of the call.
        _non_deprecated_kwargs = [
            key for key, val in dict(
                hsp=hsp, hpi=hpi, elp=elp, point_names=point_names,
                nasion=nasion, lpa=lpa, rpa=rpa,
                dig_ch_pos=dig_ch_pos,
            ).items() if val is not DEPRECATED_PARAM
        ]
        if not _non_deprecated_kwargs:
            dig = Digitization() if dig is None else dig
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
            coord_frame = 'unknown'
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
        n_points = _count_points_by_type(self.dig)
        return ('<DigMontage | {extra:d} extras (headshape), {hpi:d} HPIs,'
                ' {fid:d} fiducials, {eeg:d} channels>').format(**n_points)

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

    def __iadd__(self, other):
        """Add two DigMontages in place.

        Notes
        -----
        Two DigMontages can only be added if there are no duplicated ch_names
        and if fiducials are present they should share the same coordinate
        system and location values.
        """
        def is_fid_defined(fid):
            return not(
                fid.nasion is None and fid.lpa is None and fid.rpa is None
            )

        # Check for none duplicated ch_names
        ch_names_intersection = set(self.ch_names).intersection(other.ch_names)
        if ch_names_intersection:
            raise RuntimeError((
                "Cannot add two DigMontage objects if they contain duplicated"
                " channel names. Duplicated channel(s) found: {}."
            ).format(
                ', '.join(['%r' % v for v in sorted(ch_names_intersection)])
            ))

        # Check for unique matching fiducials
        self_fid, self_coord = _get_fid_coords(self.dig)
        other_fid, other_coord = _get_fid_coords(other.dig)

        if is_fid_defined(self_fid) and is_fid_defined(other_fid):
            if self_coord != other_coord:
                raise RuntimeError('Cannot add two DigMontage objects if '
                                   'fiducial locations are not in the same '
                                   'coordinate system.')

            for kk in self_fid:
                if not np.array_equal(self_fid[kk], other_fid[kk]):
                    raise RuntimeError('Cannot add two DigMontage objects if '
                                       'fiducial locations do not match '
                                       '(%s)' % kk)

            # keep self
            self.dig = _format_dig_points(
                self.dig + [d for d in other.dig
                            if d['kind'] != FIFF.FIFFV_POINT_CARDINAL]
            )
        else:
            self.dig = _format_dig_points(self.dig + other.dig)

        self.ch_names += other.ch_names
        return self

    def copy(self):
        """Copy the DigMontage object.

        Returns
        -------
        dig : instance of DigMontage
            The copied DigMontage instance.
        """
        return deepcopy(self)

    def __add__(self, other):
        """Add two DigMontages."""
        out = self.copy()
        out += other
        return out

    @property
    def dig_ch_pos(self):
        warn('"dig_ch_pos" attribute is deprecated and will be removed in '
             'v0.20', DeprecationWarning)
        return self._ch_pos()

    def _get_ch_pos(self):
        pos = [d['r'] for d in _get_dig_eeg(self.dig)]
        assert len(self.ch_names) == len(pos)
        return dict(zip(self.ch_names, pos))

    def _get_dig_names(self):
        NAMED_KIND = (FIFF.FIFFV_POINT_EEG,)
        is_eeg = np.array([d['kind'] in NAMED_KIND for d in self.dig])
        assert len(self.ch_names) == is_eeg.sum()
        dig_names = [None] * len(self.dig)
        for ch_name_idx, dig_idx in enumerate(np.where(is_eeg)[0]):
            dig_names[dig_idx] = self.ch_names[ch_name_idx]

        return dig_names

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


def _get_scaling(unit, scale):
    if unit not in scale:
        raise ValueError("Unit needs to be one of %s, not %r" %
                         (sorted(scale.keys()), unit))
    else:
        return scale[unit]


def _check_unit_and_get_scaling(unit, valid_scales):
    _check_option('unit', unit, list(valid_scales.keys()))
    return valid_scales[unit]


def transform_to_head(montage):
    """Transform a DigMontage object into head coordinate.

    It requires that the LPA, RPA and Nasion fiducial
    point are available. It requires that all fiducial
    points are in the same coordinate e.g. 'unknown'
    and it will convert all the point in this coordinate
    system to Neuromag head coordinate system.

    Parameters
    ----------
    montage : instance of DigMontage
        The montage.

    Returns
    -------
    montage : instance of DigMontage
        The montage after transforming the points to head
        coordinate system.
    """
    # Get fiducial points and their coord_frame
    fid_coords, coord_frame = _get_fid_coords(montage.dig)

    montage = deepcopy(montage)  # to avoid inplace modification

    if coord_frame != FIFF.FIFFV_COORD_HEAD:
        nasion, lpa, rpa = \
            fid_coords['nasion'], fid_coords['lpa'], fid_coords['rpa']
        native_head_t = get_ras_to_neuromag_trans(nasion, lpa, rpa)

        for d in montage.dig:
            if d['coord_frame'] == coord_frame:
                d['r'] = apply_trans(native_head_t, d['r'])
                d['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    montage._coord_frame = 'head'  # XXX : should desappear in 0.20
    return montage


def _read_dig_montage_deprecation_warning_helper(**kwargs):
    if kwargs.pop('fif') is not None:
        warn('Using "read_dig_montage" with "fif" not None'
             ' is deprecated and will be removed in v0.20'
             ' Please use read_dig_fif instead.', DeprecationWarning)
        return
    if kwargs.pop('egi') is not None:
        warn('Using "read_dig_montage" with "egi" not None'
             ' is deprecated and will be removed in v0.20'
             ' Please use read_dig_egi instead.', DeprecationWarning)
        return
    if kwargs.pop('bvct') is not None:
        warn('Using "read_dig_montage" with "bvct" not None'
             ' is deprecated and will be removed in v0.20.'
             ' Please use read_dig_captrack instead.', DeprecationWarning)
        return

    if [kk for kk in ('hsp', 'hpi', 'elp') if isinstance(kwargs[kk],
                                                         np.ndarray)]:
        warn('Passing "np.arrays" to "hsp", "hpi" or "elp" in'
             ' "read_dig_montage" is deprecated and will be removed in v0.20.'
             ' Please use "make_dig_montage" instead.', DeprecationWarning)
        return

    warn('Using "read_dig_montage" is deprecated and will be removed in '
         'v0.20. Use read_dig_polhemus_isotrak.', DeprecationWarning)


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

    _read_dig_montage_deprecation_warning_helper(
        hsp=hsp, hpi=hpi, elp=elp, fif=fif, egi=egi, bvct=bvct,
    )

    if fif is not None:
        _raise_transform_err = True if dev_head_t or not transform else False
        _all_data_kwargs_are_none = all(
            x is None for x in (hsp, hpi, elp, point_names, egi, bvct)
        )

        if _raise_transform_err:
            raise ValueError('transform must be True and dev_head_t must be'
                             ' False for FIF dig montage')
        if not _all_data_kwargs_are_none:
            raise ValueError('hsp, hpi, elp, point_names, egi must all be'
                             ' None if fif is not None')

        return read_dig_fif(fname=fif)

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
    montage = _make_dig_montage(
        **data, transform_to_head=transform,
        compute_dev_head_t=dev_head_t,
    )

    montage._point_names = point_names  # XXX: hack this should go!!
    return montage


def read_dig_fif(fname):
    r"""Read digitized points from a .fif file.

    Note that electrode names are not present in the .fif file so
    they are here defined with the convention from VectorView
    systems (EEG001, EEG002, etc.)

    Parameters
    ----------
    fname : path-like
        FIF file from which to read digitization locations.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    read_dig_egi
    read_dig_captrack
    read_dig_polhemus_isotrak
    read_dig_hpts
    make_dig_montage
    """
    _check_fname(fname, overwrite='read', must_exist=True)
    # Load the dig data
    f, tree = fiff_open(fname)[:2]
    with f as fid:
        dig = _read_dig_fif(fid, tree)

    ch_names = []
    for d in dig:
        if d['kind'] == FIFF.FIFFV_POINT_EEG:
            ch_names.append('EEG%03d' % d['ident'])

    montage = DigMontage(dig=dig, ch_names=ch_names)
    return montage


def read_dig_hpts(fname, unit='mm'):
    """Read historical .hpts mne-c files.

    Parameters
    ----------
    fname : str
        The filepath of .hpts file.
    unit : 'm' | 'cm' | 'mm'
        Unit of the positions. Defaults to 'mm'.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    read_dig_captrack
    read_dig_egi
    read_dig_fif
    read_dig_polhemus_isotrak
    make_dig_montage

    Notes
    -----
    The hpts format digitzer data file may contain comment lines starting
    with the pound sign (#) and data lines of the form::

         <*category*> <*identifier*> <*x/mm*> <*y/mm*> <*z/mm*>

    where:

    ``<*category*>``
        defines the type of points. Allowed categories are: `hpi`,
        `cardinal` (fiducial), `eeg`, and `extra` corresponding to
        head-position indicator coil locations, cardinal landmarks, EEG
        electrode locations, and additional head surface points,
        respectively.

    ``<*identifier*>``
        identifies the point. The identifiers are usually sequential
        numbers. For cardinal landmarks, 1 = left auricular point,
        2 = nasion, and 3 = right auricular point. For EEG electrodes,
        identifier = 0 signifies the reference electrode.

    ``<*x/mm*> , <*y/mm*> , <*z/mm*>``
        Location of the point, usually in the head coordinate system
        in millimeters. If your points are in [m] then unit parameter can
        be changed.

    For example::

        cardinal    nasion    -5.6729  -12.3873  -30.3671
        cardinal    lpa    -37.6782  -10.4957   91.5228
        cardinal    rpa    -131.3127    9.3976  -22.2363
        hpi    1    -30.4493  -11.8450   83.3601
        hpi    2    -122.5353    9.2232  -28.6828
        hpi    3    -6.8518  -47.0697  -37.0829
        hpi    4    7.3744  -50.6297  -12.1376
        hpi    5    -33.4264  -43.7352  -57.7756
        eeg    FP1  3.8676  -77.0439  -13.0212
        eeg    FP2  -31.9297  -70.6852  -57.4881
        eeg    F7  -6.1042  -68.2969   45.4939
        ...

    """
    from ._standard_montage_utils import _str_names, _str
    VALID_SCALES = dict(mm=1e-3, cm=1e-2, m=1)
    _scale = _check_unit_and_get_scaling(unit, VALID_SCALES)

    out = np.genfromtxt(fname, comments='#',
                        dtype=(_str, _str, 'f8', 'f8', 'f8'))
    kind, label = _str_names(out['f0']), _str_names(out['f1'])
    xyz = np.array([out['f%d' % ii] for ii in range(2, 5)]).T
    xyz *= _scale
    del _scale
    fid = {label[ii]: this_xyz
           for ii, this_xyz in enumerate(xyz) if kind[ii] == 'cardinal'}
    ch_pos = {label[ii]: this_xyz
              for ii, this_xyz in enumerate(xyz) if kind[ii] == 'eeg'}
    hpi = np.array([this_xyz for ii, this_xyz in enumerate(xyz)
                    if kind[ii] == 'hpi'])
    hpi.shape = (-1, 3)  # in case it's empty
    hsp = np.array([this_xyz for ii, this_xyz in enumerate(xyz)
                    if kind[ii] == 'extra'])
    hsp.shape = (-1, 3)  # in case it's empty
    return make_dig_montage(ch_pos=ch_pos, **fid, hpi=hpi, hsp=hsp)


def read_dig_egi(fname):
    """Read electrode locations from EGI system.

    Parameters
    ----------
    fname : path-like
        EGI MFF XML coordinates file from which to read digitization locations.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    read_dig_captrack
    read_dig_fif
    read_dig_hpts
    read_dig_polhemus_isotrak
    make_dig_montage
    """
    _check_fname(fname, overwrite='read', must_exist=True)

    data = _read_dig_montage_egi(
        fname=fname,
        _scaling=1.,
        _all_data_kwargs_are_none=True
    )

    # XXX: to change to the new naming in v.0.20 (all this block should go)
    data.pop('point_names')
    data['hpi'] = data.pop('elp')
    data['ch_pos'] = data.pop('dig_ch_pos')

    return make_dig_montage(**data)


def read_dig_captrack(fname):
    """Read electrode locations from CapTrak Brain Products system.

    Parameters
    ----------
    fname : path-like
        BrainVision CapTrak coordinates file from which to read EEG electrode
        locations. This is typically in XML format with the .bvct extension.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    read_dig_egi
    read_dig_fif
    read_dig_hpts
    read_dig_polhemus_isotrak
    make_dig_montage
    """
    _check_fname(fname, overwrite='read', must_exist=True)
    data = _parse_brainvision_dig_montage(fname)

    # XXX: to change to the new naming in v.0.20 (all this block should go)
    data.pop('point_names')
    data['hpi'] = data.pop('elp')
    data['ch_pos'] = data.pop('dig_ch_pos')

    return make_dig_montage(**data)


def _get_montage_in_head(montage):
    coords = set([d['coord_frame'] for d in montage.dig])
    if len(coords) == 1 and coords.pop() == FIFF.FIFFV_COORD_HEAD:
        return montage
    else:
        return transform_to_head(montage.copy())


def _set_montage_deprecation_helper(montage, update_ch_names, set_dig,
                                    raise_if_subset):
    """Manage deprecation policy for _set_montage.

    montage : instance of DigMontage | 'kind' | None
        The montage.
    update_ch_names : bool
        Whether to update or not ``ch_names`` in info.
    set_dig : bool
        Whether to copy or not ``montage.dig`` into ``info['dig']``
    raise_if_subset: bool
        Flag to grant raise/warn backward compatibility.

    Notes
    -----
    v0.19:
       - deprecate all montage types but DigMontage (or None, or valid 'kind')
       - deprecate using update_ch_names and set_dig
       - add raise_if_subset flag (defaults to False)

    v0.20:
       - montage is only DigMontage
       - update_ch_names and set_dig disappear
       - raise_if_subset defaults to True, still warns

    v0.21:
       - remove raise_if_subset
    """
    if isinstance(montage, (DigMontage, type(None))):
        # only worry about the DigMontage case
        if update_ch_names is not DEPRECATED_PARAM:
            warn((
                'Using ``update_ch_names`` to ``set_montage`` when using'
                ' DigMontage is deprecated and ``update_ch_names`` will be'
                ' removed in 0.20'
            ), DeprecationWarning)
        if set_dig is not DEPRECATED_PARAM:
            warn((
                'Using ``set_dig`` to ``set_montage`` when using'
                ' DigMontage is deprecated and ``set_dig`` will be'
                ' removed in 0.20'
            ), DeprecationWarning)

    elif isinstance(montage, str) and montage not in _BUILT_IN_MONTAGES:
        warn((
            'Using str in montage different from the built in templates '
            ' (i.e. a path) is deprecated. Please choose the proper reader to'
            ' load your montage using: '
            ' ``read_dig_fif``, ``read_dig_egi``, ``read_custom_montage``,'
            ' or ``read_dig_captrack``'
        ), DeprecationWarning)
    elif not (isinstance(montage, str) or montage is None):  # Montage
        warn((
            'Setting a montage using anything rather than DigMontage'
            ' is deprecated and will raise an error in v0.20.'
            ' Please use ``read_dig_fif``, ``read_dig_egi``,'
            ' ``read_dig_polhemus_isotrak``, or ``read_dig_captrack``'
            ' ``read_dig_hpts``, ``read_dig_captrack`` or'
            ' ``read_custom_montage`` to read a digitization based on'
            ' your needs instead; or ``make_standard_montage`` to create'
            ' ``DigMontage`` based on template; or ``make_dig_montage``'
            ' to create a ``DigMontage`` out of np.arrays.'
        ), DeprecationWarning)

    # This is unlikely to be trigger but it applies in all cases
    if raise_if_subset is not DEPRECATED_PARAM:
        # nothing to be done in 0.19
        pass  # noqa

    # Return defaults
    return (
        False if update_ch_names is DEPRECATED_PARAM else update_ch_names,
        True if set_dig is DEPRECATED_PARAM else set_dig,
        True if raise_if_subset is DEPRECATED_PARAM else raise_if_subset,
    )


def _set_montage(info, montage, update_ch_names=DEPRECATED_PARAM,
                 set_dig=DEPRECATED_PARAM, raise_if_subset=DEPRECATED_PARAM):
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

        Deprecated all types of montage but DigMontage in v0.19.
    update_ch_names : bool
        If True, overwrite the info channel names with the ones from montage.
        Defaults to False.

        Deprecated in v0.19 and will be removed in v0.20.
    raise_if_subset: bool
        If True, ValueError will be raised when montage.ch_names is a
        subset of info['ch_names']. This parameter was introduced for
        backward compatibility when set to False.

        Defaults to False in 0.19, it will change to default to True in
        0.20, and will be removed in 0.21.

        .. versionadded: 0.19
    Notes
    -----
    This function will change the info variable in place.
    """
    update_ch_names, set_dig, _raise = _set_montage_deprecation_helper(
        montage, update_ch_names, set_dig, raise_if_subset
    )

    if isinstance(montage, str):  # load builtin montage
        if montage in _BUILT_IN_MONTAGES:
            montage = make_standard_montage(montage)
        else:
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
        _mnt = _get_montage_in_head(montage)

        def _backcompat_value(pos, ref_pos):
            if any(np.isnan(pos)):
                return np.full(6, np.nan)
            else:
                return np.concatenate((pos, ref_pos))

        ch_pos = _mnt._get_ch_pos()
        refs = set(ch_pos.keys()) & {'EEG000', 'REF'}
        assert len(refs) <= 1
        eeg_ref_pos = np.zeros(3) if not(refs) else ch_pos.pop(refs.pop())

        # This raises based on info being subset/superset of montage
        _pick_chs = partial(
            pick_types, exclude=[], eeg=True, seeg=True, ecog=True, meg=False,
        )
        matched_ch_names = _check_ch_names_are_compatible(
            info_names=set([info['ch_names'][ii] for ii in _pick_chs(info)]),
            montage_names=set(ch_pos),
            raise_if_subset=_raise,  # XXX: deprecated param to remove in 0.20
        )

        for name in matched_ch_names:
            _loc_view = info['chs'][info['ch_names'].index(name)]['loc']
            _loc_view[:6] = _backcompat_value(ch_pos[name], eeg_ref_pos)

        if set_dig:  # XXX: in 0.20 it is safe to move the code out of the if.
            _names = _mnt._get_dig_names()
            info['dig'] = _format_dig_points([
                _mnt.dig[ii] for ii, name in enumerate(_names)
                if name in matched_ch_names.union({None, 'EEG000', 'REF'})
            ])

        if _mnt.dev_head_t is not None:
            info['dev_head_t'] = Transform('meg', 'head', _mnt.dev_head_t)

    elif montage is None:
        for ch in info['chs']:
            ch['loc'] = np.full(12, np.nan)
        if set_dig:
            info['dig'] = None
    else:
        raise TypeError("Montage must be a 'Montage', 'DigMontage', 'str' or "
                        "'None' instead of '%s'." % type(montage))


def _read_isotrak_elp_points(fname):
    """Read Polhemus Isotrak digitizer data from a ``.elp`` file.

    Parameters
    ----------
    fname : str
        The filepath of .elp Polhemus Isotrak file.

    Returns
    -------
    out : dict of arrays
        The dictionary containing locations for 'nasion', 'lpa', 'rpa'
        and 'points'.
    """
    value_pattern = r"\-?\d+\.?\d*e?\-?\d*"
    coord_pattern = r"({0})\s+({0})\s+({0})\s*$".format(value_pattern)

    with open(fname) as fid:
        file_str = fid.read()

    points_str = [m.groups() for m in re.finditer(coord_pattern, file_str,
                                                  re.MULTILINE)]
    points = np.array(points_str, dtype=float)

    return {
        'nasion': points[0], 'lpa': points[1], 'rpa': points[2],
        'points': points[3:]
    }


def _read_isotrak_hsp_points(fname):
    """Read Polhemus Isotrak digitizer data from a ``.hsp`` file.

    Parameters
    ----------
    fname : str
        The filepath of .hsp Polhemus Isotrak file.

    Returns
    -------
    out : dict of arrays
        The dictionary containing locations for 'nasion', 'lpa', 'rpa'
        and 'points'.
    """
    def consume(fid, predicate):  # just a consumer to move around conveniently
        while(predicate(fid.readline())):
            pass

    def get_hsp_fiducial(line):
        return np.fromstring(line.replace('%F', ''), dtype=float, sep='\t')

    with open(fname) as ff:
        consume(ff, lambda l: 'position of fiducials' not in l.lower())

        nasion = get_hsp_fiducial(ff.readline())
        lpa = get_hsp_fiducial(ff.readline())
        rpa = get_hsp_fiducial(ff.readline())

        _ = ff.readline()
        n_points, n_cols = np.fromstring(ff.readline(), dtype=int, sep='\t')
        points = np.fromstring(
            string=ff.read(), dtype=float, sep='\t',
        ).reshape(-1, n_cols)
        assert points.shape[0] == n_points

    return {
        'nasion': nasion, 'lpa': lpa, 'rpa': rpa, 'points': points
    }


def read_dig_polhemus_isotrak(fname, ch_names=None, unit='m'):
    """Read Polhemus digitizer data from a file.

    Parameters
    ----------
    fname : str
        The filepath of Polhemus ISOTrak formatted file.
        File extension is expected to be '.hsp', '.elp' or '.eeg'.
    ch_names : None | list of str
        The names of the points. This will make the points
        considered as EEG channels.
    unit : 'm' | 'cm' | 'mm'
        Unit of the digitizer file. Polhemus ISOTrak systems data is usually
        exported in meters. Defaults to 'm'

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    make_dig_montage
    read_polhemus_fastscan
    read_dig_captrack
    read_dig_egi
    read_dig_fif
    """
    VALID_FILE_EXT = ('.hsp', '.elp', '.eeg')
    VALID_SCALES = dict(mm=1e-3, cm=1e-2, m=1)
    _scale = _check_unit_and_get_scaling(unit, VALID_SCALES)

    _, ext = op.splitext(fname)
    _check_option('fname', ext, VALID_FILE_EXT)

    if ext == '.elp':
        data = _read_isotrak_elp_points(fname)
    else:
        # Default case we read points as hsp since is the most likely scenario
        data = _read_isotrak_hsp_points(fname)

    if _scale != 1:
        data = {key: val * _scale for key, val in data.items()}
    else:
        pass  # noqa

    if ch_names is None:
        keyword = 'hpi' if ext == '.elp' else 'hsp'
        data[keyword] = data.pop('points')

    else:
        points = data.pop('points')
        if points.shape[0] == len(ch_names):
            data['ch_pos'] = dict(zip(ch_names, points))
        else:
            raise ValueError((
                "Length of ``ch_names`` does not match the number of points"
                " in {fname}. Expected ``ch_names`` length {n_points:d},"
                " given {n_chnames:d}"
            ).format(
                fname=fname, n_points=points.shape[0], n_chnames=len(ch_names)
            ))

    return make_dig_montage(**data)


def _get_polhemus_fastscan_header(fname):
    with open(fname, 'r') as fid:
        header = [l for l in takewhile(lambda line: line.startswith('%'), fid)]

    return ''.join(header)


def read_polhemus_fastscan(fname, unit='mm'):
    """Read Polhemus FastSCAN digitizer data from a ``.txt`` file.

    Parameters
    ----------
    fname : str
        The filepath of .txt Polhemus FastSCAN file.
    unit : 'm' | 'cm' | 'mm'
        Unit of the digitizer file. Polhemus FastSCAN systems data is usually
        exported in millimeters. Defaults to 'mm'

    Returns
    -------
    points : array, shape (n_points, 3)
        The digitization points in digitizer coordinates.

    See Also
    --------
    read_dig_polhemus_isotrak
    make_dig_montage
    """
    VALID_FILE_EXT = ['.txt']
    VALID_SCALES = dict(mm=1e-3, cm=1e-2, m=1)
    _scale = _check_unit_and_get_scaling(unit, VALID_SCALES)

    _, ext = op.splitext(fname)
    _check_option('fname', ext, VALID_FILE_EXT)

    if _get_polhemus_fastscan_header(fname).find('FastSCAN') == -1:
        raise ValueError(
            "%s does not contain Polhemus FastSCAN header" % fname
        )

    points = _scale * np.loadtxt(fname, comments='%', ndmin=2)

    return points


def _read_eeglab_locations(fname, unit):
    ch_names = np.genfromtxt(fname, dtype=str, usecols=3).tolist()
    topo = np.loadtxt(fname, dtype=float, usecols=[1, 2])
    sph = _topo_to_sph(topo)
    pos = _sph_to_cart(sph)
    pos[:, [0, 1]] = pos[:, [1, 0]] * [-1, 1]

    return ch_names, pos


def read_custom_montage(fname, head_size=HEAD_SIZE_DEFAULT, unit='m'):
    """Read a montage from a file.

    Parameters
    ----------
    fname : str
        File extension is expected to be:
        '.loc' or '.locs' or '.eloc' (for EEGLAB files),
        '.sfp' (BESA/EGI files), '.csd',
        ‘.elc’, ‘.txt’, ‘.csd’, ‘.elp’ (BESA spherical),
        .bvef (BrainVision files).

    head_size : float | None
        The size of the head in [m]. If none, returns the values read from the
        file with no modification. Defaults to 95mm.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    Notes
    -----
    The function is a helper to read electrode positions you may have
    in various formats. Most of these format are weakly specified
    in terms of units, coordinate systems. It implies that setting
    a montage using a DigMontage produced by this function may
    be problematic. If you use a standard/template (eg. 10/20,
    10/10 or 10/05) we recommend you use :func:`make_standard_montage`.
    If you can have positions in memory you can also use
    :func:`make_dig_montage` that takes arrays as input.

    See Also
    --------
    make_dig_montage
    make_standard_montage
    """
    from ._standard_montage_utils import (
        _read_theta_phi_in_degrees, _read_sfp, _read_csd, _read_elc,
        _read_elp_besa, _read_brainvision
    )
    SUPPORTED_FILE_EXT = {
        'eeglab': ('.loc', '.locs', '.eloc', ),
        'hydrocel': ('.sfp', ),
        'matlab': ('.csd', ),
        'asa electrode': ('.elc', ),
        'generic (Theta-phi in degrees)': ('.txt', ),
        'standard BESA spherical': ('.elp', ),  # XXX: not same as polhemus elp
        'brainvision': ('.bvef', ),
    }

    _, ext = op.splitext(fname)
    _check_option('fname', ext, list(chain(*SUPPORTED_FILE_EXT.values())))

    if ext in SUPPORTED_FILE_EXT['eeglab']:
        if head_size is None:
            raise(ValueError,
                  "``head_size`` cannot be None for '{}'".format(ext))
        ch_names, pos = _read_eeglab_locations(fname, unit)
        scale = head_size / np.median(np.linalg.norm(pos, axis=-1))
        pos *= scale

        montage = make_dig_montage(
            ch_pos=dict(zip(ch_names, pos)),
            coord_frame='head',
        )

    elif ext in SUPPORTED_FILE_EXT['hydrocel']:
        montage = _read_sfp(fname, head_size=head_size)

    elif ext in SUPPORTED_FILE_EXT['matlab']:
        montage = _read_csd(fname, head_size=head_size)

    elif ext in SUPPORTED_FILE_EXT['asa electrode']:
        montage = _read_elc(fname, head_size=head_size)

    elif ext in SUPPORTED_FILE_EXT['generic (Theta-phi in degrees)']:
        if head_size is None:
            raise(ValueError,
                  "``head_size`` cannot be None for '{}'".format(ext))
        montage = _read_theta_phi_in_degrees(fname, head_size=head_size,
                                             fid_names=('Nz', 'LPA', 'RPA'))

    elif ext in SUPPORTED_FILE_EXT['standard BESA spherical']:
        montage = _read_elp_besa(fname, head_size)

    elif ext in SUPPORTED_FILE_EXT['brainvision']:
        montage = _read_brainvision(fname, head_size, unit)

    return montage


def compute_dev_head_t(montage):
    """Compute device to head transform from a DigMontage.

    Parameters
    ----------
    montage : instance of DigMontage
        The DigMontage must contain the fiducials in head
        coordinate system and hpi points in both head and
        meg device coordinate system.

    Returns
    -------
    dev_head_t : instance of Transform
        A Device-to-Head transformation matrix.
    """
    from ..coreg import fit_matched_points

    _, coord_frame = _get_fid_coords(montage.dig)
    if coord_frame != FIFF.FIFFV_COORD_HEAD:
        raise ValueError('montage should have been set to head coordinate '
                         'system with transform_to_head function.')

    hpi_head = [d['r'] for d in montage.dig
                if (d['kind'] == FIFF.FIFFV_POINT_HPI and
                    d['coord_frame'] == FIFF.FIFFV_COORD_HEAD)]
    hpi_dev = [d['r'] for d in montage.dig
               if (d['kind'] == FIFF.FIFFV_POINT_HPI and
                   d['coord_frame'] == FIFF.FIFFV_COORD_DEVICE)]

    if not (len(hpi_head) == len(hpi_dev) and len(hpi_dev) > 0):
        raise ValueError((
            "To compute Device-to-Head transformation, the same number of HPI"
            " points in device and head coordinates is required. (Got {dev}"
            " points in device and {head} points in head coordinate systems)"
        ).format(dev=len(hpi_dev), head=len(hpi_head)))

    trans = fit_matched_points(tgt_pts=hpi_head, src_pts=hpi_dev, out='trans')
    return Transform(fro='meg', to='head', trans=trans)


def make_standard_montage(kind, head_size=HEAD_SIZE_DEFAULT):
    """Read a generic (built-in) montage.

    Parameters
    ----------
    kind : str
        The name of the montage to use. See notes for valid kinds.
    head_size : float
        The head size (in meters) to use for spherical montages.
        Defaults to 95mm.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    DigMontage
    make_dig_montage
    read_custom_montage

    Notes
    -----
    Individualized (digitized) electrode positions should be read in using
    :func:`read_dig_captrack`, :func:`read_dig_egi`, :func:`read_dig_fif`,
    :func:`read_dig_polhemus_isotrak`, :func:`read_dig_hpts` or made with
    :func:`make_dig_montage`.

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

    .. versionadded:: 0.19.0
    """
    from ._standard_montage_utils import standard_montage_look_up_table
    if kind not in standard_montage_look_up_table:
        raise ValueError('Could not find the montage %s. Please provide one '
                         'among: %s' % (kind,
                                        standard_montage_look_up_table.keys()))
    return standard_montage_look_up_table[kind](head_size=head_size)
