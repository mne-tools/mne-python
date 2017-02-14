# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: Simplified BSD

from collections import Iterable
import os
import os.path as op

import numpy as np
import xml.etree.ElementTree as ElementTree

from ..viz import plot_montage
from .channels import _contains_ch_type
from ..transforms import (apply_trans, get_ras_to_neuromag_trans, _sph_to_cart,
                          _topo_to_sph, _str_to_frame, _frame_to_str)
from ..io.meas_info import (_make_dig_points, _read_dig_points, _read_dig_fif,
                            write_dig)
from ..io.pick import pick_types
from ..io.open import fiff_open
from ..io.constants import FIFF
from ..utils import _check_fname, warn, copy_function_doc_to_method_doc

from ..externals.six import string_types
from ..externals.six.moves import map


class Montage(object):
    """Montage for standard EEG electrode locations.

    .. warning:: Montages should typically be loaded from a file using
                 :func:`mne.channels.read_montage` instead of
                 instantiating this class directly.

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

    See Also
    --------
    DigMontage
    read_montage
    read_dig_montage

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    def __init__(self, pos, ch_names, kind, selection):  # noqa: D102
        self.pos = pos
        self.ch_names = ch_names
        self.kind = kind
        self.selection = selection

    def __repr__(self):
        """String representation."""
        s = ('<Montage | %s - %d channels: %s ...>'
             % (self.kind, len(self.ch_names), ', '.join(self.ch_names[:3])))
        return s

    @copy_function_doc_to_method_doc(plot_montage)
    def plot(self, scale_factor=20, show_names=False, show=True):
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names, show=True)


def read_montage(kind, ch_names=None, path=None, unit='m', transform=False):
    """Read a generic (built-in) montage.

    Individualized (digitized) electrode positions should be
    read in using :func:`read_dig_montage`.

    In most cases, you should only need the `kind` parameter to load one of
    the built-in montages (see Notes).

    Parameters
    ----------
    kind : str
        The name of the montage file without the file extension (e.g.
        kind='easycap-M10' for 'easycap-M10.txt'). Files with extensions
        '.elc', '.txt', '.csd', '.elp', '.hpts', '.sfp' or '.loc' ('.locs' and
        '.eloc') are supported.
    ch_names : list of str | None
        If not all electrodes defined in the montage are present in the EEG
        data, use this parameter to select subset of electrode positions to
        load. If None (default), all defined electrode positions are returned.

        .. note:: ``ch_names`` are compared to channel names in the montage
                  file after converting them both to upper case. If a match is
                  found, the letter case in the original ``ch_names`` is used
                  in the returned montage.

    path : str | None
        The path of the folder containing the montage file. Defaults to the
        mne/channels/data/montages folder in your mne-python installation.
    unit : 'm' | 'cm' | 'mm'
        Unit of the input file. If not 'm' (default), coordinates will be
        rescaled to 'm'.
    transform : bool
        If True, points will be transformed to Neuromag space.
        The fidicuals, 'nasion', 'lpa', 'rpa' must be specified in
        the montage file. Useful for points captured using Polhemus FastSCAN.
        Default is False.

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

    Montages can contain fiducial points in addition to electrode
    locations, e.g. ``biosemi64`` contains 67 total channels.

    The valid ``kind`` arguments are:

    ===================   =====================================================
    Kind                  description
    ===================   =====================================================
    standard_1005         Electrodes are named and positioned according to the
                          international 10-05 system.
    standard_1020         Electrodes are named and positioned according to the
                          international 10-20 system.
    standard_alphabetic   Electrodes are named with LETTER-NUMBER combinations
                          (A1, B2, F4, etc.)
    standard_postfixed    Electrodes are named according to the international
                          10-20 system using postfixes for intermediate
                          positions.
    standard_prefixed     Electrodes are named according to the international
                          10-20 system using prefixes for intermediate
                          positions.
    standard_primed       Electrodes are named according to the international
                          10-20 system using prime marks (' and '') for
                          intermediate positions.

    biosemi16             BioSemi cap with 16 electrodes
    biosemi32             BioSemi cap with 32 electrodes
    biosemi64             BioSemi cap with 64 electrodes
    biosemi128            BioSemi cap with 128 electrodes
    biosemi160            BioSemi cap with 160 electrodes
    biosemi256            BioSemi cap with 256 electrodes

    easycap-M10           Brainproducts EasyCap with electrodes named
                          according to the 10-05 system
    easycap-M1            Brainproduct EasyCap with numbered electrodes

    EGI_256               Geodesic Sensor Net with 256 channels

    GSN-HydroCel-32       HydroCel Geodesic Sensor Net with 32 electrodes
    GSN-HydroCel-64_1.0   HydroCel Geodesic Sensor Net with 64 electrodes
    GSN-HydroCel-65_1.0   HydroCel Geodesic Sensor Net with 64 electrodes + Cz
    GSN-HydroCel-128      HydroCel Geodesic Sensor Net with 128 electrodes
    GSN-HydroCel-129      HydroCel Geodesic Sensor Net with 128 electrodes + Cz
    GSN-HydroCel-256      HydroCel Geodesic Sensor Net with 256 electrodes
    GSN-HydroCel-257      HydroCel Geodesic Sensor Net with 256 electrodes + Cz
    ===================   =====================================================

    .. versionadded:: 0.9.0
    """
    if path is None:
        path = op.join(op.dirname(__file__), 'data', 'montages')
    if not op.isabs(kind):
        supported = ('.elc', '.txt', '.csd', '.sfp', '.elp', '.hpts', '.loc',
                     '.locs', '.eloc')
        montages = [op.splitext(f) for f in os.listdir(path)]
        montages = [m for m in montages if m[1] in supported and kind == m[0]]
        if len(montages) != 1:
            raise ValueError('Could not find the montage. Please provide the '
                             'full path.')
        kind, ext = montages[0]
    else:
        kind, ext = op.splitext(kind)
    fname = op.join(path, kind + ext)

    if ext == '.sfp':
        # EGI geodesic
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
                if not line or not set(line) - set([' ']):
                    break
                ch_names_.append(line.strip(' ').strip('\n'))
        pos = np.array(pos) * scale_factor
    elif ext == '.txt':
        # easycap
        try:  # newer version
            data = np.genfromtxt(fname, dtype='str', skip_header=1)
        except TypeError:
            data = np.genfromtxt(fname, dtype='str', skiprows=1)
        ch_names_ = list(data[:, 0])
        az = np.deg2rad(data[:, 2].astype(float))
        pol = np.deg2rad(data[:, 1].astype(float))
        pos = _sph_to_cart(np.array([np.ones(len(az)) * 85., az, pol]).T)
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
        az = np.deg2rad(table['theta'])
        pol = np.deg2rad(90. - table['phi'])
        pos = _sph_to_cart(np.array([np.ones(len(az)), az, pol]).T)
    elif ext == '.elp':
        # standard BESA spherical
        dtype = np.dtype('S8, S8, f8, f8, f8')
        try:
            data = np.loadtxt(fname, dtype=dtype, skip_header=1)
        except TypeError:
            data = np.loadtxt(fname, dtype=dtype, skiprows=1)

        ch_names_ = data['f1'].astype(np.str)
        az = data['f2']
        horiz = data['f3']
        radius = np.abs(az / 180.)
        az = np.deg2rad(np.array([h if a >= 0. else 180 + h
                                  for h, a in zip(horiz, az)]))
        pol = radius * np.pi
        pos = _sph_to_cart(np.array([np.ones(len(az)) * 85., az, pol]).T)
    elif ext == '.hpts':
        # MNE-C specified format for generic digitizer data
        dtype = [('type', 'S8'), ('name', 'S8'),
                 ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        data = np.loadtxt(fname, dtype=dtype)
        ch_names_ = data['name'].astype(np.str)
        pos = np.vstack((data['x'], data['y'], data['z'])).T
    elif ext in ('.loc', '.locs', '.eloc'):
        ch_names_ = np.loadtxt(fname, dtype='S4',
                               usecols=[3]).astype(np.str).tolist()
        dtype = {'names': ('angle', 'radius'), 'formats': ('f4', 'f4')}
        topo = np.loadtxt(fname, dtype=float, usecols=[1, 2])
        sph = _topo_to_sph(topo)
        pos = _sph_to_cart(sph)
        pos[:, [0, 1]] = pos[:, [1, 0]] * [-1, 1]
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
        # Ensure channels with differing case are found.
        upper_names = [ch_name.upper() for ch_name in ch_names]
        sel, ch_names_ = zip(*[(i, ch_names[upper_names.index(e)]) for i, e in
                               enumerate([n.upper() for n in ch_names_])
                               if e in upper_names])
        sel = list(sel)
        pos = pos[sel]
        selection = selection[sel]
    else:
        ch_names_ = list(ch_names_)
    kind = op.split(kind)[-1]
    return Montage(pos=pos, ch_names=ch_names_, kind=kind, selection=selection)


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
    hpi : array, shape (n_hpi, 3)
        The positions of the head-position indicator coils in 3d.
        These points are in the MEG device space.
    elp : array, shape (n_hpi, 3)
        The positions of the head-position indicator coils in 3d.
        This is typically in the native digitizer space.
    point_names : list, shape (n_elp)
        The names of the digitized points for hpi and elp.
    nasion : array, shape (1, 3)
        The position of the nasion fidicual point.
    lpa : array, shape (1, 3)
        The position of the left periauricular fidicual point.
    rpa : array, shape (1, 3)
        The position of the right periauricular fidicual point.
    dev_head_t : array, shape (4, 4)
        A Device-to-Head transformation matrix.
    dig_ch_pos : dict
        Dictionary of channel positions.

        .. versionadded:: 0.12

    coord_frame : str
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space.

    See Also
    --------
    Montage
    read_dig_montage
    read_montage

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    def __init__(self, hsp=None, hpi=None, elp=None, point_names=None,
                 nasion=None, lpa=None, rpa=None, dev_head_t=None,
                 dig_ch_pos=None, coord_frame='unknown'):  # noqa: D102
        self.hsp = hsp
        self.hpi = hpi
        if elp is not None:
            if not isinstance(point_names, Iterable):
                raise TypeError('If elp is specified, point_names must '
                                'provide a list of str with one entry per ELP '
                                'point')
            point_names = list(point_names)
            if len(point_names) != len(elp):
                raise ValueError('elp contains %i points but %i '
                                 'point_names were specified.' %
                                 (len(elp), len(point_names)))
        self.elp = elp
        self.point_names = point_names

        self.nasion = nasion
        self.lpa = lpa
        self.rpa = rpa
        self.dev_head_t = dev_head_t
        self.dig_ch_pos = dig_ch_pos
        if not isinstance(coord_frame, string_types) or \
                coord_frame not in _str_to_frame:
            raise ValueError('coord_frame must be one of %s, got %s'
                             % (sorted(_str_to_frame.keys()), coord_frame))
        self.coord_frame = coord_frame

    def __repr__(self):
        """String representation."""
        s = ('<DigMontage | %d extras (headshape), %d HPIs, %d fiducials, %d '
             'channels>' %
             (len(self.hsp) if self.hsp is not None else 0,
              len(self.point_names) if self.point_names is not None else 0,
              sum(x is not None for x in (self.lpa, self.rpa, self.nasion)),
              len(self.dig_ch_pos) if self.dig_ch_pos is not None else 0,))
        return s

    @copy_function_doc_to_method_doc(plot_montage)
    def plot(self, scale_factor=20, show_names=False, show=True):
        return plot_montage(self, scale_factor=scale_factor,
                            show_names=show_names)

    def transform_to_head(self):
        """Transform digitizer points to Neuromag head coordinates."""
        if self.coord_frame == 'head':  # nothing to do
            return
        nasion, rpa, lpa = self.nasion, self.rpa, self.lpa
        if any(x is None for x in (nasion, rpa, lpa)):
            if self.elp is None or self.point_names is None:
                raise ValueError('ELP points and names must be specified for '
                                 'transformation.')
            names = [name.lower() for name in self.point_names]

            # check that all needed points are present
            kinds = ('nasion', 'lpa', 'rpa')
            missing = [name for name in kinds if name not in names]
            if len(missing) > 0:
                raise ValueError('The points %s are missing, but are needed '
                                 'to transform the points to the MNE '
                                 'coordinate system. Either add the points, '
                                 'or read the montage with transform=False.'
                                 % str(missing))

            nasion, lpa, rpa = [self.elp[names.index(kind)] for kind in kinds]

            # remove fiducials from elp
            mask = np.ones(len(names), dtype=bool)
            for fid in ['nasion', 'lpa', 'rpa']:
                mask[names.index(fid)] = False
            self.elp = self.elp[mask]
            self.point_names = [p for pi, p in enumerate(self.point_names)
                                if mask[pi]]

        native_head_t = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        self.nasion, self.lpa, self.rpa = apply_trans(
            native_head_t, np.array([nasion, lpa, rpa]))
        if self.elp is not None:
            self.elp = apply_trans(native_head_t, self.elp)
        if self.hsp is not None:
            self.hsp = apply_trans(native_head_t, self.hsp)
        if self.dig_ch_pos is not None:
            for key, val in self.dig_ch_pos.items():
                self.dig_ch_pos[key] = apply_trans(native_head_t, val)
        self.coord_frame = 'head'

    def compute_dev_head_t(self):
        """Compute the Neuromag dev_head_t from matched points."""
        from ..coreg import fit_matched_points
        if self.elp is None or self.hpi is None:
            raise RuntimeError('must have both elp and hpi to compute the '
                               'device to head transform')
        self.dev_head_t = fit_matched_points(tgt_pts=self.elp,
                                             src_pts=self.hpi, out='trans')

    def _get_dig(self):
        """Get the digitization list."""
        return _make_dig_points(
            nasion=self.nasion, lpa=self.lpa, rpa=self.rpa, hpi=self.elp,
            extra_points=self.hsp, dig_ch_pos=self.dig_ch_pos)

    def save(self, fname):
        """Save digitization points to FIF.

        Parameters
        ----------
        fname : str
            The filename to use. Should end in .fif or .fif.gz.
        """
        if self.coord_frame != 'head':
            raise RuntimeError('Can only write out digitization points in '
                               'head coordinates.')
        write_dig(fname, self._get_dig())


_cardinal_ident_mapping = {
    FIFF.FIFFV_POINT_NASION: 'nasion',
    FIFF.FIFFV_POINT_LPA: 'lpa',
    FIFF.FIFFV_POINT_RPA: 'rpa',
}


def _check_frame(d, frame_str):
    """Helper to check coordinate frames."""
    if d['coord_frame'] != _str_to_frame[frame_str]:
        raise RuntimeError('dig point must be in %s coordinate frame, got %s'
                           % (frame_str, _frame_to_str[d['coord_frame']]))


def read_dig_montage(hsp=None, hpi=None, elp=None, point_names=None,
                     unit='auto', fif=None, egi=None, transform=True,
                     dev_head_t=False):
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

    transform : bool
        If True (default), points will be transformed to Neuromag space
        using :meth:`DigMontage.transform_to_head`.
        The fidicuals (nasion, lpa, and rpa) must be specified.
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
    if fif is not None:
        # Use a different code path
        if dev_head_t or not transform:
            raise ValueError('transform must be True and dev_head_t must be '
                             'False for FIF dig montage')
        if not all(x is None for x in (hsp, hpi, elp, point_names, egi)):
            raise ValueError('hsp, hpi, elp, point_names, egi must all be '
                             'None if fif is not None')
        _check_fname(fif, overwrite=True, must_exist=True)
        # Load the dig data
        f, tree = fiff_open(fif)[:2]
        with f as fid:
            dig = _read_dig_fif(fid, tree)
        # Split up the dig points by category
        hsp = list()
        hpi = list()
        elp = list()
        point_names = list()
        fids = dict()
        dig_ch_pos = dict()
        for d in dig:
            if d['kind'] == FIFF.FIFFV_POINT_CARDINAL:
                _check_frame(d, 'head')
                fids[_cardinal_ident_mapping[d['ident']]] = d['r']
            elif d['kind'] == FIFF.FIFFV_POINT_HPI:
                _check_frame(d, 'head')
                hpi.append(d['r'])
                elp.append(d['r'])
                point_names.append('HPI%03d' % d['ident'])
            elif d['kind'] == FIFF.FIFFV_POINT_EXTRA:
                _check_frame(d, 'head')
                hsp.append(d['r'])
            elif d['kind'] == FIFF.FIFFV_POINT_EEG:
                _check_frame(d, 'head')
                dig_ch_pos['EEG%03d' % d['ident']] = d['r']
        fids = [fids.get(key) for key in ('nasion', 'lpa', 'rpa')]
        hsp = np.array(hsp) if len(hsp) else None
        elp = np.array(elp) if len(elp) else None
        coord_frame = 'head'
    elif egi is not None:
        if not all(x is None for x in (hsp, hpi, elp, point_names, fif)):
            raise ValueError('hsp, hpi, elp, point_names, fif must all be '
                             'None if egi is not None')
        _check_fname(egi, overwrite=True, must_exist=True)

        root = ElementTree.parse(egi).getroot()
        ns = root.tag[root.tag.index('{'):root.tag.index('}') + 1]
        sensors = root.find('%ssensorLayout/%ssensors' % (ns, ns))
        fids = dict()
        dig_ch_pos = dict()

        fid_name_map = {'Nasion': 'nasion',
                        'Right periauricular point': 'rpa',
                        'Left periauricular point': 'lpa'}

        for s in sensors:
            name, number, kind = s[0].text, int(s[1].text), int(s[2].text)
            coordinates = np.array([float(s[3].text), float(s[4].text),
                                    float(s[5].text)])
            # EEG Channels
            if kind == 0:
                dig_ch_pos['EEG %03d' % number] = coordinates
            # Reference
            elif kind == 1:
                dig_ch_pos['EEG %03d' %
                           (len(dig_ch_pos.keys()) + 1)] = coordinates
            # Fiducials
            elif kind == 2:
                fid_name = fid_name_map[name]
                fids[fid_name] = coordinates
            # Unknown
            else:
                warn('Unknown sensor type %s detected. Skipping sensor...'
                     'Proceed with caution!' % kind)

        fids = [fids[key] for key in ('nasion', 'lpa', 'rpa')]
        coord_frame = 'unknown'

    else:
        fids = [None] * 3
        dig_ch_pos = None
        scale = {'mm': 1e-3, 'cm': 1e-2, 'auto': 1e-3, 'm': None}
        if unit not in scale:
            raise ValueError("Unit needs to be one of %s, not %r" %
                             (tuple(map(repr, scale)), unit))

        # HSP
        if isinstance(hsp, string_types):
            hsp = _read_dig_points(hsp, unit=unit)
        elif hsp is not None and scale[unit]:
            hsp *= scale[unit]

        # HPI
        if isinstance(hpi, string_types):
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
        if isinstance(elp, string_types):
            elp = _read_dig_points(elp, unit=unit)
        elif elp is not None and scale[unit]:
            elp *= scale[unit]
        coord_frame = 'unknown'

    # Transform digitizer coordinates to neuromag space
    out = DigMontage(hsp, hpi, elp, point_names, fids[0], fids[1], fids[2],
                     dig_ch_pos=dig_ch_pos, coord_frame=coord_frame)
    if fif is None and transform:  # only need to do this for non-Neuromag
        out.transform_to_head()
    if dev_head_t:
        out.compute_dev_head_t()
    return out


def _set_montage(info, montage, update_ch_names=False):
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
    montage : instance of Montage | instance of DigMontage
        The montage to apply.
    update_ch_names : bool
        If True, overwrite the info channel names with the ones from montage.
        Defaults to False.

    Notes
    -----
    This function will change the info variable in place.
    """
    if isinstance(montage, Montage):
        if update_ch_names:
            info['chs'] = list()
            for ii, ch_name in enumerate(montage.ch_names):
                ch_info = {'cal': 1., 'logno': ii + 1, 'scanno': ii + 1,
                           'range': 1.0, 'unit_mul': 0, 'ch_name': ch_name,
                           'unit': FIFF.FIFF_UNIT_V, 'kind': FIFF.FIFFV_EEG_CH,
                           'coord_frame': FIFF.FIFFV_COORD_HEAD,
                           'coil_type': FIFF.FIFFV_COIL_EEG}
                info['chs'].append(ch_info)
            info._update_redundant()

        if not _contains_ch_type(info, 'eeg'):
            raise ValueError('No EEG channels found.')

        sensors_found = []

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

        for pos, ch_name in zip(montage.pos, montage_ch_names):
            if ch_name not in info_ch_names:
                continue

            ch_idx = info_ch_names.index(ch_name)
            info['chs'][ch_idx]['loc'] = np.r_[pos, [0.] * 9]
            sensors_found.append(ch_idx)

        if len(sensors_found) == 0:
            raise ValueError('None of the sensors defined in the montage were '
                             'found in the info structure. Check the channel '
                             'names.')

        eeg_sensors = pick_types(info, meg=False, ref_meg=False, eeg=True,
                                 exclude=[])
        not_found = np.setdiff1d(eeg_sensors, sensors_found)
        if len(not_found) > 0:
            not_found_names = [info['ch_names'][ch] for ch in not_found]
            warn('The following EEG sensors did not have a position '
                 'specified in the selected montage: ' +
                 str(not_found_names) + '. Their position has been '
                 'left untouched.')

    elif isinstance(montage, DigMontage):
        info['dig'] = montage._get_dig()

        if montage.dev_head_t is not None:
            info['dev_head_t']['trans'] = montage.dev_head_t

        if montage.dig_ch_pos is not None:  # update channel positions, too
            eeg_ref_pos = montage.dig_ch_pos.get('EEG000', np.zeros(3))
            did_set = np.zeros(len(info['ch_names']), bool)
            is_eeg = np.zeros(len(info['ch_names']), bool)
            is_eeg[pick_types(info, meg=False, eeg=True, exclude=())] = True

            for ch_name, ch_pos in montage.dig_ch_pos.items():
                if ch_name == 'EEG000':
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
    else:
        raise TypeError("Montage must be a 'Montage' or 'DigMontage' "
                        "instead of '%s'." % type(montage))
