# Authors: Joan Massich <mailsik@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause
from collections import OrderedDict
import csv

import os.path as op
import numpy as np

from functools import partial
import xml.etree.ElementTree as ElementTree

from .montage import make_dig_montage
from .._freesurfer import get_mni_fiducials
from ..transforms import _sph_to_cart
from ..utils import warn, _pl
from . import __file__ as _CHANNELS_INIT_FILE

MONTAGE_PATH = op.join(op.dirname(_CHANNELS_INIT_FILE), 'data', 'montages')

_str = 'U100'


# In standard_1020, T9=LPA, T10=RPA, Nasion is the same as Iz with a
# sign-flipped Y value

def _egi_256(head_size):
    fname = op.join(MONTAGE_PATH, 'EGI_256.csd')
    montage = _read_csd(fname, head_size)
    ch_pos = montage._get_ch_pos()

    # For this cap, the Nasion is the frontmost electrode,
    # LPA/RPA we approximate by putting 75% of the way (toward the front)
    # between the two electrodes that are halfway down the ear holes
    nasion = ch_pos['E31']
    lpa = 0.75 * ch_pos['E67'] + 0.25 * ch_pos['E94']
    rpa = 0.75 * ch_pos['E219'] + 0.25 * ch_pos['E190']

    fids_montage = make_dig_montage(
        coord_frame='unknown', nasion=nasion, lpa=lpa, rpa=rpa,
    )

    montage += fids_montage  # add fiducials to montage

    return montage


def _easycap(basename, head_size):
    fname = op.join(MONTAGE_PATH, basename)
    montage = _read_theta_phi_in_degrees(fname, head_size, add_fiducials=True)
    return montage


def _hydrocel(basename, head_size):
    fname = op.join(MONTAGE_PATH, basename)
    return _read_sfp(fname, head_size)


def _str_names(ch_names):
    return [str(ch_name) for ch_name in ch_names]


def _safe_np_loadtxt(fname, **kwargs):
    out = np.genfromtxt(fname, **kwargs)
    ch_names = _str_names(out['f0'])
    others = tuple(out['f%d' % ii] for ii in range(1, len(out.dtype.fields)))
    return (ch_names,) + others


def _biosemi(basename, head_size):
    fname = op.join(MONTAGE_PATH, basename)
    fid_names = ('Nz', 'LPA', 'RPA')
    return _read_theta_phi_in_degrees(fname, head_size, fid_names)


def _mgh_or_standard(basename, head_size, coord_frame='unknown'):
    fid_names = ('Nz', 'LPA', 'RPA')
    fname = op.join(MONTAGE_PATH, basename)

    ch_names_, pos = [], []
    with open(fname) as fid:
        # Ignore units as we will scale later using the norms anyway
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

    pos = np.array(pos) / 1000.
    ch_pos = _check_dupes_odict(ch_names_, pos)
    nasion, lpa, rpa = [ch_pos.pop(n) for n in fid_names]
    if head_size is None:
        scale = 1.
    else:
        scale = head_size / np.median(np.linalg.norm(pos, axis=1))
    for value in ch_pos.values():
        value *= scale
    # if we are in MRI/MNI coordinates, we need to replace nasion, LPA, and RPA
    # with those of fsaverage for ``trans='fsaverage'`` to work
    if coord_frame == 'mri':
        lpa, nasion, rpa = [
            x['r'].copy() for x in get_mni_fiducials('fsaverage')]
    nasion *= scale
    lpa *= scale
    rpa *= scale

    return make_dig_montage(ch_pos=ch_pos, coord_frame=coord_frame,
                            nasion=nasion, lpa=lpa, rpa=rpa)


standard_montage_look_up_table = {
    'EGI_256': _egi_256,

    'easycap-M1': partial(_easycap, basename='easycap-M1.txt'),
    'easycap-M10': partial(_easycap, basename='easycap-M10.txt'),

    'GSN-HydroCel-128': partial(_hydrocel, basename='GSN-HydroCel-128.sfp'),
    'GSN-HydroCel-129': partial(_hydrocel, basename='GSN-HydroCel-129.sfp'),
    'GSN-HydroCel-256': partial(_hydrocel, basename='GSN-HydroCel-256.sfp'),
    'GSN-HydroCel-257': partial(_hydrocel, basename='GSN-HydroCel-257.sfp'),
    'GSN-HydroCel-32': partial(_hydrocel, basename='GSN-HydroCel-32.sfp'),
    'GSN-HydroCel-64_1.0': partial(_hydrocel,
                                   basename='GSN-HydroCel-64_1.0.sfp'),
    'GSN-HydroCel-65_1.0': partial(_hydrocel,
                                   basename='GSN-HydroCel-65_1.0.sfp'),

    'biosemi128': partial(_biosemi, basename='biosemi128.txt'),
    'biosemi16': partial(_biosemi, basename='biosemi16.txt'),
    'biosemi160': partial(_biosemi, basename='biosemi160.txt'),
    'biosemi256': partial(_biosemi, basename='biosemi256.txt'),
    'biosemi32': partial(_biosemi, basename='biosemi32.txt'),
    'biosemi64': partial(_biosemi, basename='biosemi64.txt'),

    'mgh60': partial(_mgh_or_standard, basename='mgh60.elc',
                     coord_frame='mri'),
    'mgh70': partial(_mgh_or_standard, basename='mgh70.elc',
                     coord_frame='mri'),
    'standard_1005': partial(_mgh_or_standard,
                             basename='standard_1005.elc', coord_frame='mri'),
    'standard_1020': partial(_mgh_or_standard,
                             basename='standard_1020.elc', coord_frame='mri'),
    'standard_alphabetic': partial(_mgh_or_standard,
                                   basename='standard_alphabetic.elc',
                                   coord_frame='mri'),
    'standard_postfixed': partial(_mgh_or_standard,
                                  basename='standard_postfixed.elc',
                                  coord_frame='mri'),
    'standard_prefixed': partial(_mgh_or_standard,
                                 basename='standard_prefixed.elc',
                                 coord_frame='mri'),
    'standard_primed': partial(_mgh_or_standard,
                               basename='standard_primed.elc',
                               coord_frame='mri'),
    'artinis-octamon': partial(_mgh_or_standard, coord_frame='mri',
                               basename='artinis-octamon.elc'),
    'artinis-brite23': partial(_mgh_or_standard, coord_frame='mri',
                               basename='artinis-brite23.elc'),
    'brainproducts-RNP-BA-128': partial(
        _easycap, basename='brainproducts-RNP-BA-128.txt')
}


def _read_sfp(fname, head_size):
    """Read .sfp BESA/EGI files."""
    # fname has been already checked
    fid_names = ('FidNz', 'FidT9', 'FidT10')
    options = dict(dtype=(_str, 'f4', 'f4', 'f4'))
    ch_names, xs, ys, zs = _safe_np_loadtxt(fname, **options)
    # deal with "headshape"
    mask = np.array([ch_name == 'headshape' for ch_name in ch_names], bool)
    hsp = np.stack([xs[mask], ys[mask], zs[mask]], axis=-1)
    mask = ~mask
    pos = np.stack([xs[mask], ys[mask], zs[mask]], axis=-1)
    ch_names = [ch_name for ch_name, m in zip(ch_names, mask) if m]
    ch_pos = _check_dupes_odict(ch_names, pos)
    del xs, ys, zs, ch_names
    # no one grants that fid names are there.
    nasion, lpa, rpa = [ch_pos.pop(n, None) for n in fid_names]

    if head_size is not None:
        scale = head_size / np.median(np.linalg.norm(pos, axis=-1))
        for value in ch_pos.values():
            value *= scale
        nasion = nasion * scale if nasion is not None else None
        lpa = lpa * scale if lpa is not None else None
        rpa = rpa * scale if rpa is not None else None

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, rpa=rpa, lpa=lpa, hsp=hsp)


def _read_csd(fname, head_size):
    # Label, Theta, Phi, Radius, X, Y, Z, off sphere surface
    options = dict(comments='//',
                   dtype=(_str, 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    ch_names, _, _, _, xs, ys, zs, _ = _safe_np_loadtxt(fname, **options)
    pos = np.stack([xs, ys, zs], axis=-1)

    if head_size is not None:
        pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

    return make_dig_montage(ch_pos=_check_dupes_odict(ch_names, pos))


def _check_dupes_odict(ch_names, pos):
    """Warn if there are duplicates, then turn to ordered dict."""
    ch_names = list(ch_names)
    dups = OrderedDict((ch_name, ch_names.count(ch_name))
                       for ch_name in ch_names)
    dups = OrderedDict((ch_name, count) for ch_name, count in dups.items()
                       if count > 1)
    n = len(dups)
    if n:
        dups = ', '.join(
            f'{ch_name} ({count})' for ch_name, count in dups.items())
        warn(f'Duplicate channel position{_pl(n)} found, the last will be '
             f'used for {dups}')
    return OrderedDict(zip(ch_names, pos))


def _read_elc(fname, head_size):
    """Read .elc files.

    Parameters
    ----------
    fname : str
        File extension is expected to be '.elc'.
    head_size : float | None
        The size of the head in [m]. If none, returns the values read from the
        file with no modification.

    Returns
    -------
    montage : instance of DigMontage
        The montage in [m].
    """
    fid_names = ('Nz', 'LPA', 'RPA')

    ch_names_, pos = [], []
    with open(fname) as fid:
        # _read_elc does require to detect the units. (see _mgh_or_standard)
        for line in fid:
            if 'UnitPosition' in line:
                units = line.split()[1]
                scale = dict(m=1., mm=1e-3)[units]
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

    pos = np.array(pos) * scale
    if head_size is not None:
        pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

    ch_pos = _check_dupes_odict(ch_names_, pos)
    nasion, lpa, rpa = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, lpa=lpa, rpa=rpa)


def _read_theta_phi_in_degrees(fname, head_size, fid_names=None,
                               add_fiducials=False):
    ch_names, theta, phi = _safe_np_loadtxt(fname, skip_header=1,
                                            dtype=(_str, 'i4', 'i4'))
    if add_fiducials:
        # Add fiducials based on 10/20 spherical coordinate definitions
        # http://chgd.umich.edu/wp-content/uploads/2014/06/
        # 10-20_system_positioning.pdf
        # extrapolated from other sensor coordinates in the Easycap layouts
        # https://www.easycap.de/wp-content/uploads/2018/02/
        # Easycap-Equidistant-Layouts.pdf
        assert fid_names is None
        fid_names = ['Nasion', 'LPA', 'RPA']
        ch_names.extend(fid_names)
        theta = np.append(theta, [115, -115, 115])
        phi = np.append(phi, [90, 0, 0])

    radii = np.full(len(phi), head_size)
    pos = _sph_to_cart(np.array([radii, np.deg2rad(phi), np.deg2rad(theta)]).T)
    ch_pos = _check_dupes_odict(ch_names, pos)

    nasion, lpa, rpa = None, None, None
    if fid_names is not None:
        nasion, lpa, rpa = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, coord_frame='unknown',
                            nasion=nasion, lpa=lpa, rpa=rpa)


def _read_elp_besa(fname, head_size):
    # This .elp is not the same as polhemus elp. see _read_isotrak_elp_points
    dtype = np.dtype('S8, S8, f8, f8, f8')
    data = np.loadtxt(fname, dtype=dtype)

    ch_names = data['f1'].astype(str).tolist()
    az = data['f2']
    horiz = data['f3']
    radius = np.abs(az / 180.)
    az = np.deg2rad(np.array([h if a >= 0. else 180 + h
                              for h, a in zip(horiz, az)]))
    pol = radius * np.pi
    rad = data['f4'] / 100
    pos = _sph_to_cart(np.array([rad, az, pol]).T)

    if head_size is not None:
        pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

    ch_pos = _check_dupes_odict(ch_names, pos)

    fid_names = ('Nz', 'LPA', 'RPA')
    # No one grants that the fid names actually exist.
    nasion, lpa, rpa = [ch_pos.pop(n, None) for n in fid_names]

    return make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa)


def _read_brainvision(fname, head_size):
    # 'BrainVision Electrodes File' format
    # Based on BrainVision Analyzer coordinate system: Defined between
    # standard electrode positions: X-axis from T7 to T8, Y-axis from Oz to
    # Fpz, Z-axis orthogonal from XY-plane through Cz, fit to a sphere if
    # idealized (when radius=1), specified in millimeters
    root = ElementTree.parse(fname).getroot()
    ch_names = [s.text for s in root.findall("./Electrode/Name")]
    theta = [float(s.text) for s in root.findall("./Electrode/Theta")]
    pol = np.deg2rad(np.array(theta))
    phi = [float(s.text) for s in root.findall("./Electrode/Phi")]
    az = np.deg2rad(np.array(phi))
    rad = [float(s.text) for s in root.findall("./Electrode/Radius")]
    rad = np.array(rad)  # specified in mm
    pos = _sph_to_cart(np.array([rad, az, pol]).T)

    if head_size is not None:
        pos *= head_size / np.median(np.linalg.norm(pos, axis=1))

    return make_dig_montage(ch_pos=_check_dupes_odict(ch_names, pos))


def _read_xyz(fname):
    """Import EEG channel locations from CSV, TSV, or XYZ files.

    CSV and TSV files should have columns 4 columns containing
    ch_name, x, y, and z. Each row represents one channel.
    XYZ files should have 5 columns containing
    count, x, y, z, and ch_name. Each row represents one channel
    CSV files should be separated by commas, TSV and XYZ files should be
    separated by tabs.

    Parameters
    ----------
    fname : str
        Name of the file to read channel locations from.

    Returns
    -------
    montage : instance of DigMontage
        The montage.
    """
    ch_names = []
    pos = []
    file_format = op.splitext(fname)[1].lower()
    with open(fname, "r") as f:
        if file_format != ".xyz":
            f.readline()  # skip header
        delimiter = "," if file_format == ".csv" else "\t"
        for row in csv.reader(f, delimiter=delimiter):
            if file_format == ".xyz":
                _, x, y, z, ch_name, *_ = row
                ch_name = ch_name.strip()  # deals with variable tab size
            else:
                ch_name, x, y, z, *_ = row
            ch_names.append(ch_name)
            pos.append((x, y, z))
    d = _check_dupes_odict(ch_names, np.array(pos, dtype=float))
    return make_dig_montage(ch_pos=d)
