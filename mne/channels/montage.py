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
import os.path as op
import re
from copy import deepcopy
from itertools import takewhile, chain
from functools import partial

import numpy as np

from ..viz import plot_montage
from ..transforms import (apply_trans, get_ras_to_neuromag_trans, _sph_to_cart,
                          _topo_to_sph, _frame_to_str, Transform)
from ..io._digitization import (_count_points_by_type,
                                _get_dig_eeg, _make_dig_points, write_dig,
                                _read_dig_fif, _format_dig_points,
                                _get_fid_coords, _coord_frame_const)
from ..io.pick import pick_types
from ..io.open import fiff_open
from ..io.constants import FIFF
from ..utils import (warn, logger, copy_function_doc_to_method_doc,
                     _check_option, _validate_type, _check_fname,
                     fill_doc)

from ._dig_montage_utils import _read_dig_montage_egi
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


def _check_ch_names_are_compatible(info_names, montage_names):
    assert isinstance(info_names, set) and isinstance(montage_names, set)

    match_set = info_names & montage_names
    not_in_montage = info_names - montage_names
    not_in_info = montage_names - info_names

    if len(not_in_montage):  # DigMontage is subset of info
        raise ValueError((
            'DigMontage is a only a subset of info.'
            ' There are {n_ch} channel positions not present in the'
            ' DigMontage. The required channels are: {ch_names}.'
        ).format(n_ch=len(not_in_montage), ch_names=not_in_montage))
    else:
        pass  # noqa

    if len(not_in_info):  # DigMontage is superset of info
        logger.info((
            'DigMontage is a superset of info. {n_ch} in DigMontage will be'
            ' ignored. The ignored channels are: {ch_names}'
        ).format(n_ch=len(not_in_info), ch_names=not_in_info))

    else:
        pass  # noqa

    return match_set


def get_builtin_montages():
    """Get a list of all builtin montages.

    Returns
    -------
    montages : list
        Names of all builtin montages that can be used by
        :func:`make_standard_montage`.
    """
    return _BUILT_IN_MONTAGES


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


class DigMontage(object):
    """Montage for digitized electrode and headshape position data.

    .. warning:: Montages are typically created using one of the helper
                 functions in the ``See Also`` section below instead of
                 instantiating this class directly.

    Parameters
    ----------
    dev_head_t : array, shape (4, 4)
        A Device-to-Head transformation matrix.
    dig : list of dict
        The object containing all the dig points.
    ch_names : list of str
        The names of the EEG channels.

    See Also
    --------
    read_dig_captrack
    read_dig_dat
    read_dig_egi
    read_dig_fif
    read_dig_hpts
    read_dig_polhemus_isotrak
    make_dig_montage

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    def __init__(self, dev_head_t=None, dig=None, ch_names=None):
        # XXX: dev_head_t now is np.array, we should add dev_head_transform
        #      (being instance of Transformation) and move the parameter to the
        #      end of the call.
        dig = list() if dig is None else dig
        _validate_type(item=dig, types=list, item_name='dig')
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

    def save(self, fname):
        """Save digitization points to FIF.

        Parameters
        ----------
        fname : str
            The filename to use. Should end in .fif or .fif.gz.
        """
        if _check_get_coord_frame(self.dig) != 'head':
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
    native_head_t = compute_native_head_t(montage)
    montage = montage.copy()  # to avoid inplace modification
    if native_head_t['from'] != FIFF.FIFFV_COORD_HEAD:
        for d in montage.dig:
            if d['coord_frame'] == native_head_t['from']:
                d['r'] = apply_trans(native_head_t, d['r'])
                d['coord_frame'] = FIFF.FIFFV_COORD_HEAD
    return montage


def read_dig_dat(fname):
    r"""Read electrode positions from a ``*.dat`` file.

    .. Warning::
        This function was implemented based on ``*.dat`` files available from
        `Compumedics <https://compumedicsneuroscan.com/scan-acquire-
        configuration-files/>`_ and might not work as expected with novel
        files. If it does not read your files correctly please contact the
        mne-python developers.

    Parameters
    ----------
    fname : path-like
        File from which to read electrode locations.

    Returns
    -------
    montage : DigMontage
        The montage.

    See Also
    --------
    read_dig_captrack
    read_dig_dat
    read_dig_egi
    read_dig_fif
    read_dig_hpts
    read_dig_polhemus_isotrak
    make_dig_montage

    Notes
    -----
    ``*.dat`` files are plain text files and can be inspected and amended with
    a plain text editor.
    """
    fname = _check_fname(fname, overwrite='read', must_exist=True)

    with open(fname, 'r') as fid:
        lines = fid.readlines()

    electrodes = {}
    nasion = lpa = rpa = None
    for i, line in enumerate(lines):
        items = line.split()
        if not items:
            continue
        elif len(items) != 5:
            raise ValueError(
                "Error reading %s, line %s has unexpected number of entries:\n"
                "%s" % (fname, i, line.rstrip()))
        num = items[1]
        if num == '67':
            continue  # centroid
        pos = np.array([float(item) for item in items[2:]])
        if num == '78':
            nasion = pos
        elif num == '76':
            lpa = pos
        elif num == '82':
            rpa = pos
        else:
            electrodes[items[0]] = pos
    return make_dig_montage(electrodes, nasion, lpa, rpa)


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
    read_dig_dat
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
    read_dig_dat
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

        cardinal    2    -5.6729  -12.3873  -30.3671
        cardinal    1    -37.6782  -10.4957   91.5228
        cardinal    3    -131.3127    9.3976  -22.2363
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
    fid_idx_to_label = {'1': 'lpa', '2': 'nasion', '3': 'rpa'}
    fid = {fid_idx_to_label[label[ii]]: this_xyz
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
    read_dig_dat
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
    read_dig_dat
    read_dig_egi
    read_dig_fif
    read_dig_hpts
    read_dig_polhemus_isotrak
    make_dig_montage
    """
    _check_fname(fname, overwrite='read', must_exist=True)
    data = _parse_brainvision_dig_montage(fname, scale=1e-3)

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
    update_ch_names : bool | None
        Whether to update or not ``ch_names`` in info. None in 0.20
    set_dig : bool
        Whether to copy or not ``montage.dig`` into ``info['dig']``.
        None in 0.20
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
    assert update_ch_names is None
    assert set_dig is None

    # This is unlikely to be trigger but it applies in all cases
    if raise_if_subset is not DEPRECATED_PARAM:
        if raise_if_subset:
            warn((
                'Using ``raise_if_subset`` to ``set_montage``  is deprecated'
                ' and ``set_dig`` will be  removed in 0.21'
            ), DeprecationWarning)
        else:
            raise ValueError(
                'Using ``raise_if_subset`` to ``set_montage``  is deprecated'
                ' and since 0.20 its value can only be True.'
                ' It will be  removed in 0.21'
            )


@fill_doc
def _set_montage(info, montage, raise_if_subset=DEPRECATED_PARAM):
    """Apply montage to data.

    With a DigMontage, this function will replace the digitizer info with
    the values specified for the particular montage.

    Usually, a montage is expected to contain the positions of all EEG
    electrodes and a warning is raised when this is not the case.

    Parameters
    ----------
    info : instance of Info
        The measurement info to update.
    %(montage)s
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
    _validate_type(montage, types=(DigMontage, type(None), str),
                   item_name='montage')
    _set_montage_deprecation_helper(montage, None, None, raise_if_subset)

    if isinstance(montage, str):  # load builtin montage
        _check_option('montage', montage, _BUILT_IN_MONTAGES)
        montage = make_standard_montage(montage)

    if isinstance(montage, DigMontage):
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
        )

        for name in matched_ch_names:
            _loc_view = info['chs'][info['ch_names'].index(name)]['loc']
            _loc_view[:6] = _backcompat_value(ch_pos[name], eeg_ref_pos)

        _names = _mnt._get_dig_names()
        info['dig'] = _format_dig_points([
            _mnt.dig[ii] for ii, name in enumerate(_names)
            if name in matched_ch_names.union({None, 'EEG000', 'REF'})
        ])

        if _mnt.dev_head_t is not None:
            info['dev_head_t'] = Transform('meg', 'head', _mnt.dev_head_t)

    else:  # None case
        info['dig'] = None
        for ch in info['chs']:
            ch['loc'] = np.full(12, np.nan)


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
    read_dig_dat
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


def read_custom_montage(fname, head_size=HEAD_SIZE_DEFAULT, unit='m',
                        coord_frame=None):
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
    coord_frame : str | None
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space. Defaults to None, which is "unknown" for
        most readers but "head" for EEGLAB.

        .. versionadded:: 0.20

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

    if coord_frame is not None:
        coord_frame = _coord_frame_const(coord_frame)
        for d in montage.dig:
            d['coord_frame'] = coord_frame

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


def compute_native_head_t(montage):
    """Compute the native-to-head transformation for a montage.

    This uses the fiducials in the native space to transform to compute the
    transform to the head coordinate frame.

    Parameters
    ----------
    montage : instance of DigMontage
        The montage.

    Returns
    -------
    native_head_t : instance of Transform
        A native-to-head transformation matrix.
    """
    # Get fiducial points and their coord_frame
    fid_coords, coord_frame = _get_fid_coords(montage.dig)
    if coord_frame == FIFF.FIFFV_COORD_HEAD:
        native_head_t = np.eye(3)
    else:
        nasion, lpa, rpa = \
            fid_coords['nasion'], fid_coords['lpa'], fid_coords['rpa']
        native_head_t = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    return Transform(coord_frame, 'head', native_head_t)


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
    :func:`read_dig_captrack`, :func:`read_dig_dat`, :func:`read_dig_egi`,
    :func:`read_dig_fif`, :func:`read_dig_polhemus_isotrak`,
    :func:`read_dig_hpts` or made with :func:`make_dig_montage`.

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
    _check_option('kind', kind, _BUILT_IN_MONTAGES)
    return standard_montage_look_up_table[kind](head_size=head_size)
