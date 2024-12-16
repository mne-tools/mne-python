# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import heapq
from collections import Counter

import numpy as np

from ..utils import Bunch, _check_fname, _validate_type, logger, verbose, warn
from .constants import FIFF, _coord_frame_named
from .tag import read_tag
from .tree import dir_tree_find
from .write import _safe_name_list, start_and_end_file, write_dig_points

_dig_kind_dict = {
    "cardinal": FIFF.FIFFV_POINT_CARDINAL,
    "hpi": FIFF.FIFFV_POINT_HPI,
    "eeg": FIFF.FIFFV_POINT_EEG,
    "extra": FIFF.FIFFV_POINT_EXTRA,
}
_dig_kind_ints = tuple(sorted(_dig_kind_dict.values()))
_dig_kind_proper = {
    "cardinal": "Cardinal",
    "hpi": "HPI",
    "eeg": "EEG",
    "extra": "Extra",
    "unknown": "Unknown",
}
_dig_kind_rev = {val: key for key, val in _dig_kind_dict.items()}
_cardinal_kind_rev = {1: "LPA", 2: "Nasion", 3: "RPA", 4: "Inion"}


def _format_dig_points(dig, enforce_order=False):
    """Format the dig points nicely."""
    if enforce_order and dig is not None:
        # reorder points based on type:
        # Fiducials/HPI, EEG, extra (headshape)
        fids_digpoints = []
        hpi_digpoints = []
        eeg_digpoints = []
        extra_digpoints = []
        head_digpoints = []

        # use a heap to enforce order on FIDS, EEG, Extra
        for idx, digpoint in enumerate(dig):
            ident = digpoint["ident"]
            kind = digpoint["kind"]

            # push onto heap based on 'ident' (for the order) for
            # each of the possible DigPoint 'kind's
            # keep track of 'idx' in case of any clashes in
            # the 'ident' variable, which can occur when
            # user passes in DigMontage + DigMontage
            if kind == FIFF.FIFFV_POINT_CARDINAL:
                heapq.heappush(fids_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_HPI:
                heapq.heappush(hpi_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_EEG:
                heapq.heappush(eeg_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_EXTRA:
                heapq.heappush(extra_digpoints, (ident, idx, digpoint))
            elif kind == FIFF.FIFFV_POINT_HEAD:
                heapq.heappush(head_digpoints, (ident, idx, digpoint))

        # now recreate dig based on sorted order
        fids_digpoints.sort(), hpi_digpoints.sort()
        eeg_digpoints.sort()
        extra_digpoints.sort(), head_digpoints.sort()
        new_dig = []
        for idx, d in enumerate(
            fids_digpoints
            + hpi_digpoints
            + extra_digpoints
            + eeg_digpoints
            + head_digpoints
        ):
            new_dig.append(d[-1])
        dig = new_dig

    return [DigPoint(d) for d in dig] if dig is not None else dig


def _get_dig_eeg(dig):
    return [d for d in dig if d["kind"] == FIFF.FIFFV_POINT_EEG]


def _count_points_by_type(dig):
    """Get the number of points of each type."""
    occurrences = Counter([d["kind"] for d in dig])
    return dict(
        fid=occurrences[FIFF.FIFFV_POINT_CARDINAL],
        hpi=occurrences[FIFF.FIFFV_POINT_HPI],
        eeg=occurrences[FIFF.FIFFV_POINT_EEG],
        extra=occurrences[FIFF.FIFFV_POINT_EXTRA],
    )


_dig_keys = {"kind", "ident", "r", "coord_frame"}


class DigPoint(dict):
    """Container for a digitization point.

    This is a simple subclass of the standard dict type designed to provide
    a readable string representation.

    Parameters
    ----------
    kind : int
        The kind of channel,
        e.g. ``FIFFV_POINT_EEG``, ``FIFFV_POINT_CARDINAL``.
    r : array, shape (3,)
        3D position in m. and coord_frame.
    ident : int
        Number specifying the identity of the point.
        e.g.  ``FIFFV_POINT_NASION`` if kind is ``FIFFV_POINT_CARDINAL``,
        or 42 if kind is ``FIFFV_POINT_EEG``.
    coord_frame : int
        The coordinate frame used, e.g. ``FIFFV_COORD_HEAD``.
    """

    def __repr__(self):  # noqa: D105
        from ..transforms import _coord_frame_name

        if self["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            id_ = _cardinal_kind_rev.get(self["ident"], "Unknown cardinal")
        else:
            id_ = _dig_kind_proper[_dig_kind_rev.get(self["kind"], "unknown")]
            id_ = f"{id_} #{self['ident']}"
        id_ = id_.rjust(10)
        cf = _coord_frame_name(self["coord_frame"])
        x, y, z = self["r"]
        if "voxel" in cf:
            pos = (f"({x:0.1f}, {y:0.1f}, {z:0.1f})").ljust(25)
        else:
            pos = (f"({x * 1e3:0.1f}, {y * 1e3:0.1f}, {z * 1e3:0.1f}) mm").ljust(25)
        return f"<DigPoint | {id_} : {pos} : {cf} frame>"

    # speed up info copy by only deep copying the mutable item
    def __deepcopy__(self, memodict):
        """Make a deepcopy."""
        return DigPoint(
            kind=self["kind"],
            r=self["r"].copy(),
            ident=self["ident"],
            coord_frame=self["coord_frame"],
        )

    def __eq__(self, other):  # noqa: D105
        """Compare two DigPoints.

        Two digpoints are equal if they are the same kind, share the same
        coordinate frame and position.
        """
        my_keys = ["kind", "ident", "coord_frame"]
        if set(self.keys()) != set(other.keys()):
            return False
        elif any(self[_] != other[_] for _ in my_keys):
            return False
        else:
            return np.allclose(self["r"], other["r"])


def _read_dig_fif(fid, meas_info, *, return_ch_names=False):
    """Read digitizer data from a FIFF file."""
    isotrak = dir_tree_find(meas_info, FIFF.FIFFB_ISOTRAK)
    dig = None
    ch_names = None
    if len(isotrak) == 0:
        logger.info("Isotrak not found")
    elif len(isotrak) > 1:
        warn("Multiple Isotrak found")
    else:
        isotrak = isotrak[0]
        coord_frame = FIFF.FIFFV_COORD_HEAD
        dig = []
        for k in range(isotrak["nent"]):
            kind = isotrak["directory"][k].kind
            pos = isotrak["directory"][k].pos
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid, pos)
                dig.append(tag.data)
            elif kind == FIFF.FIFF_MNE_COORD_FRAME:
                tag = read_tag(fid, pos)
                coord_frame = _coord_frame_named.get(int(tag.data.item()))
            elif kind == FIFF.FIFF_MNE_CH_NAME_LIST:
                tag = read_tag(fid, pos)
                ch_names = _safe_name_list(tag.data, "read", "ch_names")
        for d in dig:
            d["coord_frame"] = coord_frame
    out = _format_dig_points(dig)
    if return_ch_names:
        out = (out, ch_names)
    return out


@verbose
def write_dig(
    fname, pts, coord_frame=None, *, ch_names=None, overwrite=False, verbose=None
):
    """Write digitization data to a FIF file.

    Parameters
    ----------
    fname : path-like
        Destination file name.
    pts : iterator of dict
        Iterator through digitizer points. Each point is a dictionary with
        the keys 'kind', 'ident' and 'r'.
    coord_frame : int | str | None
        If all the points have the same coordinate frame, specify the type
        here. Can be None (default) if the points could have varying
        coordinate frames.
    ch_names : list of str | None
        Channel names associated with the digitization points, if available.

        .. versionadded:: 1.9
    %(overwrite)s

        .. versionadded:: 1.0
    %(verbose)s

        .. versionadded:: 1.0
    """
    from ..transforms import _to_const

    fname = _check_fname(fname, overwrite=overwrite)
    if coord_frame is not None:
        coord_frame = _to_const(coord_frame)
        pts_frames = {pt.get("coord_frame", coord_frame) for pt in pts}
        bad_frames = pts_frames - {coord_frame}
        if len(bad_frames) > 0:
            raise ValueError(
                "Points have coord_frame entries that are incompatible with "
                f"coord_frame={coord_frame}: {tuple(bad_frames)}."
            )
    _validate_type(ch_names, (None, list, tuple), "ch_names")
    if ch_names is not None:
        for ci, ch_name in enumerate(ch_names):
            _validate_type(ch_name, str, f"ch_names[{ci}]")

    with start_and_end_file(fname) as fid:
        write_dig_points(
            fid, pts, block=True, coord_frame=coord_frame, ch_names=ch_names
        )


_cardinal_ident_mapping = {
    FIFF.FIFFV_POINT_NASION: "nasion",
    FIFF.FIFFV_POINT_LPA: "lpa",
    FIFF.FIFFV_POINT_RPA: "rpa",
}


def _ensure_fiducials_head(dig):
    # Ensure that there are all three fiducials in the head coord frame
    fids = dict()
    for d in dig:
        if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            name = _cardinal_ident_mapping.get(d["ident"], None)
            if name is not None:
                fids[name] = d
    radius = None
    mults = dict(
        lpa=[-1, 0, 0],
        rpa=[1, 0, 0],
        nasion=[0, 1, 0],
    )
    for ident, name in _cardinal_ident_mapping.items():
        if name not in fids:
            if radius is None:
                radius = [
                    np.linalg.norm(d["r"])
                    for d in dig
                    if d["coord_frame"] == FIFF.FIFFV_COORD_HEAD
                    and not np.isnan(d["r"]).any()
                ]
                if not radius:
                    return  # can't complete, no head points
                radius = np.mean(radius)
            dig.append(
                DigPoint(
                    kind=FIFF.FIFFV_POINT_CARDINAL,
                    ident=ident,
                    r=np.array(mults[name], float) * radius,
                    coord_frame=FIFF.FIFFV_COORD_HEAD,
                )
            )


# XXXX:
# This does something really similar to _read_dig_montage_fif but:
#   - does not check coord_frame
#   - does not do any operation that implies assumptions with the names
def _get_data_as_dict_from_dig(dig, exclude_ref_channel=True):
    """Obtain coordinate data from a Dig.

    Parameters
    ----------
    dig : list of dicts
        A container of DigPoints to be added to the info['dig'].

    Returns
    -------
    ch_pos : dict
        The container of all relevant channel positions inside dig.
    """
    # Split up the dig points by category
    hsp, hpi, elp = list(), list(), list()
    fids, dig_ch_pos_location = dict(), list()
    dig = [] if dig is None else dig

    for d in dig:
        if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            fids[_cardinal_ident_mapping[d["ident"]]] = d["r"]
        elif d["kind"] == FIFF.FIFFV_POINT_HPI:
            hpi.append(d["r"])
            elp.append(d["r"])
        elif d["kind"] == FIFF.FIFFV_POINT_EXTRA:
            hsp.append(d["r"])
        elif d["kind"] == FIFF.FIFFV_POINT_EEG:
            if d["ident"] != 0 or not exclude_ref_channel:
                dig_ch_pos_location.append(d["r"])

    dig_coord_frames = set([d["coord_frame"] for d in dig])
    if len(dig_coord_frames) == 0:
        dig_coord_frames = set([FIFF.FIFFV_COORD_HEAD])
    if len(dig_coord_frames) != 1:
        raise RuntimeError(
            "Only single coordinate frame in dig is supported, "
            f"got {dig_coord_frames}"
        )
    dig_ch_pos_location = np.array(dig_ch_pos_location)
    dig_ch_pos_location.shape = (-1, 3)  # empty will be (0, 3)
    return Bunch(
        nasion=fids.get("nasion", None),
        lpa=fids.get("lpa", None),
        rpa=fids.get("rpa", None),
        hsp=np.array(hsp) if len(hsp) else None,
        hpi=np.array(hpi) if len(hpi) else None,
        elp=np.array(elp) if len(elp) else None,
        dig_ch_pos_location=dig_ch_pos_location,
        coord_frame=dig_coord_frames.pop(),
    )


def _get_fid_coords(dig, raise_error=True):
    fid_coords = Bunch(nasion=None, lpa=None, rpa=None)
    fid_coord_frames = dict()

    for d in dig:
        if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            key = _cardinal_ident_mapping[d["ident"]]
            fid_coords[key] = d["r"]
            fid_coord_frames[key] = d["coord_frame"]

    if len(fid_coord_frames) > 0 and raise_error:
        if set(fid_coord_frames.keys()) != set(["nasion", "lpa", "rpa"]):
            raise ValueError(
                f"Some fiducial points are missing (got {fid_coord_frames.keys()})."
            )

        if len(set(fid_coord_frames.values())) > 1:
            raise ValueError(
                "All fiducial points must be in the same coordinate system "
                f"(got {len(fid_coord_frames)})"
            )

    coord_frame = fid_coord_frames.popitem()[1] if fid_coord_frames else None

    return fid_coords, coord_frame


def _coord_frame_const(coord_frame):
    from ..transforms import _str_to_frame

    if not isinstance(coord_frame, str) or coord_frame not in _str_to_frame:
        raise ValueError(
            f"coord_frame must be one of {sorted(_str_to_frame.keys())}, got "
            f"{coord_frame}"
        )
    return _str_to_frame[coord_frame]


def _make_dig_points(
    nasion=None,
    lpa=None,
    rpa=None,
    hpi=None,
    extra_points=None,
    dig_ch_pos=None,
    *,
    coord_frame="head",
    add_missing_fiducials=False,
):
    """Construct digitizer info for the info.

    Parameters
    ----------
    nasion : array-like | numpy.ndarray, shape (3,) | None
        Point designated as the nasion point.
    lpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the left auricular point.
    rpa : array-like |  numpy.ndarray, shape (3,) | None
        Point designated as the right auricular point.
    hpi : array-like | numpy.ndarray, shape (n_points, 3) | None
        Points designated as head position indicator points.
    extra_points : array-like | numpy.ndarray, shape (n_points, 3)
        Points designed as the headshape points.
    dig_ch_pos : dict
        Dict of EEG channel positions.
    coord_frame : str
        The coordinate frame of the points. Usually this is "unknown"
        for native digitizer space. Defaults to "head".
    add_missing_fiducials : bool
        If True, add fiducials to the dig points if they are not present.
        Requires that coord_frame='head' and that lpa, nasion, and rpa are all
        None.

    Returns
    -------
    dig : list of dicts
        A container of DigPoints to be added to the info['dig'].
    """
    coord_frame = _coord_frame_const(coord_frame)

    dig = []
    if lpa is not None:
        lpa = np.asarray(lpa)
        if lpa.shape != (3,):
            raise ValueError(f"LPA should have the shape (3,) instead of {lpa.shape}")
        dig.append(
            {
                "r": lpa,
                "ident": FIFF.FIFFV_POINT_LPA,
                "kind": FIFF.FIFFV_POINT_CARDINAL,
                "coord_frame": coord_frame,
            }
        )
    if nasion is not None:
        nasion = np.asarray(nasion)
        if nasion.shape != (3,):
            raise ValueError(
                f"Nasion should have the shape (3,) instead of {nasion.shape}"
            )
        dig.append(
            {
                "r": nasion,
                "ident": FIFF.FIFFV_POINT_NASION,
                "kind": FIFF.FIFFV_POINT_CARDINAL,
                "coord_frame": coord_frame,
            }
        )
    if rpa is not None:
        rpa = np.asarray(rpa)
        if rpa.shape != (3,):
            raise ValueError(f"RPA should have the shape (3,) instead of {rpa.shape}")
        dig.append(
            {
                "r": rpa,
                "ident": FIFF.FIFFV_POINT_RPA,
                "kind": FIFF.FIFFV_POINT_CARDINAL,
                "coord_frame": coord_frame,
            }
        )
    if hpi is not None:
        hpi = np.asarray(hpi)
        if hpi.ndim != 2 or hpi.shape[1] != 3:
            raise ValueError(
                f"HPI should have the shape (n_points, 3) instead of {hpi.shape}"
            )
        for idx, point in enumerate(hpi):
            dig.append(
                {
                    "r": point,
                    "ident": idx + 1,
                    "kind": FIFF.FIFFV_POINT_HPI,
                    "coord_frame": coord_frame,
                }
            )
    if extra_points is not None:
        extra_points = np.asarray(extra_points)
        if len(extra_points) and extra_points.shape[1] != 3:
            raise ValueError(
                "Points should have the shape (n_points, 3) instead of "
                f"{extra_points.shape}"
            )
        for idx, point in enumerate(extra_points):
            dig.append(
                {
                    "r": point,
                    "ident": idx + 1,
                    "kind": FIFF.FIFFV_POINT_EXTRA,
                    "coord_frame": coord_frame,
                }
            )
    if dig_ch_pos is not None:
        idents = []
        use_arange = False
        for key, value in dig_ch_pos.items():
            _validate_type(key, str, "dig_ch_pos")
            try:
                idents.append(int(key[-3:]))
            except ValueError:
                use_arange = True
            _validate_type(value, (np.ndarray, list, tuple), "dig_ch_pos")
            value = np.array(value, dtype=float)
            dig_ch_pos[key] = value
            if value.shape != (3,):
                raise RuntimeError(
                    "The position should be a 1D array of 3 floats. "
                    f"Provided shape {value.shape}."
                )
        if use_arange:
            idents = np.arange(1, len(dig_ch_pos) + 1)
        for key, ident in zip(dig_ch_pos, idents):
            dig.append(
                {
                    "r": dig_ch_pos[key],
                    "ident": int(ident),
                    "kind": FIFF.FIFFV_POINT_EEG,
                    "coord_frame": coord_frame,
                }
            )
    if add_missing_fiducials:
        assert coord_frame == FIFF.FIFFV_COORD_HEAD
        # These being none is really an assumption that if you have one you
        # should have all three. But we can relax this later if necessary.
        assert lpa is None
        assert rpa is None
        assert nasion is None
        _ensure_fiducials_head(dig)

    return _format_dig_points(dig)


def _call_make_dig_points(nasion, lpa, rpa, hpi, extra, convert=True):
    from ..transforms import (
        Transform,
        apply_trans,
        get_ras_to_neuromag_trans,
    )

    if convert:
        neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        nasion = apply_trans(neuromag_trans, nasion)
        lpa = apply_trans(neuromag_trans, lpa)
        rpa = apply_trans(neuromag_trans, rpa)

        if hpi is not None:
            hpi = apply_trans(neuromag_trans, hpi)

        extra = apply_trans(neuromag_trans, extra).astype(np.float32)
    else:
        neuromag_trans = None

    ctf_head_t = Transform(fro="ctf_head", to="head", trans=neuromag_trans)

    info_dig = _make_dig_points(
        nasion=nasion, lpa=lpa, rpa=rpa, hpi=hpi, extra_points=extra
    )

    return info_dig, ctf_head_t


##############################################################################
# From artemis123 (we have modified the function a bit)
def _artemis123_read_pos(nas, lpa, rpa, hpi, extra):
    # move into MNE head coords
    dig_points, _ = _call_make_dig_points(nas, lpa, rpa, hpi, extra)
    return dig_points


##############################################################################
# From bti
def _make_bti_dig_points(
    nasion,
    lpa,
    rpa,
    hpi,
    extra,
    convert=False,
    use_hpi=False,
    bti_dev_t=False,
    dev_ctf_t=False,
):
    from ..transforms import (
        Transform,
        combine_transforms,
        invert_transform,
    )

    _hpi = hpi if use_hpi else None
    info_dig, ctf_head_t = _call_make_dig_points(nasion, lpa, rpa, _hpi, extra, convert)

    if convert:
        t = combine_transforms(
            invert_transform(bti_dev_t), dev_ctf_t, "meg", "ctf_head"
        )
        dev_head_t = combine_transforms(t, ctf_head_t, "meg", "head")
    else:
        dev_head_t = Transform("meg", "head", trans=None)

    return info_dig, dev_head_t, ctf_head_t  # ctf_head_t should not be needed
