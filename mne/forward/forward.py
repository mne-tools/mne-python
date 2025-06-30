# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# The computations in this code were primarily derived from Matti Hämäläinen's
# C code.

import os
import re
import shutil
import tempfile
from copy import deepcopy
from os import PathLike
from os import path as op
from pathlib import Path
from time import time

import numpy as np
from scipy import sparse

from .._fiff.constants import FIFF
from .._fiff.matrix import (
    _read_named_matrix,
    _transpose_named_matrix,
    write_named_matrix,
)
from .._fiff.meas_info import (
    Info,
    _make_ch_names_mapping,
    _read_bad_channels,
    _read_extended_ch_info,
    _write_bad_channels,
    _write_ch_infos,
    write_info,
)
from .._fiff.open import fiff_open
from .._fiff.pick import pick_channels, pick_channels_forward, pick_info, pick_types
from .._fiff.tag import find_tag, read_tag
from .._fiff.tree import dir_tree_find
from .._fiff.write import (
    end_block,
    start_and_end_file,
    start_block,
    write_coord_trans,
    write_id,
    write_int,
    write_string,
)
from ..epochs import BaseEpochs
from ..evoked import Evoked, EvokedArray
from ..html_templates import _get_html_template
from ..io import BaseRaw, RawArray
from ..label import Label
from ..source_estimate import _BaseSourceEstimate, _BaseVectorSourceEstimate
from ..source_space._source_space import (
    SourceSpaces,
    _get_src_nn,
    _read_source_spaces_from_tree,
    _set_source_space_vertices,
    _src_kind_dict,
    _write_source_spaces_to_fid,
    find_source_space_hemi,
)
from ..surface import _normal_orth
from ..transforms import invert_transform, transform_surface_to, write_trans
from ..utils import (
    _check_compensation_grade,
    _check_fname,
    _check_option,
    _check_stc_units,
    _import_h5io_funcs,
    _on_missing,
    _stamp_to_dt,
    _validate_type,
    check_fname,
    fill_doc,
    get_subjects_dir,
    has_mne_c,
    logger,
    repr_html,
    run_subprocess,
    verbose,
    warn,
)


class Forward(dict):
    """Forward class to represent info from forward solution.

    Like :class:`mne.Info`, this data structure behaves like a dictionary.
    It contains all metadata necessary for a forward solution.

    .. warning::
        This class should not be modified or created by users.
        Forward objects should be obtained using
        :func:`mne.make_forward_solution` or :func:`mne.read_forward_solution`.

    Attributes
    ----------
    ch_names : list of str
        A convenience wrapper accessible as ``fwd.ch_names`` which wraps
        ``fwd['info']['ch_names']``.

    See Also
    --------
    mne.make_forward_solution
    mne.read_forward_solution

    Notes
    -----
    Forward data is accessible via string keys using standard
    :class:`python:dict` access (e.g., ``fwd['nsource'] == 4096``):

        source_ori : int
            The source orientation, either ``FIFF.FIFFV_MNE_FIXED_ORI`` or
            ``FIFF.FIFFV_MNE_FREE_ORI``.
        coord_frame : int
            The coordinate frame of the forward solution, usually
            ``FIFF.FIFFV_COORD_HEAD``.
        nsource : int
            The number of source locations.
        nchan : int
            The number of channels.
        sol : dict
            The forward solution, with entries:

            ``'data'`` : ndarray, shape (n_channels, nsource * n_ori)
                The forward solution data. The shape will be
                ``(n_channels, nsource)`` for a fixed-orientation forward and
                ``(n_channels, nsource * 3)`` for a free-orientation forward.
            ``'row_names'`` : list of str
                The channel names.
        mri_head_t : instance of Transform
            The mri ↔ head transformation that was used.
        info : instance of :class:`~mne.Info`
            The measurement information (with contents reduced compared to that
            of the original data).
        src : instance of :class:`~mne.SourceSpaces`
            The source space used during forward computation. This can differ
            from the original source space as:

            1. Source points are removed due to proximity to (or existing
               outside)
               the inner skull surface.
            2. The source space will be converted to the ``coord_frame`` of the
               forward solution, which typically means it gets converted from
               MRI to head coordinates.
        source_rr : ndarray, shape (n_sources, 3)
            The source locations.
        source_nn : ndarray, shape (n_sources, 3)
            The source normals. Will be all +Z (``(0, 0, 1.)``) for volume
            source spaces. For surface source spaces, these are normal to the
            cortical surface.
        surf_ori : int
            Whether ``sol`` is surface-oriented with the surface normal in the
            Z component (``FIFF.FIFFV_MNE_FIXED_ORI``) or +Z in the given
            ``coord_frame`` in the Z component (``FIFF.FIFFV_MNE_FREE_ORI``).

    Forward objects also have some attributes that are accessible via ``.``
    access, like ``fwd.ch_names``.
    """

    def copy(self):
        """Copy the Forward instance."""
        return Forward(deepcopy(self))

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save the forward solution.

        Parameters
        ----------
        %(fname_fwd)s
        %(overwrite)s
        %(verbose)s
        """
        write_forward_solution(fname, self, overwrite=overwrite)

    def _get_src_type_and_ori_for_repr(self):
        src_types = np.array([src["type"] for src in self["src"]])

        if (src_types == "surf").all():
            src_type = f"Surface with {self['nsource']} vertices"
        elif (src_types == "vol").all():
            src_type = f"Volume with {self['nsource']} grid points"
        elif (src_types == "discrete").all():
            src_type = f"Discrete with {self['nsource']} dipoles"
        else:
            count_string = ""
            if (src_types == "surf").any():
                count_string += f"{(src_types == 'surf').sum()} surface, "
            if (src_types == "vol").any():
                count_string += f"{(src_types == 'vol').sum()} volume, "
            if (src_types == "discrete").any():
                count_string += f"{(src_types == 'discrete').sum()} discrete, "
            count_string = count_string.rstrip(", ")
            src_type = f"Mixed ({count_string}) with {self['nsource']} vertices"

        if self["source_ori"] == FIFF.FIFFV_MNE_UNKNOWN_ORI:
            src_ori = "Unknown"
        elif self["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
            src_ori = "Fixed"
        elif self["source_ori"] == FIFF.FIFFV_MNE_FREE_ORI:
            src_ori = "Free"

        return src_type, src_ori

    def __repr__(self):
        """Summarize forward info instead of printing all."""
        entr = "<Forward"

        nchan = len(pick_types(self["info"], meg=True, eeg=False, exclude=[]))
        entr += " | " + f"MEG channels: {nchan}"
        nchan = len(pick_types(self["info"], meg=False, eeg=True, exclude=[]))
        entr += " | " + f"EEG channels: {nchan}"

        src_type, src_ori = self._get_src_type_and_ori_for_repr()
        entr += f" | Source space: {src_type}"
        entr += f" | Source orientation: {src_ori}"
        entr += ">"

        return entr

    @repr_html
    def _repr_html_(self):
        src_descr, src_ori = self._get_src_type_and_ori_for_repr()

        t = _get_html_template("repr", "forward.html.jinja")
        html = t.render(
            info=self["info"],
            source_space_descr=src_descr,
            source_orientation=src_ori,
        )
        return html

    @property
    def ch_names(self):
        return self["info"]["ch_names"]

    def pick_channels(self, ch_names, ordered=False):
        """Pick channels from this forward operator.

        Parameters
        ----------
        ch_names : list of str
            List of channels to include.
        ordered : bool
            If true (default False), treat ``include`` as an ordered list
            rather than a set.

        Returns
        -------
        fwd : instance of Forward.
            The modified forward model.

        Notes
        -----
        Operates in-place.

        .. versionadded:: 0.20.0
        """
        return pick_channels_forward(
            self, ch_names, exclude=[], ordered=ordered, copy=False, verbose=False
        )


def _block_diag(A, n):
    """Construct a block diagonal from a packed structure.

    You have to try it on a matrix to see what it's doing.

    If A is not sparse, then returns a sparse block diagonal "bd",
    diagonalized from the
    elements in "A".
    "A" is ma x na, comprising bdn=(na/"n") blocks of submatrices.
    Each submatrix is ma x "n", and these submatrices are
    placed down the diagonal of the matrix.

    If A is already sparse, then the operation is reversed, yielding
    a block
    row matrix, where each set of n columns corresponds to a block element
    from the block diagonal.

    Parameters
    ----------
    A : array
        The matrix
    n : int
        The block size

    Returns
    -------
    bd : scipy.sparse.csc_array
        The block diagonal matrix
    """
    if sparse.issparse(A):  # then make block sparse
        raise NotImplementedError("sparse reversal not implemented yet")
    ma, na = A.shape
    bdn = na // int(n)  # number of submatrices

    if na % n > 0:
        raise ValueError("Width of matrix must be a multiple of n")

    tmp = np.arange(ma * bdn, dtype=np.int64).reshape(bdn, ma)
    tmp = np.tile(tmp, (1, n))
    ii = tmp.ravel()

    jj = np.arange(na, dtype=np.int64)[None, :]
    jj = jj * np.ones(ma, dtype=np.int64)[:, None]
    jj = jj.T.ravel()  # column indices foreach sparse bd

    bd = sparse.coo_array((A.T.ravel(), np.c_[ii, jj].T)).tocsc()

    return bd


def _get_tag_int(fid, node, name, id_):
    """Check we have an appropriate tag."""
    tag = find_tag(fid, node, id_)
    if tag is None:
        fid.close()
        raise ValueError(name + " tag not found")
    return int(tag.data.item())


def _read_one(fid, node):
    """Read all interesting stuff for one forward solution."""
    # This function assumes the fid is open as a context manager
    if node is None:
        return None

    one = Forward()
    one["source_ori"] = _get_tag_int(
        fid, node, "Source orientation", FIFF.FIFF_MNE_SOURCE_ORIENTATION
    )
    one["coord_frame"] = _get_tag_int(
        fid, node, "Coordinate frame", FIFF.FIFF_MNE_COORD_FRAME
    )
    one["nsource"] = _get_tag_int(
        fid, node, "Number of sources", FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS
    )
    one["nchan"] = _get_tag_int(fid, node, "Number of channels", FIFF.FIFF_NCHAN)
    try:
        one["sol"] = _read_named_matrix(
            fid, node, FIFF.FIFF_MNE_FORWARD_SOLUTION, transpose=True
        )
        one["_orig_sol"] = one["sol"]["data"].copy()
    except Exception:
        logger.error("Forward solution data not found")
        raise

    try:
        fwd_type = FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD
        one["sol_grad"] = _read_named_matrix(fid, node, fwd_type, transpose=True)
        one["_orig_sol_grad"] = one["sol_grad"]["data"].copy()
    except Exception:
        one["sol_grad"] = None

    if one["sol"]["data"].shape[0] != one["nchan"] or (
        one["sol"]["data"].shape[1] != one["nsource"]
        and one["sol"]["data"].shape[1] != 3 * one["nsource"]
    ):
        raise ValueError("Forward solution matrix has wrong dimensions")

    if one["sol_grad"] is not None:
        if one["sol_grad"]["data"].shape[0] != one["nchan"] or (
            one["sol_grad"]["data"].shape[1] != 3 * one["nsource"]
            and one["sol_grad"]["data"].shape[1] != 3 * 3 * one["nsource"]
        ):
            raise ValueError("Forward solution gradient matrix has wrong dimensions")

    return one


@fill_doc
def _read_forward_meas_info(tree, fid):
    """Read light measurement info from forward operator.

    Parameters
    ----------
    tree : tree
        FIF tree structure.
    fid : file id
        The file id.

    Returns
    -------
    %(info_not_none)s
    """
    # This function assumes fid is being used as a context manager
    info = Info()
    info._unlocked = True

    # Information from the MRI file
    parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    if len(parent_mri) == 0:
        raise ValueError("No parent MEG information found in operator")
    parent_mri = parent_mri[0]

    tag = find_tag(fid, parent_mri, FIFF.FIFF_MNE_FILE_NAME)
    info["mri_file"] = tag.data if tag is not None else None
    tag = find_tag(fid, parent_mri, FIFF.FIFF_PARENT_FILE_ID)
    info["mri_id"] = tag.data if tag is not None else None

    # Information from the MEG file
    parent_meg = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)
    if len(parent_meg) == 0:
        raise ValueError("No parent MEG information found in operator")
    parent_meg = parent_meg[0]

    tag = find_tag(fid, parent_meg, FIFF.FIFF_MNE_FILE_NAME)
    info["meas_file"] = tag.data if tag is not None else None
    tag = find_tag(fid, parent_meg, FIFF.FIFF_PARENT_FILE_ID)
    info["meas_id"] = tag.data if tag is not None else None

    # Add channel information
    info["chs"] = chs = list()
    for k in range(parent_meg["nent"]):
        kind = parent_meg["directory"][k].kind
        pos = parent_meg["directory"][k].pos
        if kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
    ch_names_mapping = _read_extended_ch_info(chs, parent_meg, fid)
    info._update_redundant()

    # Get the MRI <-> head coordinate transformation
    tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
    coord_head = FIFF.FIFFV_COORD_HEAD
    coord_mri = FIFF.FIFFV_COORD_MRI
    coord_device = FIFF.FIFFV_COORD_DEVICE
    coord_ctf_head = FIFF.FIFFV_MNE_COORD_CTF_HEAD
    if tag is None:
        raise ValueError("MRI/head coordinate transformation not found")
    cand = tag.data
    if cand["from"] == coord_mri and cand["to"] == coord_head:
        info["mri_head_t"] = cand
    else:
        raise ValueError("MRI/head coordinate transformation not found")

    # Get the MEG device <-> head coordinate transformation
    tag = find_tag(fid, parent_meg, FIFF.FIFF_COORD_TRANS)
    if tag is None:
        raise ValueError("MEG/head coordinate transformation not found")
    cand = tag.data
    if cand["from"] == coord_device and cand["to"] == coord_head:
        info["dev_head_t"] = cand
    elif cand["from"] == coord_ctf_head and cand["to"] == coord_head:
        info["ctf_head_t"] = cand
    else:
        raise ValueError("MEG/head coordinate transformation not found")

    bads = _read_bad_channels(fid, parent_meg, ch_names_mapping=ch_names_mapping)
    # clean up our bad list, old versions could have non-existent bads
    info["bads"] = [bad for bad in bads if bad in info["ch_names"]]

    # Check if a custom reference has been applied
    tag = find_tag(fid, parent_mri, FIFF.FIFF_MNE_CUSTOM_REF)
    if tag is None:
        tag = find_tag(fid, parent_mri, 236)  # Constant 236 used before v0.11

    info["custom_ref_applied"] = int(tag.data.item()) if tag is not None else False
    info._unlocked = False
    return info


def _subject_from_forward(forward):
    """Get subject id from inverse operator."""
    return forward["src"]._subject


# This sets the forward solution order (and gives human-readable names)
_FWD_ORDER = dict(
    meg="MEG",
    eeg="EEG",
)


@verbose
def _merge_fwds(fwds, *, verbose=None):
    """Merge loaded forward dicts into one dict."""
    fwd = None
    first_key = None
    combined = list()
    for key in _FWD_ORDER:
        if key not in fwds:
            continue
        if fwd is None:  # assign
            fwd = fwds[key]
            first_key = key
            combined.append(_FWD_ORDER[key])
            continue
        a = fwd
        b = fwds[key]
        a_kind, b_kind = _FWD_ORDER[first_key], _FWD_ORDER[key]
        combined.append(b_kind)
        if (
            a["sol"]["data"].shape[1] != b["sol"]["data"].shape[1]
            or a["source_ori"] != b["source_ori"]
            or a["nsource"] != b["nsource"]
            or a["coord_frame"] != b["coord_frame"]
        ):
            raise ValueError(
                f"The {a_kind} and {b_kind} forward solutions do not match"
            )
        for k in ("sol", "sol_grad"):
            if a[k] is None:
                continue
            a[k]["data"] = np.r_[a[k]["data"], b[k]["data"]]
            a[f"_orig_{k}"] = np.r_[a[f"_orig_{k}"], b[f"_orig_{k}"]]
            a[k]["nrow"] = a[k]["nrow"] + b[k]["nrow"]
            a[k]["row_names"] = a[k]["row_names"] + b[k]["row_names"]
        a["nchan"] = a["nchan"] + b["nchan"]
    if len(fwds) > 1:
        logger.info(f"    Forward solutions combined: {', '.join(combined)}")
    return fwd


@verbose
def read_forward_solution(fname, include=(), exclude=(), *, ordered=True, verbose=None):
    """Read a forward solution a.k.a. lead field.

    Parameters
    ----------
    fname : path-like
        The file name, which should end with ``-fwd.fif``, ``-fwd.fif.gz``,
        ``_fwd.fif``, ``_fwd.fif.gz``, ``-fwd.h5``, or ``_fwd.h5``.
    include : list, optional
        List of names of channels to include. If empty all channels
        are included.
    exclude : list, optional
        List of names of channels to exclude. If empty include all channels.
    %(ordered)s
    %(verbose)s

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    See Also
    --------
    write_forward_solution, make_forward_solution

    Notes
    -----
    Forward solutions, which are derived from an original forward solution with
    free orientation, are always stored on disk as forward solution with free
    orientation in X/Y/Z RAS coordinates. To apply any transformation to the
    forward operator (surface orientation, fixed orientation) please apply
    :func:`convert_forward_solution` after reading the forward solution with
    :func:`read_forward_solution`.

    Forward solutions, which are derived from an original forward solution with
    fixed orientation, are stored on disk as forward solution with fixed
    surface-based orientations. Please note that the transformation to
    surface-based, fixed orientation cannot be reverted after loading the
    forward solution with :func:`read_forward_solution`.
    """
    check_fname(
        fname,
        "forward",
        ("-fwd.fif", "-fwd.fif.gz", "_fwd.fif", "_fwd.fif.gz", "-fwd.h5", "_fwd.h5"),
    )
    fname = _check_fname(fname=fname, must_exist=True, overwrite="read")
    #   Open the file, create directory
    logger.info(f"Reading forward solution from {fname}...")
    if fname.suffix == ".h5":
        return _read_forward_hdf5(fname)
    f, tree, _ = fiff_open(fname)
    with f as fid:
        #   Find all forward solutions
        fwds = dir_tree_find(tree, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
        if len(fwds) == 0:
            raise ValueError(f"No forward solutions in {fname}")

        #   Parent MRI data
        parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        if len(parent_mri) == 0:
            raise ValueError(f"No parent MRI information in {fname}")
        parent_mri = parent_mri[0]

        src = _read_source_spaces_from_tree(fid, tree, patch_stats=False)
        for s in src:
            s["id"] = find_source_space_hemi(s)

        fwd = None

        #   Locate and read the forward solutions
        megnode = None
        eegnode = None
        for k in range(len(fwds)):
            tag = find_tag(fid, fwds[k], FIFF.FIFF_MNE_INCLUDED_METHODS)
            if tag is None:
                raise ValueError("Methods not listed for one of the forward solutions")

            if tag.data == FIFF.FIFFV_MNE_MEG:
                megnode = fwds[k]
            elif tag.data == FIFF.FIFFV_MNE_EEG:
                eegnode = fwds[k]

        fwds = dict()
        megfwd = _read_one(fid, megnode)
        if megfwd is not None:
            fwds["meg"] = megfwd
            if is_fixed_orient(megfwd):
                ori = "fixed"
            else:
                ori = "free"
            logger.info(
                "    Read MEG forward solution (%d sources, "
                "%d channels, %s orientations)",
                megfwd["nsource"],
                megfwd["nchan"],
                ori,
            )
        del megfwd

        eegfwd = _read_one(fid, eegnode)
        if eegfwd is not None:
            fwds["eeg"] = eegfwd
            if is_fixed_orient(eegfwd):
                ori = "fixed"
            else:
                ori = "free"
            logger.info(
                "    Read EEG forward solution (%d sources, "
                "%d channels, %s orientations)",
                eegfwd["nsource"],
                eegfwd["nchan"],
                ori,
            )
        del eegfwd

        fwd = _merge_fwds(fwds)
        del fwds

        #   Get the MRI <-> head coordinate transformation
        tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
        if tag is None:
            raise ValueError("MRI/head coordinate transformation not found")
        mri_head_t = tag.data
        if (
            mri_head_t["from"] != FIFF.FIFFV_COORD_MRI
            or mri_head_t["to"] != FIFF.FIFFV_COORD_HEAD
        ):
            mri_head_t = invert_transform(mri_head_t)
            if (
                mri_head_t["from"] != FIFF.FIFFV_COORD_MRI
                or mri_head_t["to"] != FIFF.FIFFV_COORD_HEAD
            ):
                fid.close()
                raise ValueError("MRI/head coordinate transformation not found")
        fwd["mri_head_t"] = mri_head_t

        #
        # get parent MEG info
        #
        fwd["info"] = _read_forward_meas_info(tree, fid)

        # MNE environment
        parent_env = dir_tree_find(tree, FIFF.FIFFB_MNE_ENV)
        if len(parent_env) > 0:
            parent_env = parent_env[0]
            tag = find_tag(fid, parent_env, FIFF.FIFF_MNE_ENV_WORKING_DIR)
            if tag is not None:
                with fwd["info"]._unlock():
                    fwd["info"]["working_dir"] = tag.data
            tag = find_tag(fid, parent_env, FIFF.FIFF_MNE_ENV_COMMAND_LINE)
            if tag is not None:
                with fwd["info"]._unlock():
                    fwd["info"]["command_line"] = tag.data

    #   Transform the source spaces to the correct coordinate frame
    #   if necessary

    # Make sure forward solution is in either the MRI or HEAD coordinate frame
    if fwd["coord_frame"] not in (FIFF.FIFFV_COORD_MRI, FIFF.FIFFV_COORD_HEAD):
        raise ValueError(
            "Only forward solutions computed in MRI or head coordinates are acceptable"
        )

    # Transform each source space to the HEAD or MRI coordinate frame,
    # depending on the coordinate frame of the forward solution
    # NOTE: the function transform_surface_to will also work on discrete and
    # volume sources
    nuse = 0
    for s in src:
        try:
            s = transform_surface_to(s, fwd["coord_frame"], mri_head_t)
        except Exception as inst:
            raise ValueError(f"Could not transform source space ({inst})")

        nuse += s["nuse"]

    # Make sure the number of sources match after transformation
    if nuse != fwd["nsource"]:
        raise ValueError("Source spaces do not match the forward solution.")

    logger.info(
        "    Source spaces transformed to the forward solution coordinate frame"
    )
    fwd["src"] = src

    #   Handle the source locations and orientations
    fwd["source_rr"] = np.concatenate([ss["rr"][ss["vertno"], :] for ss in src], axis=0)

    #   Store original source orientations
    fwd["_orig_source_ori"] = fwd["source_ori"]

    #   Deal with include and exclude
    pick_channels_forward(fwd, include=include, exclude=exclude, copy=False)

    if is_fixed_orient(fwd, orig=True):
        fwd["source_nn"] = np.concatenate(
            [_src["nn"][_src["vertno"], :] for _src in fwd["src"]], axis=0
        )
        fwd["source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
        fwd["surf_ori"] = True
    else:
        fwd["source_nn"] = np.kron(np.ones((fwd["nsource"], 1)), np.eye(3))
        fwd["source_ori"] = FIFF.FIFFV_MNE_FREE_ORI
        fwd["surf_ori"] = False
    return Forward(fwd)


@verbose
def convert_forward_solution(
    fwd, surf_ori=False, force_fixed=False, copy=True, use_cps=True, *, verbose=None
):
    """Convert forward solution between different source orientations.

    Parameters
    ----------
    fwd : Forward
        The forward solution to modify.
    surf_ori : bool, optional (default False)
        Use surface-based source coordinate system? Note that force_fixed=True
        implies surf_ori=True.
    force_fixed : bool, optional (default False)
        If True, force fixed source orientation mode.
    copy : bool
        Whether to return a new instance or modify in place.
    %(use_cps)s
    %(verbose)s

    Returns
    -------
    fwd : Forward
        The modified forward solution.
    """
    fwd = fwd.copy() if copy else fwd

    if force_fixed is True:
        surf_ori = True

    if any([src["type"] == "vol" for src in fwd["src"]]) and force_fixed:
        raise ValueError(
            "Forward operator was generated with sources from a "
            "volume source space. Conversion to fixed orientation is not "
            "possible. Consider using a discrete source space if you have "
            "meaningful normal orientations."
        )

    if surf_ori and use_cps:
        if any(s.get("patch_inds") is not None for s in fwd["src"]):
            logger.info(
                "    Average patch normals will be employed in "
                "the rotation to the local surface coordinates.."
                ".."
            )
        else:
            use_cps = False
            logger.info(
                "    No patch info available. The standard source "
                "space normals will be employed in the rotation "
                "to the local surface coordinates...."
            )

    # We need to change these entries (only):
    # 1. source_nn
    # 2. sol['data']
    # 3. sol['ncol']
    # 4. sol_grad['data']
    # 5. sol_grad['ncol']
    # 6. source_ori

    if is_fixed_orient(fwd, orig=True) or (force_fixed and not use_cps):
        # Fixed
        fwd["source_nn"] = np.concatenate(
            [_get_src_nn(s, use_cps) for s in fwd["src"]], axis=0
        )
        if not is_fixed_orient(fwd, orig=True):
            logger.info(
                "    Changing to fixed-orientation forward "
                "solution with surface-based source orientations..."
            )
            fix_rot = _block_diag(fwd["source_nn"].T, 1)
            # newer versions of numpy require explicit casting here, so *= no
            # longer works
            fwd["sol"]["data"] = (fwd["_orig_sol"] @ fix_rot).astype("float32")
            fwd["sol"]["ncol"] = fwd["nsource"]
            if fwd["sol_grad"] is not None:
                x = sparse.block_diag([fix_rot] * 3)
                fwd["sol_grad"]["data"] = fwd["_orig_sol_grad"] @ x
                fwd["sol_grad"]["ncol"] = 3 * fwd["nsource"]
        fwd["source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
        fwd["surf_ori"] = True

    elif surf_ori:  # Free, surf-oriented
        #   Rotate the local source coordinate systems
        fwd["source_nn"] = np.kron(np.ones((fwd["nsource"], 1)), np.eye(3))
        logger.info("    Converting to surface-based source orientations...")
        #   Actually determine the source orientations
        pp = 0
        for s in fwd["src"]:
            if s["type"] in ["surf", "discrete"]:
                nn = _get_src_nn(s, use_cps)
                stop = pp + 3 * s["nuse"]
                fwd["source_nn"][pp:stop] = _normal_orth(nn).reshape(-1, 3)
                pp = stop
                del nn
            else:
                pp += 3 * s["nuse"]

        #   Rotate the solution components as well
        if force_fixed:
            fwd["source_nn"] = fwd["source_nn"][2::3, :]
            fix_rot = _block_diag(fwd["source_nn"].T, 1)
            # newer versions of numpy require explicit casting here, so *= no
            # longer works
            fwd["sol"]["data"] = (fwd["_orig_sol"] @ fix_rot).astype("float32")
            fwd["sol"]["ncol"] = fwd["nsource"]
            if fwd["sol_grad"] is not None:
                x = sparse.block_diag([fix_rot] * 3)
                fwd["sol_grad"]["data"] = fwd["_orig_sol_grad"] @ x
                fwd["sol_grad"]["ncol"] = 3 * fwd["nsource"]
            fwd["source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
            fwd["surf_ori"] = True
        else:
            surf_rot = _block_diag(fwd["source_nn"].T, 3)
            fwd["sol"]["data"] = fwd["_orig_sol"] @ surf_rot
            fwd["sol"]["ncol"] = 3 * fwd["nsource"]
            if fwd["sol_grad"] is not None:
                x = sparse.block_diag([surf_rot] * 3)
                fwd["sol_grad"]["data"] = fwd["_orig_sol_grad"] @ x
                fwd["sol_grad"]["ncol"] = 9 * fwd["nsource"]
            fwd["source_ori"] = FIFF.FIFFV_MNE_FREE_ORI
            fwd["surf_ori"] = True

    else:  # Free, cartesian
        logger.info("    Cartesian source orientations...")
        fwd["source_nn"] = np.tile(np.eye(3), (fwd["nsource"], 1))
        fwd["sol"]["data"] = fwd["_orig_sol"].copy()
        fwd["sol"]["ncol"] = 3 * fwd["nsource"]
        if fwd["sol_grad"] is not None:
            fwd["sol_grad"]["data"] = fwd["_orig_sol_grad"].copy()
            fwd["sol_grad"]["ncol"] = 9 * fwd["nsource"]
        fwd["source_ori"] = FIFF.FIFFV_MNE_FREE_ORI
        fwd["surf_ori"] = False

    logger.info("    [done]")

    return fwd


@verbose
def write_forward_solution(fname, fwd, overwrite=False, verbose=None):
    """Write forward solution to a file.

    Parameters
    ----------
    %(fname_fwd)s
    fwd : Forward
        Forward solution.
    %(overwrite)s
    %(verbose)s

    See Also
    --------
    read_forward_solution

    Notes
    -----
    Forward solutions, which are derived from an original forward solution with
    free orientation, are always stored on disk as forward solution with free
    orientation in X/Y/Z RAS coordinates. Transformations (surface orientation,
    fixed orientation) will be reverted. To reapply any transformation to the
    forward operator please apply :func:`convert_forward_solution` after
    reading the forward solution with :func:`read_forward_solution`.

    Forward solutions, which are derived from an original forward solution with
    fixed orientation, are stored on disk as forward solution with fixed
    surface-based orientations. Please note that the transformation to
    surface-based, fixed orientation cannot be reverted after loading the
    forward solution with :func:`read_forward_solution`.
    """
    check_fname(
        fname,
        "forward",
        ("-fwd.fif", "-fwd.fif.gz", "_fwd.fif", "_fwd.fif.gz", "-fwd.h5", "_fwd.h5"),
    )

    # check for file existence and expand `~` if present
    fname = _check_fname(fname, overwrite)
    if fname.suffix == ".h5":
        _write_forward_hdf5(fname, fwd)
    else:
        with start_and_end_file(fname) as fid:
            _write_forward_solution(fid, fwd)


def _write_forward_hdf5(fname, fwd):
    _, write_hdf5 = _import_h5io_funcs()
    write_hdf5(fname, dict(fwd=fwd), overwrite=True)


def _read_forward_hdf5(fname):
    read_hdf5, _ = _import_h5io_funcs()
    fwd = Forward(read_hdf5(fname)["fwd"])
    fwd["info"] = Info(fwd["info"])
    fwd["src"] = SourceSpaces(fwd["src"])
    return fwd


def _write_forward_solution(fid, fwd):
    start_block(fid, FIFF.FIFFB_MNE)

    #
    # MNE env
    #
    start_block(fid, FIFF.FIFFB_MNE_ENV)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    data = fwd["info"].get("working_dir", None)
    if data is not None:
        write_string(fid, FIFF.FIFF_MNE_ENV_WORKING_DIR, data)
    data = fwd["info"].get("command_line", None)
    if data is not None:
        write_string(fid, FIFF.FIFF_MNE_ENV_COMMAND_LINE, data)
    end_block(fid, FIFF.FIFFB_MNE_ENV)

    #
    # Information from the MRI file
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    write_string(fid, FIFF.FIFF_MNE_FILE_NAME, fwd["info"]["mri_file"])
    if fwd["info"]["mri_id"] is not None:
        write_id(fid, FIFF.FIFF_PARENT_FILE_ID, fwd["info"]["mri_id"])
    # store the MRI to HEAD transform in MRI file
    write_coord_trans(fid, fwd["info"]["mri_head_t"])
    end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    # write measurement info
    write_forward_meas_info(fid, fwd["info"])

    # invert our original source space transform
    src = list()
    for s in fwd["src"]:
        s = deepcopy(s)
        try:
            # returns source space to original coordinate frame
            # usually MRI
            s = transform_surface_to(s, fwd["mri_head_t"]["from"], fwd["mri_head_t"])
        except Exception as inst:
            raise ValueError(f"Could not transform source space ({inst})")
        src.append(s)

    #
    # Write the source spaces (again)
    #
    _write_source_spaces_to_fid(fid, src)
    n_vert = sum([ss["nuse"] for ss in src])
    if fwd["_orig_source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
        n_col = n_vert
    else:
        n_col = 3 * n_vert

    # Undo transformations
    sol = fwd["_orig_sol"].copy()
    if fwd["sol_grad"] is not None:
        sol_grad = fwd["_orig_sol_grad"].copy()
    else:
        sol_grad = None

    if fwd["surf_ori"] is True:
        if fwd["_orig_source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI:
            warn(
                "The forward solution, which is stored on disk now, is based "
                "on a forward solution with fixed orientation. Please note "
                "that the transformation to surface-based, fixed orientation "
                "cannot be reverted after loading the forward solution with "
                "read_forward_solution.",
                RuntimeWarning,
            )
        else:
            warn(
                "This forward solution is based on a forward solution with "
                "free orientation. The original forward solution is stored "
                "on disk in X/Y/Z RAS coordinates. Any transformation "
                "(surface orientation or fixed orientation) will be "
                "reverted. To reapply any transformation to the forward "
                "operator please apply convert_forward_solution after "
                "reading the forward solution with read_forward_solution.",
                RuntimeWarning,
            )

    #
    # MEG forward solution
    #
    picks_meg = pick_types(fwd["info"], meg=True, eeg=False, ref_meg=False, exclude=[])
    picks_eeg = pick_types(fwd["info"], meg=False, eeg=True, ref_meg=False, exclude=[])
    n_meg = len(picks_meg)
    n_eeg = len(picks_eeg)
    row_names_meg = [fwd["sol"]["row_names"][p] for p in picks_meg]
    row_names_eeg = [fwd["sol"]["row_names"][p] for p in picks_eeg]

    if n_meg > 0:
        meg_solution = dict(
            data=sol[picks_meg],
            nrow=n_meg,
            ncol=n_col,
            row_names=row_names_meg,
            col_names=[],
        )
        _transpose_named_matrix(meg_solution)
        start_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
        write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, FIFF.FIFFV_MNE_MEG)
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, fwd["coord_frame"])
        write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION, fwd["_orig_source_ori"])
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, n_vert)
        write_int(fid, FIFF.FIFF_NCHAN, n_meg)
        write_named_matrix(fid, FIFF.FIFF_MNE_FORWARD_SOLUTION, meg_solution)
        if sol_grad is not None:
            meg_solution_grad = dict(
                data=sol_grad[picks_meg],
                nrow=n_meg,
                ncol=n_col * 3,
                row_names=row_names_meg,
                col_names=[],
            )
            _transpose_named_matrix(meg_solution_grad)
            write_named_matrix(
                fid, FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD, meg_solution_grad
            )
        end_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)

    #
    #  EEG forward solution
    #
    if n_eeg > 0:
        eeg_solution = dict(
            data=sol[picks_eeg],
            nrow=n_eeg,
            ncol=n_col,
            row_names=row_names_eeg,
            col_names=[],
        )
        _transpose_named_matrix(eeg_solution)
        start_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
        write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, FIFF.FIFFV_MNE_EEG)
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, fwd["coord_frame"])
        write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION, fwd["_orig_source_ori"])
        write_int(fid, FIFF.FIFF_NCHAN, n_eeg)
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, n_vert)
        write_named_matrix(fid, FIFF.FIFF_MNE_FORWARD_SOLUTION, eeg_solution)
        if sol_grad is not None:
            eeg_solution_grad = dict(
                data=sol_grad[picks_eeg],
                nrow=n_eeg,
                ncol=n_col * 3,
                row_names=row_names_eeg,
                col_names=[],
            )
            _transpose_named_matrix(eeg_solution_grad)
            write_named_matrix(
                fid, FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD, eeg_solution_grad
            )
        end_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)

    end_block(fid, FIFF.FIFFB_MNE)


def is_fixed_orient(forward, orig=False):
    """Check if the forward operator is fixed orientation.

    Parameters
    ----------
    forward : instance of Forward
        The forward.
    orig : bool
        If True, consider the original source orientation.
        If False (default), consider the current source orientation.

    Returns
    -------
    fixed_ori : bool
        Whether or not it is fixed orientation.
    """
    if orig:  # if we want to know about the original version
        fixed_ori = forward["_orig_source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI
    else:  # most of the time we want to know about the current version
        fixed_ori = forward["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI
    return fixed_ori


@fill_doc
def write_forward_meas_info(fid, info):
    """Write measurement info stored in forward solution.

    Parameters
    ----------
    fid : file id
        The file id
    %(info_not_none)s
    """
    info._check_consistency()
    #
    # Information from the MEG file
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)
    write_string(fid, FIFF.FIFF_MNE_FILE_NAME, info["meas_file"])
    if info["meas_id"] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info["meas_id"])
    # get transformation from CTF and DEVICE to HEAD coordinate frame
    meg_head_t = info.get("dev_head_t", info.get("ctf_head_t"))
    if meg_head_t is None:
        fid.close()
        raise ValueError("Head<-->sensor transform not found")
    write_coord_trans(fid, meg_head_t)

    ch_names_mapping = dict()
    if "chs" in info:
        #  Channel information
        ch_names_mapping = _make_ch_names_mapping(info["chs"])
        write_int(fid, FIFF.FIFF_NCHAN, len(info["chs"]))
        _write_ch_infos(fid, info["chs"], False, ch_names_mapping)
    if "bads" in info and len(info["bads"]) > 0:
        #   Bad channels
        _write_bad_channels(fid, info["bads"], ch_names_mapping)

    end_block(fid, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)


def _select_orient_forward(forward, info, noise_cov=None, copy=True):
    """Prepare forward solution for inverse solvers."""
    # fwd['sol']['row_names'] may be different order from fwd['info']['chs']
    fwd_sol_ch_names = forward["sol"]["row_names"]
    all_ch_names = set(fwd_sol_ch_names)
    all_bads = set(info["bads"])
    if noise_cov is not None:
        all_ch_names &= set(noise_cov["names"])
        all_bads |= set(noise_cov["bads"])
    else:
        noise_cov = dict(bads=info["bads"])
    ch_names = [
        c["ch_name"]
        for c in info["chs"]
        if c["ch_name"] not in all_bads and c["ch_name"] in all_ch_names
    ]

    if not len(info["bads"]) == len(noise_cov["bads"]) or not all(
        b in noise_cov["bads"] for b in info["bads"]
    ):
        logger.info(
            'info["bads"] and noise_cov["bads"] do not match, '
            "excluding bad channels from both"
        )

    # check the compensation grade
    _check_compensation_grade(forward["info"], info, "forward")

    n_chan = len(ch_names)
    logger.info("Computing inverse operator with %d channels.", n_chan)
    forward = pick_channels_forward(forward, ch_names, ordered=True, copy=copy)
    info_idx = [info["ch_names"].index(name) for name in ch_names]
    info_picked = pick_info(info, info_idx)
    forward["info"]._check_consistency()
    info_picked._check_consistency()
    return forward, info_picked


def _triage_loose(src, loose, fixed="auto"):
    _validate_type(loose, (str, dict, "numeric"), "loose")
    _validate_type(fixed, (str, bool), "fixed")
    orig_loose = loose
    if isinstance(loose, str):
        _check_option("loose", loose, ("auto",))
        if fixed is True:
            loose = 0.0
        else:  # False or auto
            loose = 0.2 if src.kind == "surface" else 1.0
    src_types = set(_src_kind_dict[s["type"]] for s in src)
    if not isinstance(loose, dict):
        loose = float(loose)
        loose = {key: loose for key in src_types}
    loose_keys = set(loose.keys())
    if loose_keys != src_types:
        raise ValueError(
            f"loose, if dict, must have keys {sorted(src_types)} to match the "
            f"source space, got {sorted(loose_keys)}"
        )
    # if fixed is auto it can be ignored, if it's False it can be ignored,
    # only really need to care about fixed=True
    if fixed is True:
        if not all(v == 0.0 for v in loose.values()):
            raise ValueError(
                f'When using fixed=True, loose must be 0. or "auto", got {orig_loose}'
            )
    elif fixed is False:
        if any(v == 0.0 for v in loose.values()):
            raise ValueError(
                'If loose==0., then fixed must be True or "auto", got False'
            )
    del fixed

    for key, this_loose in loose.items():
        if key not in ("surface", "discrete") and this_loose != 1:
            raise ValueError(
                'loose parameter has to be 1 or "auto" for non-surface/'
                f'discrete source spaces, got loose["{key}"] = {this_loose}'
            )
        if not 0 <= this_loose <= 1:
            raise ValueError(f"loose ({key}) must be between 0 and 1, got {this_loose}")
    return loose


@verbose
def compute_orient_prior(forward, loose="auto", verbose=None):
    """Compute orientation prior.

    Parameters
    ----------
    forward : instance of Forward
        Forward operator.
    %(loose)s
    %(verbose)s

    Returns
    -------
    orient_prior : ndarray, shape (n_sources,)
        Orientation priors.

    See Also
    --------
    compute_depth_prior
    """
    _validate_type(forward, Forward, "forward")
    n_sources = forward["sol"]["data"].shape[1]

    loose = _triage_loose(forward["src"], loose)
    orient_prior = np.ones(n_sources, dtype=np.float64)
    if is_fixed_orient(forward):
        if any(v > 0.0 for v in loose.values()):
            raise ValueError(
                "loose must be 0. with forward operator "
                f"with fixed orientation, got {loose}"
            )
        return orient_prior
    if all(v == 1.0 for v in loose.values()):
        return orient_prior
    # We actually need non-unity prior, compute it for each source space
    # separately
    if not forward["surf_ori"]:
        raise ValueError(
            "Forward operator is not oriented in surface "
            "coordinates. loose parameter should be 1. "
            f"not {loose}."
        )
    start = 0
    logged = dict()
    for s in forward["src"]:
        this_type = _src_kind_dict[s["type"]]
        use_loose = loose[this_type]
        if not logged.get(this_type):
            if use_loose == 1.0:
                name = "free"
            else:
                name = "fixed" if use_loose == 0.0 else "loose"
            logger.info(
                f"Applying {name.ljust(5)} dipole orientations to "
                f"{this_type.ljust(7)} source spaces: {use_loose}"
            )
            logged[this_type] = True
        stop = start + 3 * s["nuse"]
        orient_prior[start:stop:3] *= use_loose
        orient_prior[start + 1 : stop : 3] *= use_loose
        start = stop
    return orient_prior


def _restrict_gain_matrix(G, info):
    """Restrict gain matrix entries for optimal depth weighting."""
    # Figure out which ones have been used
    if len(info["chs"]) != G.shape[0]:
        raise ValueError(
            f'G.shape[0] ({G.shape[0]}) and length of info["chs"] ({len(info["chs"])}) '
            "do not match."
        )
    for meg, eeg, kind in (
        ("grad", False, "planar"),
        ("mag", False, "magnetometer or axial gradiometer"),
        (False, True, "EEG"),
    ):
        sel = pick_types(info, meg=meg, eeg=eeg, ref_meg=False, exclude=[])
        if len(sel) > 0:
            logger.info("    %d %s channels", len(sel), kind)
            break
    else:
        warn("Could not find MEG or EEG channels to limit depth channels")
        sel = slice(None)
    return G[sel]


@verbose
def compute_depth_prior(
    forward,
    info,
    exp=0.8,
    limit=10.0,
    limit_depth_chs=False,
    combine_xyz="spectral",
    noise_cov=None,
    rank=None,
    verbose=None,
):
    """Compute depth prior for depth weighting.

    Parameters
    ----------
    forward : instance of Forward
        The forward solution.
    %(info_not_none)s
    exp : float
        Exponent for the depth weighting, must be between 0 and 1.
    limit : float | None
        The upper bound on depth weighting.
        Can be None to be bounded by the largest finite prior.
    limit_depth_chs : bool | 'whiten'
        How to deal with multiple channel types in depth weighting.
        The default is True, which whitens based on the source sensitivity
        of the highest-SNR channel type. See Notes for details.

        .. versionchanged:: 0.18
           Added the "whiten" option.
    combine_xyz : 'spectral' | 'fro'
        When a loose (or free) orientation is used, how the depth weighting
        for each triplet should be calculated.
        If 'spectral', use the squared spectral norm of Gk.
        If 'fro', use the squared Frobenius norm of Gk.

        .. versionadded:: 0.18
    noise_cov : instance of Covariance | None
        The noise covariance to use to whiten the gain matrix when
        ``limit_depth_chs='whiten'``.

        .. versionadded:: 0.18
    %(rank_none)s

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    depth_prior : ndarray, shape (n_vertices,)
        The depth prior.

    See Also
    --------
    compute_orient_prior

    Notes
    -----
    The defaults used by the minimum norm code and sparse solvers differ.
    In particular, the values for MNE are::

        compute_depth_prior(..., limit=10., limit_depth_chs=True,
                            combine_xyz='spectral')

    In sparse solvers and LCMV, the values are::

        compute_depth_prior(..., limit=None, limit_depth_chs='whiten',
                            combine_xyz='fro')

    The ``limit_depth_chs`` argument can take the following values:

    * :data:`python:True` (default)
          Use only grad channels in depth weighting (equivalent to MNE C
          minimum-norm code). If grad channels aren't present, only mag
          channels will be used (if no mag, then eeg). This makes the depth
          prior dependent only on the sensor geometry (and relationship
          to the sources).
    * ``'whiten'``
          Compute a whitener and apply it to the gain matrix before computing
          the depth prior. In this case ``noise_cov`` must not be None.
          Whitening the gain matrix makes the depth prior
          depend on both sensor geometry and the data of interest captured
          by the noise covariance (e.g., projections, SNR).

          .. versionadded:: 0.18
    * :data:`python:False`
          Use all channels. Not recommended since the depth weighting will be
          biased toward whichever channel type has the largest values in
          SI units (such as EEG being orders of magnitude larger than MEG).
    """
    from ..cov import Covariance, compute_whitener

    _validate_type(forward, Forward, "forward")
    patch_areas = forward.get("patch_areas", None)
    is_fixed_ori = is_fixed_orient(forward)
    G = forward["sol"]["data"]
    logger.info("Creating the depth weighting matrix...")
    _validate_type(noise_cov, (Covariance, None), "noise_cov", "Covariance or None")
    _validate_type(limit_depth_chs, (str, bool), "limit_depth_chs")
    if isinstance(limit_depth_chs, str):
        if limit_depth_chs != "whiten":
            raise ValueError(
                f'limit_depth_chs, if str, must be "whiten", got {limit_depth_chs}'
            )
        if not isinstance(noise_cov, Covariance):
            raise ValueError(
                'With limit_depth_chs="whiten", noise_cov must be'
                f" a Covariance, got {type(noise_cov)}"
            )
    if combine_xyz is not False:  # private / expert option
        _check_option("combine_xyz", combine_xyz, ("fro", "spectral"))

    # If possible, pick best depth-weighting channels
    if limit_depth_chs is True:
        G = _restrict_gain_matrix(G, info)
    elif limit_depth_chs == "whiten":
        whitener, _ = compute_whitener(
            noise_cov, info, pca=True, rank=rank, verbose=False
        )
        G = np.dot(whitener, G)

    # Compute the gain matrix
    if is_fixed_ori or combine_xyz in ("fro", False):
        d = np.sum(G**2, axis=0)
        if not (is_fixed_ori or combine_xyz is False):
            d = d.reshape(-1, 3).sum(axis=1)
        # Spherical leadfield can be zero at the center
        d[d == 0.0] = np.min(d[d != 0.0])
    else:  # 'spectral'
        # n_pos = G.shape[1] // 3
        # The following is equivalent to this, but 4-10x faster
        # d = np.zeros(n_pos)
        # for k in range(n_pos):
        #     Gk = G[:, 3 * k:3 * (k + 1)]
        #     x = np.dot(Gk.T, Gk)
        #     d[k] = linalg.svdvals(x)[0]
        G.shape = (G.shape[0], -1, 3)
        d = np.linalg.norm(
            np.einsum("svj,svk->vjk", G, G),  # vector dot prods
            ord=2,  # ord=2 spectral (largest s.v.)
            axis=(1, 2),
        )
        G.shape = (G.shape[0], -1)

    # XXX Currently the fwd solns never have "patch_areas" defined
    if patch_areas is not None:
        if not is_fixed_ori and combine_xyz is False:
            patch_areas = np.repeat(patch_areas, 3)
        d /= patch_areas**2
        logger.info("    Patch areas taken into account in the depth weighting")

    w = 1.0 / d
    if limit is not None:
        ws = np.sort(w)
        weight_limit = limit**2
        if limit_depth_chs is False:
            # match old mne-python behavior
            # we used to do ind = np.argmin(ws), but this is 0 by sort above
            n_limit = 0
            limit = ws[0] * weight_limit
        else:
            # match C code behavior
            limit = ws[-1]
            n_limit = len(d)
            if ws[-1] > weight_limit * ws[0]:
                ind = np.where(ws > weight_limit * ws[0])[0][0]
                limit = ws[ind]
                n_limit = ind

        logger.info(
            "    limit = %d/%d = %f", n_limit + 1, len(d), np.sqrt(limit / ws[0])
        )
        scale = 1.0 / limit
        logger.info(f"    scale = {scale:g} exp = {exp:g}")
        w = np.minimum(w / limit, 1)
    depth_prior = w**exp

    if not (is_fixed_ori or combine_xyz is False):
        depth_prior = np.repeat(depth_prior, 3)

    return depth_prior


def _stc_src_sel(
    src, stc, on_missing="raise", extra=", likely due to forward calculations"
):
    """Select the vertex indices of a source space using a source estimate."""
    if isinstance(stc, list):
        vertices = stc
    else:
        assert isinstance(stc, _BaseSourceEstimate)
        vertices = stc.vertices
    del stc
    if not len(src) == len(vertices):
        raise RuntimeError(
            f"Mismatch between number of source spaces ({len(src)}) and "
            f"STC vertices ({len(vertices)})"
        )
    src_sels, stc_sels, out_vertices = [], [], []
    src_offset = stc_offset = 0
    for s, v in zip(src, vertices):
        joint_sel = np.intersect1d(s["vertno"], v)
        src_sels.append(np.searchsorted(s["vertno"], joint_sel) + src_offset)
        src_offset += len(s["vertno"])
        idx = np.searchsorted(v, joint_sel)
        stc_sels.append(idx + stc_offset)
        stc_offset += len(v)
        out_vertices.append(np.array(v)[idx])
    src_sel = np.concatenate(src_sels)
    stc_sel = np.concatenate(stc_sels)
    assert len(src_sel) == len(stc_sel) == sum(len(v) for v in out_vertices)

    n_stc = sum(len(v) for v in vertices)
    n_joint = len(src_sel)
    if n_joint != n_stc:
        msg = (
            f"Only {n_joint} of {n_stc} SourceEstimate "
            f"{'vertex' if n_stc == 1 else 'vertices'} found in source space{extra}"
        )
        _on_missing(on_missing, msg)
    return src_sel, stc_sel, out_vertices


def _fill_measurement_info(info, fwd, sfreq, data):
    """Fill the measurement info of a Raw or Evoked object."""
    sel = pick_channels(info["ch_names"], fwd["sol"]["row_names"], ordered=False)
    info = pick_info(info, sel)
    info["bads"] = []

    now = time()
    sec = np.floor(now)
    usec = 1e6 * (now - sec)

    # this is probably correct based on what's done in meas_info.py...
    with info._unlock(check_after=True):
        info.update(
            meas_id=fwd["info"]["meas_id"],
            file_id=info["meas_id"],
            meas_date=_stamp_to_dt((int(sec), int(usec))),
            highpass=0.0,
            lowpass=sfreq / 2.0,
            sfreq=sfreq,
            projs=[],
        )

    # reorder data (which is in fwd order) to match that of info
    order = [fwd["sol"]["row_names"].index(name) for name in info["ch_names"]]
    data = data[order]

    return info, data


@verbose
def _apply_forward(
    fwd, stc, start=None, stop=None, on_missing="raise", use_cps=True, verbose=None
):
    """Apply forward model and return data, times, ch_names."""
    _validate_type(stc, _BaseSourceEstimate, "stc", "SourceEstimate")
    _validate_type(fwd, Forward, "fwd")
    if isinstance(stc, _BaseVectorSourceEstimate):
        vector = True
        fwd = convert_forward_solution(fwd, force_fixed=False, surf_ori=False)
    else:
        vector = False
        if not is_fixed_orient(fwd):
            fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=use_cps)

    if np.all(stc.data > 0):
        warn(
            "Source estimate only contains currents with positive values. "
            'Use pick_ori="normal" when computing the inverse to compute '
            "currents not current magnitudes."
        )

    _check_stc_units(stc)

    src_sel, stc_sel, _ = _stc_src_sel(fwd["src"], stc, on_missing=on_missing)
    gain = fwd["sol"]["data"]
    stc_sel = slice(None) if len(stc_sel) == len(stc.data) else stc_sel
    times = stc.times[start:stop].copy()
    stc_data = stc.data[stc_sel, ..., start:stop].reshape(-1, len(times))
    del stc
    if vector:
        gain = gain.reshape(len(gain), gain.shape[1] // 3, 3)
    gain = gain[:, src_sel].reshape(len(gain), -1)
    # save some memory if possible

    logger.info("Projecting source estimate to sensor space...")
    data = np.dot(gain, stc_data)
    logger.info("[done]")
    return data, times


@verbose
def apply_forward(
    fwd,
    stc,
    info,
    start=None,
    stop=None,
    use_cps=True,
    on_missing="raise",
    verbose=None,
):
    """Project source space currents to sensor space using a forward operator.

    The sensor space data is computed for all channels present in fwd. Use
    pick_channels_forward or pick_types_forward to restrict the solution to a
    subset of channels.

    The function returns an Evoked object, which is constructed from
    evoked_template. The evoked_template should be from the same MEG system on
    which the original data was acquired. An exception will be raised if the
    forward operator contains channels that are not present in the template.

    Parameters
    ----------
    fwd : Forward
        Forward operator to use.
    stc : SourceEstimate
        The source estimate from which the sensor space data is computed.
    %(info_not_none)s
    start : int, optional
        Index of first time sample (index not time is seconds).
    stop : int, optional
        Index of first time sample not to include (index not time is seconds).
    %(use_cps)s

        .. versionadded:: 0.15
    %(on_missing_fwd)s
        Default is "raise".

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    evoked : Evoked
        Evoked object with computed sensor space data.

    See Also
    --------
    apply_forward_raw: Compute sensor space data and return a Raw object.
    """
    _validate_type(info, Info, "info")
    _validate_type(fwd, Forward, "forward")
    info._check_consistency()

    # make sure evoked_template contains all channels in fwd
    for ch_name in fwd["sol"]["row_names"]:
        if ch_name not in info["ch_names"]:
            raise ValueError(
                f"Channel {ch_name} of forward operator not present in evoked_template."
            )

    # project the source estimate to the sensor space
    data, times = _apply_forward(
        fwd, stc, start, stop, on_missing=on_missing, use_cps=use_cps
    )

    # fill the measurement info
    sfreq = float(1.0 / stc.tstep)
    info, data = _fill_measurement_info(info, fwd, sfreq, data)

    evoked = EvokedArray(data, info, times[0], nave=1)

    evoked._set_times(times)
    evoked._update_first_last()

    return evoked


@verbose
def apply_forward_raw(
    fwd,
    stc,
    info,
    start=None,
    stop=None,
    on_missing="raise",
    use_cps=True,
    verbose=None,
):
    """Project source space currents to sensor space using a forward operator.

    The sensor space data is computed for all channels present in fwd. Use
    pick_channels_forward or pick_types_forward to restrict the solution to a
    subset of channels.

    The function returns a Raw object, which is constructed using provided
    info. The info object should be from the same MEG system on which the
    original data was acquired. An exception will be raised if the forward
    operator contains channels that are not present in the info.

    Parameters
    ----------
    fwd : Forward
        Forward operator to use.
    stc : SourceEstimate
        The source estimate from which the sensor space data is computed.
    %(info_not_none)s
    start : int, optional
        Index of first time sample (index not time is seconds).
    stop : int, optional
        Index of first time sample not to include (index not time is seconds).
    %(on_missing_fwd)s
        Default is "raise".

        .. versionadded:: 0.18
    %(use_cps)s

        .. versionadded:: 0.21
    %(verbose)s

    Returns
    -------
    raw : Raw object
        Raw object with computed sensor space data.

    See Also
    --------
    apply_forward: Compute sensor space data and return an Evoked object.
    """
    # make sure info contains all channels in fwd
    for ch_name in fwd["sol"]["row_names"]:
        if ch_name not in info["ch_names"]:
            raise ValueError(
                f"Channel {ch_name} of forward operator not present in info."
            )

    # project the source estimate to the sensor space
    data, times = _apply_forward(
        fwd, stc, start, stop, on_missing=on_missing, use_cps=use_cps
    )

    sfreq = 1.0 / stc.tstep
    info, data = _fill_measurement_info(info, fwd, sfreq, data)
    with info._unlock():
        info["projs"] = []
    # store sensor data in Raw object using the info
    raw = RawArray(data, info, first_samp=int(np.round(times[0] * sfreq)))
    raw._projector = None
    return raw


@fill_doc
def restrict_forward_to_stc(fwd, stc, on_missing="ignore"):
    """Restrict forward operator to active sources in a source estimate.

    Parameters
    ----------
    fwd : instance of Forward
        Forward operator.
    stc : instance of SourceEstimate
        Source estimate.
    %(on_missing_fwd)s
        Default is "ignore".

        .. versionadded:: 0.18

    Returns
    -------
    fwd_out : instance of Forward
        Restricted forward operator.

    See Also
    --------
    restrict_forward_to_label
    """
    _validate_type(on_missing, str, "on_missing")
    _check_option("on_missing", on_missing, ("ignore", "warn", "raise"))
    src_sel, _, _ = _stc_src_sel(fwd["src"], stc, on_missing=on_missing)
    del stc
    return _restrict_forward_to_src_sel(fwd, src_sel)


def _restrict_forward_to_src_sel(fwd, src_sel):
    fwd_out = deepcopy(fwd)
    # figure out the vertno we are keeping
    idx_sel = np.concatenate(
        [[[si] * len(s["vertno"]), s["vertno"]] for si, s in enumerate(fwd["src"])],
        axis=-1,
    )
    assert idx_sel.ndim == 2 and idx_sel.shape[0] == 2
    assert idx_sel.shape[1] == fwd["nsource"]
    idx_sel = idx_sel[:, src_sel]

    fwd_out["source_rr"] = fwd["source_rr"][src_sel]
    fwd_out["nsource"] = len(src_sel)

    if is_fixed_orient(fwd):
        idx = src_sel
        if fwd["sol_grad"] is not None:
            idx_grad = (3 * src_sel[:, None] + np.arange(3)).ravel()
    else:
        idx = (3 * src_sel[:, None] + np.arange(3)).ravel()
        if fwd["sol_grad"] is not None:
            idx_grad = (9 * src_sel[:, None] + np.arange(9)).ravel()

    fwd_out["source_nn"] = fwd["source_nn"][idx]
    fwd_out["sol"]["data"] = fwd["sol"]["data"][:, idx]
    if fwd["sol_grad"] is not None:
        fwd_out["sol_grad"]["data"] = fwd["sol_grad"]["data"][:, idx_grad]
    fwd_out["sol"]["ncol"] = len(idx)

    if is_fixed_orient(fwd, orig=True):
        idx = src_sel
        if fwd["sol_grad"] is not None:
            idx_grad = (3 * src_sel[:, None] + np.arange(3)).ravel()
    else:
        idx = (3 * src_sel[:, None] + np.arange(3)).ravel()
        if fwd["sol_grad"] is not None:
            idx_grad = (9 * src_sel[:, None] + np.arange(9)).ravel()

    fwd_out["_orig_sol"] = fwd["_orig_sol"][:, idx]
    if fwd["sol_grad"] is not None:
        fwd_out["_orig_sol_grad"] = fwd["_orig_sol_grad"][:, idx_grad]

    vertices = [idx_sel[1][idx_sel[0] == si] for si in range(len(fwd_out["src"]))]
    _set_source_space_vertices(fwd_out["src"], vertices)

    return fwd_out


def restrict_forward_to_label(fwd, labels):
    """Restrict forward operator to labels.

    Parameters
    ----------
    fwd : Forward
        Forward operator.
    labels : instance of Label | list
        Label object or list of label objects.

    Returns
    -------
    fwd_out : dict
        Restricted forward operator.

    See Also
    --------
    restrict_forward_to_stc
    """
    vertices = [np.array([], int), np.array([], int)]

    if not isinstance(labels, list):
        labels = [labels]

    # Get vertices separately of each hemisphere from all label
    for label in labels:
        _validate_type(label, Label, "label", "Label or list")
        i = 0 if label.hemi == "lh" else 1
        vertices[i] = np.append(vertices[i], label.vertices)
    # Remove duplicates and sort
    vertices = [np.unique(vert_hemi) for vert_hemi in vertices]
    vertices = [
        vert_hemi[np.isin(vert_hemi, s["vertno"])]
        for vert_hemi, s in zip(vertices, fwd["src"])
    ]
    src_sel, _, _ = _stc_src_sel(fwd["src"], vertices, on_missing="raise")
    return _restrict_forward_to_src_sel(fwd, src_sel)


def _do_forward_solution(
    subject,
    meas,
    fname=None,
    src=None,
    spacing=None,
    mindist=None,
    bem=None,
    mri=None,
    trans=None,
    eeg=True,
    meg=True,
    fixed=False,
    grad=False,
    mricoord=False,
    overwrite=False,
    subjects_dir=None,
    verbose=None,
):
    """Calculate a forward solution for a subject using MNE-C routines.

    This is kept around for testing purposes.

    This function wraps to mne_do_forward_solution, so the mne
    command-line tools must be installed and accessible from Python.

    Parameters
    ----------
    subject : str
        Name of the subject.
    meas : Raw | Epochs | Evoked | str
        If Raw or Epochs, a temporary evoked file will be created and
        saved to a temporary directory. If str, then it should be a
        filename to a file with measurement information the mne
        command-line tools can understand (i.e., raw or evoked).
    fname : path-like | None
        Destination forward solution filename. If None, the solution
        will be created in a temporary directory, loaded, and deleted.
    src : str | None
        Source space name. If None, the MNE default is used.
    spacing : str
        The spacing to use. Can be ``'#'`` for spacing in mm, ``'ico#'`` for a
        recursively subdivided icosahedron, or ``'oct#'`` for a recursively
        subdivided octahedron (e.g., ``spacing='ico4'``). Default is 7 mm.
    mindist : float | str | None
        Minimum distance measof sources from inner skull surface (in mm).
        If None, the MNE default value is used. If string, ``'all'``
        indicates to include all points.
    bem : str | None
        Name of the BEM to use (e.g., ``"sample-5120-5120-5120"``). If None
        (Default), the MNE default will be used.
    mri : dict | path-like | None
        The name of the trans file in FIF format.
        If None, ``trans`` must not be None.
    trans : dict | path-like | None
        File name of the trans file in text format.
        If None, ``mri`` must not be None.
    eeg : bool
        If True (Default), include EEG computations.
    meg : bool
        If True (Default), include MEG computations.
    fixed : bool
        If True, make a fixed-orientation forward solution (Default:
        False). Note that fixed-orientation inverses can still be
        created from free-orientation forward solutions.
    grad : bool
        If True, compute the gradient of the field with respect to the
        dipole coordinates as well (Default: False).
    mricoord : bool
        If True, calculate in MRI coordinates (Default: False)
    %(overwrite)s
    %(subjects_dir)s
    %(verbose)s

    See Also
    --------
    make_forward_solution

    Returns
    -------
    fwd : Forward
        The generated forward solution.
    """
    if not has_mne_c():
        raise RuntimeError("mne command line tools could not be found")

    # check for file existence
    temp_dir = Path(tempfile.mkdtemp())
    if fname is None:
        fname = temp_dir / "temp-fwd.fif"
    _check_fname(fname, overwrite)
    _validate_type(subject, "str", "subject")

    # check for meas to exist as string, or try to make evoked
    _validate_type(meas, ("path-like", BaseRaw, BaseEpochs, Evoked), "meas")
    if isinstance(meas, BaseRaw | BaseEpochs | Evoked):
        meas_file = op.join(temp_dir, "info.fif")
        write_info(meas_file, meas.info)
        meas = meas_file
    else:
        meas = str(_check_fname(meas, overwrite="read", must_exist=True))

    # deal with trans/mri
    if mri is not None and trans is not None:
        raise ValueError("trans and mri cannot both be specified")
    if mri is None and trans is None:
        # MNE allows this to default to a trans/mri in the subject's dir,
        # but let's be safe here and force the user to pass us a trans/mri
        raise ValueError("Either trans or mri must be specified")

    if trans is not None:
        if isinstance(trans, dict):
            trans_data = deepcopy(trans)
            trans = temp_dir / "trans-trans.fif"
            try:
                write_trans(trans, trans_data)
            except Exception:
                raise OSError(
                    "trans was a dict, but could not be "
                    "written to disk as a transform file"
                )
        elif isinstance(trans, str | Path | PathLike):
            _check_fname(trans, "read", must_exist=True, name="trans")
            trans = Path(trans)
        else:
            raise ValueError("trans must be a path or dict")
    if mri is not None:
        if isinstance(mri, dict):
            mri_data = deepcopy(trans)
            mri = temp_dir / "mri-trans.fif"
            try:
                write_trans(mri, mri_data)
            except Exception:
                raise OSError(
                    "mri was a dict, but could not be "
                    "written to disk as a transform file"
                )
        elif isinstance(mri, str | Path | PathLike):
            _check_fname(mri, "read", must_exist=True, name="mri")
            mri = Path(mri)
        else:
            raise ValueError("mri must be a path or dict")

    # deal with meg/eeg
    if not meg and not eeg:
        raise ValueError("meg or eeg (or both) must be True")

    if not fname.suffix == ".fif":
        raise ValueError("Forward name does not end with .fif")
    path = fname.parent.absolute()
    fname = fname.name

    # deal with mindist
    if mindist is not None:
        if isinstance(mindist, str):
            if not mindist.lower() == "all":
                raise ValueError('mindist, if string, must be "all"')
            mindist = ["--all"]
        else:
            mindist = ["--mindist", f"{mindist:g}"]

    # src, spacing, bem
    for element, name, kind in zip(
        (src, spacing, bem),
        ("src", "spacing", "bem"),
        ("path-like", "str", "path-like"),
    ):
        if element is not None:
            _validate_type(element, kind, name, f"{kind} or None")

    # put together the actual call
    cmd = [
        "mne_do_forward_solution",
        "--subject",
        subject,
        "--meas",
        meas,
        "--fwd",
        fname,
        "--destdir",
        str(path),
    ]
    if src is not None:
        cmd += ["--src", src]
    if spacing is not None:
        if spacing.isdigit():
            pass  # spacing in mm
        else:
            # allow both "ico4" and "ico-4" style values
            match = re.match(r"(oct|ico)-?(\d+)$", spacing)
            if match is None:
                raise ValueError(f"Invalid spacing parameter: {spacing!r}")
            spacing = "-".join(match.groups())
        cmd += ["--spacing", spacing]
    if mindist is not None:
        cmd += mindist
    if bem is not None:
        cmd += ["--bem", bem]
    if mri is not None:
        cmd += ["--mri", f"{mri.absolute()}"]
    if trans is not None:
        cmd += ["--trans", f"{trans.absolute()}"]
    if not meg:
        cmd.append("--eegonly")
    if not eeg:
        cmd.append("--megonly")
    if fixed:
        cmd.append("--fixed")
    if grad:
        cmd.append("--grad")
    if mricoord:
        cmd.append("--mricoord")
    if overwrite:
        cmd.append("--overwrite")

    env = os.environ.copy()
    subjects_dir = str(get_subjects_dir(subjects_dir, raise_error=True))
    env["SUBJECTS_DIR"] = subjects_dir

    try:
        logger.info(
            "Running forward solution generation command with "
            f"subjects_dir {subjects_dir}"
        )
        run_subprocess(cmd, env=env)
    except Exception:
        raise
    else:
        fwd = read_forward_solution(path / fname, verbose=False)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return fwd


@verbose
def average_forward_solutions(fwds, weights=None, verbose=None):
    """Average forward solutions.

    Parameters
    ----------
    fwds : list of Forward
        Forward solutions to average. Each entry (dict) should be a
        forward solution.
    weights : array | None
        Weights to apply to each forward solution in averaging. If None,
        forward solutions will be equally weighted. Weights must be
        non-negative, and will be adjusted to sum to one.
    %(verbose)s

    Returns
    -------
    fwd : Forward
        The averaged forward solution.
    """
    # check for fwds being a list
    _validate_type(fwds, list, "fwds")
    if not len(fwds) > 0:
        raise ValueError("fwds must not be empty")

    # check weights
    if weights is None:
        weights = np.ones(len(fwds))
    weights = np.asanyarray(weights)  # in case it's a list, convert it
    if not np.all(weights >= 0):
        raise ValueError("weights must be non-negative")
    if not len(weights) == len(fwds):
        raise ValueError("weights must be None or the same length as fwds")
    w_sum = np.sum(weights)
    if not w_sum > 0:
        raise ValueError("weights cannot all be zero")
    weights /= w_sum

    # check our forward solutions
    for fwd in fwds:
        # check to make sure it's a forward solution
        _validate_type(fwd, dict, "each entry in fwds", "dict")
        # check to make sure the dict is actually a fwd
        check_keys = [
            "info",
            "sol_grad",
            "nchan",
            "src",
            "source_nn",
            "sol",
            "source_rr",
            "source_ori",
            "surf_ori",
            "coord_frame",
            "mri_head_t",
            "nsource",
        ]
        if not all(key in fwd for key in check_keys):
            raise KeyError(
                "forward solution dict does not have all standard "
                "entries, cannot compute average."
            )

    # check forward solution compatibility
    if any(
        fwd["sol"][k] != fwds[0]["sol"][k] for fwd in fwds[1:] for k in ["nrow", "ncol"]
    ):
        raise ValueError("Forward solutions have incompatible dimensions")
    if any(
        fwd[k] != fwds[0][k]
        for fwd in fwds[1:]
        for k in ["source_ori", "surf_ori", "coord_frame"]
    ):
        raise ValueError("Forward solutions have incompatible orientations")

    # actually average them (solutions and gradients)
    fwd_ave = deepcopy(fwds[0])
    fwd_ave["sol"]["data"] *= weights[0]
    fwd_ave["_orig_sol"] *= weights[0]
    for fwd, w in zip(fwds[1:], weights[1:]):
        fwd_ave["sol"]["data"] += w * fwd["sol"]["data"]
        fwd_ave["_orig_sol"] += w * fwd["_orig_sol"]
    if fwd_ave["sol_grad"] is not None:
        fwd_ave["sol_grad"]["data"] *= weights[0]
        fwd_ave["_orig_sol_grad"] *= weights[0]
        for fwd, w in zip(fwds[1:], weights[1:]):
            fwd_ave["sol_grad"]["data"] += w * fwd["sol_grad"]["data"]
            fwd_ave["_orig_sol_grad"] += w * fwd["_orig_sol_grad"]
    return fwd_ave
