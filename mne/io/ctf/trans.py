"""Create coordinate transforms."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..._fiff.constants import FIFF
from ...transforms import (
    Transform,
    _fit_matched_points,
    _quat_to_affine,
    apply_trans,
    combine_transforms,
    get_ras_to_neuromag_trans,
    invert_transform,
)
from ...utils import logger
from .constants import CTF


def _make_transform_card(fro, to, r_lpa, r_nasion, r_rpa):
    """Make a transform from cardinal landmarks."""
    return invert_transform(
        Transform(to, fro, get_ras_to_neuromag_trans(r_nasion, r_lpa, r_rpa))
    )


def _quaternion_align(from_frame, to_frame, from_pts, to_pts, diff_tol=1e-4):
    """Perform an alignment using the unit quaternions (modifies points)."""
    assert from_pts.shape[1] == to_pts.shape[1] == 3
    trans = _quat_to_affine(_fit_matched_points(from_pts, to_pts)[0])

    # Test the transformation and print the results
    logger.info("    Quaternion matching (desired vs. transformed):")
    for fro, to in zip(from_pts, to_pts):
        rr = apply_trans(trans, fro)
        diff = np.linalg.norm(to - rr)
        logger.info(
            "    %7.2f %7.2f %7.2f mm <-> %7.2f %7.2f %7.2f mm "
            "(orig : %7.2f %7.2f %7.2f mm) diff = %8.3f mm"
            % (tuple(1000 * to) + tuple(1000 * rr) + tuple(1000 * fro) + (1000 * diff,))
        )
        if diff > diff_tol:
            raise RuntimeError(
                "Something is wrong: quaternion matching did not work (see above)"
            )
    return Transform(from_frame, to_frame, trans)


def _make_ctf_coord_trans_set(res4, coils):
    """Figure out the necessary coordinate transforms."""
    # CTF head > Neuromag head
    lpa = rpa = nas = T1 = T2 = T3 = T5 = None
    if coils is not None:
        for p in coils:
            if p["valid"] and (p["coord_frame"] == FIFF.FIFFV_MNE_COORD_CTF_HEAD):
                if lpa is None and p["kind"] == CTF.CTFV_COIL_LPA:
                    lpa = p
                elif rpa is None and p["kind"] == CTF.CTFV_COIL_RPA:
                    rpa = p
                elif nas is None and p["kind"] == CTF.CTFV_COIL_NAS:
                    nas = p
        if lpa is None or rpa is None or nas is None:
            raise RuntimeError(
                "Some of the mandatory HPI device-coordinate info was not there."
            )
        t = _make_transform_card("head", "ctf_head", lpa["r"], nas["r"], rpa["r"])
        T3 = invert_transform(t)

    # CTF device -> Neuromag device
    #
    # Rotate the CTF coordinate frame by 45 degrees and shift by 190 mm
    # in z direction to get a coordinate system comparable to the Neuromag one
    #
    R = np.eye(4)
    R[:3, 3] = [0.0, 0.0, 0.19]
    val = 0.5 * np.sqrt(2.0)
    R[0, 0] = val
    R[0, 1] = -val
    R[1, 0] = val
    R[1, 1] = val
    T4 = Transform("ctf_meg", "meg", R)

    # CTF device -> CTF head
    # We need to make the implicit transform explicit!
    h_pts = dict()
    d_pts = dict()
    kinds = (
        CTF.CTFV_COIL_LPA,
        CTF.CTFV_COIL_RPA,
        CTF.CTFV_COIL_NAS,
        CTF.CTFV_COIL_SPARE,
    )
    if coils is not None:
        for p in coils:
            if p["valid"]:
                if p["coord_frame"] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
                    for kind in kinds:
                        if kind not in h_pts and p["kind"] == kind:
                            h_pts[kind] = p["r"]
                elif p["coord_frame"] == FIFF.FIFFV_MNE_COORD_CTF_DEVICE:
                    for kind in kinds:
                        if kind not in d_pts and p["kind"] == kind:
                            d_pts[kind] = p["r"]
        if any(kind not in h_pts for kind in kinds[:-1]):
            raise RuntimeError(
                "Some of the mandatory HPI device-coordinate info was not there."
            )
        if any(kind not in d_pts for kind in kinds[:-1]):
            raise RuntimeError(
                "Some of the mandatory HPI head-coordinate info was not there."
            )
        use_kinds = [kind for kind in kinds if (kind in h_pts and kind in d_pts)]
        r_head = np.array([h_pts[kind] for kind in use_kinds])
        r_dev = np.array([d_pts[kind] for kind in use_kinds])
        T2 = _quaternion_align("ctf_meg", "ctf_head", r_dev, r_head)

    # The final missing transform
    if T3 is not None and T2 is not None:
        T5 = combine_transforms(T2, T3, "ctf_meg", "head")
        T1 = combine_transforms(invert_transform(T4), T5, "meg", "head")
    s = dict(
        t_dev_head=T1,
        t_ctf_dev_ctf_head=T2,
        t_ctf_head_head=T3,
        t_ctf_dev_dev=T4,
        t_ctf_dev_head=T5,
    )
    logger.info("    Coordinate transformations established.")
    return s
