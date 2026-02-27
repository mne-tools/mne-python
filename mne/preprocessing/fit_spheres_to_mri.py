import nibabel as nib
import numpy as np
import vedo
from scipy.special import KDTree

from .._fiff.constants import FIFF
from ..surface import _CheckInside
from ..transforms import (
    apply_trans,
    invert_transform,
    read_trans,
)


def fit_spheres_to_mri(subjects_dir, subject, bem, trans, n_spheres):
    mindist = 2e-3
    assert bem[0]["id"] == FIFF.FIFFV_BEM_SURF_ID_HEAD
    assert bem[2]["id"] == FIFF.FIFFV_BEM_SURF_ID_BRAIN
    scalp, _, inner_skull = bem
    inside_scalp = _CheckInside(scalp, mode="pyvista")
    inside_skull = _CheckInside(inner_skull, mode="pyvista")
    m3_to_cc = 100**3
    assert inside_scalp(inner_skull["rr"]).all()
    assert not inside_skull(scalp["rr"]).any()
    b = vedo.Mesh([inner_skull["rr"], inner_skull["tris"]])
    s = vedo.Mesh([scalp["rr"], scalp["tris"]])
    s_tree = KDTree(scalp["rr"])
    brain_volume = b.volume()
    print(f"Brain vedo:     {brain_volume * m3_to_cc:8.2f} cc")
    brain_vol = nib.load(subjects_dir / subject / "mri" / "brainmask.mgz")
    brain_rr = np.array(np.where(brain_vol.get_fdata())).T
    brain_rr = (
        apply_trans(brain_vol.header.get_vox2ras_tkr(), brain_rr) / 1000.0
    )  # apply a transformation matrix
    del brain_vol
    brain_rr = brain_rr[inside_skull(brain_rr)]
    vox_to_m3 = 1e-9
    brain_volume_vox = len(brain_rr) * vox_to_m3

    def _print_q(title, got, want):
        title = f"{title}:".ljust(15)
        print(f"{title} {got * m3_to_cc:8.2f} cc ({(want - got) / want * 100:6.2f} %)")

    _print_q("Brain vox", brain_volume_vox, brain_volume_vox)

    # 1. Compute a naive sphere using the center of mass of brain surf verts
    naive_c = np.mean(inner_skull["rr"], axis=0)

    # 2. Define optimization functions
    from scipy.optimize import fmin_cobyla

    def _cost(c):
        cs = c.reshape(-1, 3)
        rs = np.maximum(s_tree.query(cs)[0] - mindist, 0.0)
        resid = brain_volume
        mask = None
        for c, r in zip(cs, rs):
            if not (r and s.contains(c)):  # was is_inside
                continue
            m = np.linalg.norm(brain_rr - c, axis=1) <= r
            if mask is None:
                mask = m
            else:
                mask |= m
        resid = brain_volume_vox
        if mask is not None:
            resid = resid - np.sum(mask) * vox_to_m3
        return resid

    def _cons(c):
        cs = c.reshape(-1, 3)
        sign = np.array([2 * s.contains(c) - 1 for c in cs], float)  # was "is_inside"
        cons = sign * s_tree.query(cs)[0] - mindist
        return cons

    # 3. Now optimize spheres and find centers
    if n_spheres == 1:
        x = naive_c
        c_opt = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)

    elif n_spheres == 2:
        c_opt_1 = fmin_cobyla(_cost, naive_c, _cons, rhobeg=1e-2, rhoend=1e-4)
        x = np.concatenate([c_opt_1, naive_c])
        c_opt = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)

    elif n_spheres == 3:
        print("WARNING: mSSS method has been optimized with two origins")
        c_opt_1 = fmin_cobyla(_cost, naive_c, _cons, rhobeg=1e-2, rhoend=1e-4)
        x = np.concatenate([c_opt_1, naive_c])
        c_opt_2 = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
        x = np.concatenate([c_opt_2, naive_c])
        c_opt = fmin_cobyla(_cost, x, _cons, rhobeg=1e-2, rhoend=1e-4)
    else:
        raise ValueError("Implementation is for 3 or less origins")

    # 4. transform centers for return using "trans" matrix
    mri_head_t = invert_transform(read_trans(trans))
    assert mri_head_t["from"] == FIFF.FIFFV_COORD_MRI, mri_head_t["from"]
    centers = apply_trans(mri_head_t, c_opt.reshape(-1, 3))
    return centers
