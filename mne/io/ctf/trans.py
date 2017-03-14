"""Create coordinate transforms."""

# Author: Eric Larson <larson.eric.d<gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ...transforms import combine_transforms, invert_transform, Transform
from ...utils import logger
from ..constants import FIFF
from .constants import CTF


def _make_transform_card(fro, to, r_lpa, r_nasion, r_rpa):
    """Make a transform from cardinal landmarks."""
    # XXX de-duplicate this with code from Montage somewhere?
    diff_1 = r_nasion - r_lpa
    ex = r_rpa - r_lpa
    alpha = np.dot(diff_1, ex) / np.dot(ex, ex)
    ex /= np.sqrt(np.sum(ex * ex))
    trans = np.eye(4)
    move = (1. - alpha) * r_lpa + alpha * r_rpa
    trans[:3, 3] = move
    trans[:3, 0] = ex
    ey = r_nasion - move
    ey /= np.sqrt(np.sum(ey * ey))
    trans[:3, 1] = ey
    trans[:3, 2] = np.cross(ex, ey)  # ez
    return Transform(fro, to, trans)


def _quaternion_align(from_frame, to_frame, from_pts, to_pts):
    """Perform an alignment using the unit quaternions (modifies points)."""
    assert from_pts.shape[1] == to_pts.shape[1] == 3

    # Calculate the centroids and subtract
    from_c, to_c = from_pts.mean(axis=0), to_pts.mean(axis=0)
    from_ = from_pts - from_c
    to_ = to_pts - to_c

    # Compute the dot products
    S = np.dot(from_.T, to_)

    # Compute the magical N matrix
    N = np.array([[S[0, 0] + S[1, 1] + S[2, 2], 0., 0., 0.],
                  [S[1, 2] - S[2, 1], S[0, 0] - S[1, 1] - S[2, 2], 0., 0.],
                  [S[2, 0] - S[0, 2], S[0, 1] + S[1, 0],
                   -S[0, 0] + S[1, 1] - S[2, 2], 0.],
                  [S[0, 1] - S[1, 0], S[2, 0] + S[0, 2],
                   S[1, 2] + S[2, 1], -S[0, 0] - S[1, 1] + S[2, 2]]])

    # Compute the eigenvalues and eigenvectors
    # Use the eigenvector corresponding to the largest eigenvalue as the
    # unit quaternion defining the rotation
    eig_vals, eig_vecs = linalg.eigh(N, overwrite_a=True)
    which = np.argmax(eig_vals)
    if eig_vals[which] < 0:
        raise RuntimeError('No positive eigenvalues. Cannot do the alignment.')
    q = eig_vecs[:, which]

    # Write out the rotation
    trans = np.eye(4)
    trans[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    trans[0, 1] = 2.0 * (q[1] * q[2] - q[0] * q[3])
    trans[0, 2] = 2.0 * (q[1] * q[3] + q[0] * q[2])
    trans[1, 0] = 2.0 * (q[2] * q[1] + q[0] * q[3])
    trans[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    trans[1, 2] = 2.0 * (q[2] * q[3] - q[0] * q[1])
    trans[2, 0] = 2.0 * (q[3] * q[1] - q[0] * q[2])
    trans[2, 1] = 2.0 * (q[3] * q[2] + q[0] * q[1])
    trans[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]

    # Now we need to generate a transformed translation vector
    trans[:3, 3] = to_c - np.dot(trans[:3, :3], from_c)
    del to_c, from_c

    # Test the transformation and print the results
    logger.info('    Quaternion matching (desired vs. transformed):')
    for fro, to in zip(from_pts, to_pts):
        rr = np.dot(trans[:3, :3], fro) + trans[:3, 3]
        diff = np.sqrt(np.sum((to - rr) ** 2))
        logger.info('    %7.2f %7.2f %7.2f mm <-> %7.2f %7.2f %7.2f mm '
                    '(orig : %7.2f %7.2f %7.2f mm) diff = %8.3f mm'
                    % (tuple(1000 * to) + tuple(1000 * rr) +
                       tuple(1000 * fro) + (1000 * diff,)))
        if diff > 1e-4:
            raise RuntimeError('Something is wrong: quaternion matching did '
                               'not work (see above)')
    return Transform(from_frame, to_frame, trans)


def _make_ctf_coord_trans_set(res4, coils):
    """Figure out the necessary coordinate transforms."""
    # CTF head > Neuromag head
    lpa = rpa = nas = T1 = T2 = T3 = T5 = None
    if coils is not None:
        for p in coils:
            if p['valid'] and (p['coord_frame'] ==
                               FIFF.FIFFV_MNE_COORD_CTF_HEAD):
                if lpa is None and p['kind'] == CTF.CTFV_COIL_LPA:
                    lpa = p
                elif rpa is None and p['kind'] == CTF.CTFV_COIL_RPA:
                    rpa = p
                elif nas is None and p['kind'] == CTF.CTFV_COIL_NAS:
                    nas = p
        if lpa is None or rpa is None or nas is None:
            raise RuntimeError('Some of the mandatory HPI device-coordinate '
                               'info was not there.')
        t = _make_transform_card(FIFF.FIFFV_COORD_HEAD,
                                 FIFF.FIFFV_MNE_COORD_CTF_HEAD,
                                 lpa['r'], nas['r'], rpa['r'])
        T3 = invert_transform(t)

    # CTF device -> Neuromag device
    #
    # Rotate the CTF coordinate frame by 45 degrees and shift by 190 mm
    # in z direction to get a coordinate system comparable to the Neuromag one
    #
    R = np.eye(4)
    R[:3, 3] = [0., 0., 0.19]
    val = 0.5 * np.sqrt(2.)
    R[0, 0] = val
    R[0, 1] = -val
    R[1, 0] = val
    R[1, 1] = val
    T4 = Transform(FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
                   FIFF.FIFFV_COORD_DEVICE, R)

    # CTF device -> CTF head
    # We need to make the implicit transform explicit!
    h_pts = dict()
    d_pts = dict()
    kinds = (CTF.CTFV_COIL_LPA, CTF.CTFV_COIL_RPA, CTF.CTFV_COIL_NAS,
             CTF.CTFV_COIL_SPARE)
    if coils is not None:
        for p in coils:
            if p['valid']:
                if p['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_HEAD:
                    for kind in kinds:
                        if kind not in h_pts and p['kind'] == kind:
                            h_pts[kind] = p['r']
                elif p['coord_frame'] == FIFF.FIFFV_MNE_COORD_CTF_DEVICE:
                    for kind in kinds:
                        if kind not in d_pts and p['kind'] == kind:
                            d_pts[kind] = p['r']
        if any(kind not in h_pts for kind in kinds[:-1]):
            raise RuntimeError('Some of the mandatory HPI device-coordinate '
                               'info was not there.')
        if any(kind not in d_pts for kind in kinds[:-1]):
            raise RuntimeError('Some of the mandatory HPI head-coordinate '
                               'info was not there.')
        use_kinds = [kind for kind in kinds
                     if (kind in h_pts and kind in d_pts)]
        r_head = np.array([h_pts[kind] for kind in use_kinds])
        r_dev = np.array([d_pts[kind] for kind in use_kinds])
        T2 = _quaternion_align(FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
                               FIFF.FIFFV_MNE_COORD_CTF_HEAD, r_dev, r_head)

    # The final missing transform
    if T3 is not None and T2 is not None:
        T5 = combine_transforms(T2, T3, FIFF.FIFFV_MNE_COORD_CTF_DEVICE,
                                FIFF.FIFFV_COORD_HEAD)
        T1 = combine_transforms(invert_transform(T4), T5,
                                FIFF.FIFFV_COORD_DEVICE, FIFF.FIFFV_COORD_HEAD)
    s = dict(t_dev_head=T1, t_ctf_dev_ctf_head=T2, t_ctf_head_head=T3,
             t_ctf_dev_dev=T4, t_ctf_dev_head=T5)
    logger.info('    Coordinate transformations established.')
    return s
