"""Coregistration between different coordinate frames"""

# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

from numpy import dot
from scipy.optimize import leastsq

from .transforms import apply_trans, rotation, translation


def fit_matched_pts(src_pts, tgt_pts, params=False):
    """Find a transform that minimizes the squared distance between two
    matching sets of points.

    Uses :func:`scipy.optimize.leastsq` to find a transformation involving
    rotation and translation.

    Parameters
    ----------
    src_pts : array, shape = (n, 3)
        Points to which the transform should be applied.
    tgt_pts : array, shape = (n, 3)
        Points to which src_pts should be fitted. Each point in tgt_pts should
        correspond to the point in src_pts with the same index.
    params : bool
        Also return the estimated rotation and translation parameters.

    Returns
    -------
    trans : array, shape = (4, 4)
        Transformation that, if applied to src_pts, minimizes the squared
        distance to tgt_pts.
    [rotation : array, len = 3, optional]
        The rotation parameters around the x, y, and z axes (in radians).
    [translation : array, len = 3, optional]
        The translation parameters in x, y, and z direction.

    """
    print src_pts, tgt_pts
    def error(params):
        trans = dot(translation(*params[:3]), rotation(*params[3:]))
        est = apply_trans(trans, src_pts)
        return (tgt_pts - est).ravel()

    x0 = (0, 0, 0, 0, 0, 0)
    x, _ = leastsq(error, x0)

    transl = x[:3]
    rot = x[3:]
    trans = dot(translation(*transl), rotation(*rot))
    if params:
        return trans, rot, transl
    else:
        return trans
