from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal

from mne.transforms.coreg import fit_matched_pts
from mne.transforms.transforms import apply_trans, rotation, translation


def test_fit_matched_pts():
    """Test fitting two matching sets of points"""
    src_pts = np.random.normal(size=(5, 3))
    trans0 = np.dot(translation(2, 65, 3), rotation(2, 6, 3))
    tgt_pts = apply_trans(trans0, src_pts)
    trans = fit_matched_pts(tgt_pts, src_pts)
    est_pts = apply_trans(trans, tgt_pts)
    assert_array_almost_equal(src_pts, est_pts)

    # test exceeding tolerance
    src_pts[0, :] += 20
    assert_raises(RuntimeError, fit_matched_pts, src_pts, tgt_pts, tol=10)
