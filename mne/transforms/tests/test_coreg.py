import numpy as np
from numpy.testing import assert_array_almost_equal

from mne.transforms.coreg import fit_matched_pts
from mne.transforms.transforms import apply_trans, rotation, translation


def test_fit_matched_pts():
    """Test fitting two matching sets of points"""
    pts0 = np.random.normal(size=(5, 3))
    trans0 = np.dot(translation(2, 65, 3), rotation(2, 6, 3))
    pts1 = apply_trans(trans0, pts0)
    trans = fit_matched_pts(pts1, pts0)
    pts1t = apply_trans(trans, pts1)
    assert_array_almost_equal(pts0, pts1t)
