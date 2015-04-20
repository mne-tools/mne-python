# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal

from mne.preprocessing.maxfilter import fit_sphere_to_headshape
from mne.io.constants import FIFF
from mne.transforms import rotation


def test_fit_sphere_to_headshape():
    """ Test fitting a sphere to digitization points. """
    # Create points of various kinds
    dig = [
        # Left auricular
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'ident': FIFF.FIFFV_POINT_LPA,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([-1.0, 0.0, 0.0])},
        # Nasion
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'ident': FIFF.FIFFV_POINT_NASION,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([0.0, 1.0, 0.0])},
        # Right auricular
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'ident': FIFF.FIFFV_POINT_RPA,
         'kind': FIFF.FIFFV_POINT_CARDINAL,
         'r': np.array([1.0, 0.0, 0.0])},

        # Top of the head (extra point)
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EXTRA,
         'r': np.array([0.0, 0.0, 1.0])},

        # EEG points
        # Fz
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0, .72, .69])},
        # F3
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([-.55, .67, .50])},
        # F4
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([.55, .67, .50])},
        # Cz
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0.0, 0.0, 1.0])},
        # Pz
        {'coord_frame': FIFF.FIFFV_COORD_DEVICE,
         'kind': FIFF.FIFFV_POINT_EEG,
         'r': np.array([0, -.72, .69])},
    ]

    # Device to head transformation (rotate .2 rad over X-axis)
    dev_head_t = {
        'from': FIFF.FIFFV_COORD_DEVICE,
        'to': FIFF.FIFFV_COORD_HEAD,
        'trans': rotation(x=0.2),
    }

    info = {'dig': dig, 'dev_head_t': dev_head_t}

    #  # Test with 4 points that match a perfect sphere
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_almost_equal(r / 1000., 1.0, decimal=10)
    assert_almost_equal(oh / 1000., [0.0, 0.0, 0.0], decimal=10)
    assert_almost_equal(od / 1000., [0.0, 0.0, 0.0], decimal=10)

    # Test with all points. Digitization points are no longer perfect, so
    # allow for a wider margin of error.
    dig_kinds = (FIFF.FIFFV_POINT_CARDINAL, FIFF.FIFFV_POINT_EXTRA,
                 FIFF.FIFFV_POINT_EXTRA)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_almost_equal(r / 1000., 1.0, decimal=3)
    assert_almost_equal(oh / 1000., [0.0, 0.0, 0.0], decimal=3)
    assert_almost_equal(od / 1000., [0.0, 0.0, 0.0], decimal=3)

    # Test with some noisy EEG points only.
    dig_kinds = (FIFF.FIFFV_POINT_EEG,)
    r, oh, od = fit_sphere_to_headshape(info, dig_kinds=dig_kinds)
    assert_almost_equal(r / 1000., 1.0, decimal=2)
    assert_almost_equal(oh / 1000., [0.0, 0.0, 0.0], decimal=2)
    assert_almost_equal(od / 1000., [0.0, 0.0, 0.0], decimal=2)
