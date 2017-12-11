# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD

import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import filtfilt
from scipy.special import sph_harm

from numpy.testing import assert_array_equal

from mne.utils import run_tests_if_main, requires_version
from mne.fixes import _sosfiltfilt as mne_sosfiltfilt, _sph_harm


def test_filtfilt():
    """Test SOS filtfilt replacement"""
    x = np.r_[1, np.zeros(100)]
    # Filter with an impulse
    y = filtfilt([1, 0], [1, 0], x, padlen=0)
    assert_array_equal(x, y)
    y = mne_sosfiltfilt(np.array([[1., 0., 0., 1, 0., 0.]]), x, padlen=0)
    assert_array_equal(x, y)


@requires_version('scipy', '0.17.1')
def test_spherical_harmonics():
    """Test spherical harmonic functions."""
    az, pol = np.meshgrid(np.linspace(0, 2 * np.pi, 30),
                          np.linspace(0, np.pi, 20))
    # Test our basic spherical harmonics
    for degree in range(1, 10):
        for order in range(0, degree + 1):
            sph = _sph_harm(order, degree, az, pol)
            sph_scipy = sph_harm(order, degree, az, pol)
            assert_allclose(sph, sph_scipy, atol=1e-7)

run_tests_if_main()
