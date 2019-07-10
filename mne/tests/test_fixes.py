# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD

import numpy as np
from scipy.signal import filtfilt

from numpy.testing import assert_array_equal

from mne.utils import run_tests_if_main
from mne.fixes import _sosfiltfilt as mne_sosfiltfilt


def test_filtfilt():
    """Test SOS filtfilt replacement."""
    x = np.r_[1, np.zeros(100)]
    # Filter with an impulse
    y = filtfilt([1, 0], [1, 0], x, padlen=0)
    assert_array_equal(x, y)
    y = mne_sosfiltfilt(np.array([[1., 0., 0., 1, 0., 0.]]), x, padlen=0)
    assert_array_equal(x, y)


run_tests_if_main()
