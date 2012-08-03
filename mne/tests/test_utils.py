from scipy import signal
from numpy.testing import assert_equal

from mne.utils import _firwin2 as mne_firwin2


def test_firwin2():
    """Test firwin2 backport
    """
    taps1 = mne_firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    taps2 = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    assert_equal(taps1, taps2)
