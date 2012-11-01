from scipy import signal
from numpy.testing import assert_equal
from nose.tools import assert_true
import os.path as op

from mne.utils import _firwin2 as mne_firwin2
from mne import set_log_level, set_log_file
from mne.fiff import Evoked

fname_evoked = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                       'test-ave.fif')
fname_log = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                    'test-ave.log')
test_name = 'test.log'


def test_firwin2():
    """Test firwin2 backport
    """
    taps1 = mne_firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    taps2 = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    assert_equal(taps1, taps2)


def test_logging():
    """Test logging (to file)
    """
    set_log_file(test_name)
    set_log_level('WARNING')
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1)
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose=False)
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose='WARNING')
    assert_true(open(test_name).readlines() == [])
    set_log_level('INFO')
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose='WARNING')
    assert_true(open(test_name).readlines() == [])
    # should NOT print
    evoked = Evoked(fname_evoked, setno=1, verbose=False)
    assert_true(open(test_name).readlines() == [])
    # SHOULD print
    evoked = Evoked(fname_evoked, setno=1)
    set_log_file()
    new_log_file = open(test_name, 'r')
    old_log_file = open(fname_log, 'r')
    assert_equal(new_log_file.readlines(), old_log_file.readlines())
