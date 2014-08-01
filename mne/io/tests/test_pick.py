from numpy.testing import assert_array_equal
from mne import pick_channels_regexp


def test_pick_channels_regexp():
    """Test pick with regular expression
    """
    ch_names = ['MEG 2331', 'MEG 2332', 'MEG 2333']
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...1'), [0])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...[2-3]'), [1, 2])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG *'), [0, 1, 2])
