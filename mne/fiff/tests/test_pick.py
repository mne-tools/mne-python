from numpy.testing import assert_array_equal
from ..pick import pick_channels_regexp, pick_selection


def test_pick_channels_regexp():
    """Test pick with regular expression"""
    ch_names = ['MEG 2331', 'MEG 2332', 'MEG 2333']
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...1'), [0])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...[2-3]'), [1, 2])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG *'), [0, 1, 2])


def test_pick_selection():
    """Test pick using named selection"""
    # test one channel for each selection
    ch_names = ['MEG 2211', 'MEG 0223', 'MEG 1312', 'MEG 0412', 'MEG 1043',
                'MEG 2042', 'MEG 2032', 'MEG 0522', 'MEG 1031']
    sel_names = ['Vertex', 'Left-temporal', 'Right-temporal', 'Left-parietal',
                 'Right-parietal', 'Left-occipital', 'Right-occipital',
                 'Left-frontal', 'Right-frontal']

    for i, sel in enumerate(sel_names):
        picks = pick_selection(ch_names, sel)
        assert(i in picks)
