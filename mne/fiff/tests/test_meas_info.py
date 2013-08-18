import os.path as op

from nose.tools import assert_true, assert_equal

from mne import fiff, Epochs, read_events
from mne.fiff.meas_info import Info

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')


def test_info():
    """Test info object"""
    raw = fiff.Raw(raw_fname)
    event_id, tmin, tmax = 1, -0.2, 0.5
    events = read_events(event_name)
    event_id = int(events[0, 2])
    epochs = Epochs(raw, events[:1], event_id, tmin, tmax, picks=None,
                    baseline=(None, 0))

    evoked = epochs.average()

    events = read_events(event_name)

    # Test subclassing was successful.
    info = Info(a=7, b='aaaaa')
    assert_true('a' in info)
    assert_true('b' in info)
    info[42] = 'foo'
    assert_true(info[42] == 'foo')

    # test info attribute in API objects
    for obj in [raw, epochs, evoked]:
        assert_true(isinstance(obj.info, Info))
        info_str = '%s' % obj.info
        assert_equal(len(info_str.split('\n')), (len(obj.info.keys()) + 2))
        assert_true(all(k in info_str for k in obj.info.keys()))
