import os.path as op

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_equal

from mne import fiff, Epochs, read_events
from mne.fiff.meas_info import Info
from mne.utils import _TempDir

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

tempdir = _TempDir()


def test_fiducials_io():
    """Test fiducials i/o"""
    pts, coord_frame = fiff.read_fiducials(raw_fname)
    temp_fname = op.join(tempdir, 'test.fif')
    fiff.write_fiducials(temp_fname, pts, coord_frame)
    pts_1, coord_frame_1 = fiff.read_fiducials(temp_fname)
    assert_equal(coord_frame, coord_frame_1)
    for pt, pt_1 in zip(pts, pts_1):
        assert_equal(pt['kind'], pt_1['kind'])
        assert_equal(pt['ident'], pt_1['ident'])
        assert_equal(pt['coord_frame'], pt_1['coord_frame'])
        assert_array_equal(pt['r'], pt_1['r'])


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
