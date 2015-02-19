import os.path as op

from nose.tools import assert_false, assert_equal, assert_raises, assert_true
import numpy as np
from numpy.testing import assert_array_equal

from mne import io, Epochs, read_events
from mne.io import read_fiducials, write_fiducials
from mne.io.constants import FIFF
from mne.io.meas_info import (Info, create_info, _write_dig_points,
                              _read_dig_points, _make_dig_points)
from mne.utils import _TempDir
from mne.io.kit.tests import data_dir

base_dir = op.join(op.dirname(__file__), 'data')
fiducials_fname = op.join(base_dir, 'fsaverage-fiducials.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')
hsp_fname = op.join(data_dir, 'test_hsp.txt')
elp_fname = op.join(data_dir, 'test_elp.txt')


def test_fiducials_io():
    """Test fiducials i/o"""
    tempdir = _TempDir()
    pts, coord_frame = read_fiducials(fiducials_fname)
    assert_equal(pts[0]['coord_frame'], FIFF.FIFFV_COORD_MRI)
    assert_equal(pts[0]['ident'], FIFF.FIFFV_POINT_CARDINAL)

    temp_fname = op.join(tempdir, 'test.fif')
    write_fiducials(temp_fname, pts, coord_frame)
    pts_1, coord_frame_1 = read_fiducials(temp_fname)
    assert_equal(coord_frame, coord_frame_1)
    for pt, pt_1 in zip(pts, pts_1):
        assert_equal(pt['kind'], pt_1['kind'])
        assert_equal(pt['ident'], pt_1['ident'])
        assert_equal(pt['coord_frame'], pt_1['coord_frame'])
        assert_array_equal(pt['r'], pt_1['r'])

    # test safeguards
    pts[0]['coord_frame'] += 1
    assert_raises(ValueError, write_fiducials, temp_fname, pts, coord_frame)


def test_info():
    """Test info object"""
    raw = io.Raw(raw_fname)
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


def test_read_write_info():
    """Test IO of info
    """
    tempdir = _TempDir()
    info = io.read_info(raw_fname)
    temp_file = op.join(tempdir, 'info.fif')
    # check for bug `#1198`
    info['dev_head_t']['trans'] = np.eye(4)
    t1 = info['dev_head_t']['trans']
    io.write_info(temp_file, info)
    info2 = io.read_info(temp_file)
    t2 = info2['dev_head_t']['trans']
    assert_true(len(info['chs']) == len(info2['chs']))
    assert_array_equal(t1, t2)


def test_io_dig_points():
    """Test Writing for dig files"""
    tempdir = _TempDir()
    points = _read_dig_points(hsp_fname)

    dest = op.join(tempdir, 'test.txt')
    assert_raises(ValueError, _write_dig_points, dest, points[:, :2])
    _write_dig_points(dest, points)
    points1 = _read_dig_points(dest)
    err = "Dig points diverged after writing and reading."
    assert_array_equal(points, points1, err)

    points2 = np.array([[-106.93, 99.80], [99.80, 68.81]])
    np.savetxt(dest, points2, delimiter='\t', newline='\n')
    assert_raises(ValueError, _read_dig_points, dest)


def test_make_dig_points():
    """Test application of Polhemus HSP to info"""
    dig_points = _read_dig_points(hsp_fname)
    info = create_info(ch_names=['Test Ch'], sfreq=1000., ch_types=None)
    assert_false(info['dig'])

    info['dig'] = _make_dig_points(dig_points=dig_points)
    assert_true(info['dig'])
    assert_array_equal(info['dig'][0]['r'], [-106.93, 99.80, 68.81])

    dig_points = _read_dig_points(elp_fname)
    nasion, lpa, rpa = dig_points[:3]
    info = create_info(ch_names=['Test Ch'], sfreq=1000., ch_types=None)
    assert_false(info['dig'])

    info['dig'] = _make_dig_points(nasion, lpa, rpa, dig_points[3:], None)
    assert_true(info['dig'])
    idx = [d['ident'] for d in info['dig']].index(FIFF.FIFFV_POINT_NASION)
    assert_array_equal(info['dig'][idx]['r'],
                       np.array([1.3930, 13.1613, -4.6967]))
    assert_raises(ValueError, _make_dig_points, nasion[:2])
    assert_raises(ValueError, _make_dig_points, None, lpa[:2])
    assert_raises(ValueError, _make_dig_points, None, None, rpa[:2])
    assert_raises(ValueError, _make_dig_points, None, None, None,
                  dig_points[:, :2])
    assert_raises(ValueError, _make_dig_points, None, None, None, None,
                  dig_points[:, :2])
