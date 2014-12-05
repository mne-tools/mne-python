import os.path as op

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_true, assert_false)
import numpy as np
from numpy.testing import assert_array_equal

from mne import io, Epochs, read_events
from mne.io import read_fiducials, write_fiducials
from mne.io.constants import FIFF
from mne.io.meas_info import (Info, create_info, write_polhemus_hsp,
                              add_dig_points)
from mne.transforms import get_ras_to_neuromag_trans, apply_trans
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


def test_write_polhemus_hsp():
    """Test Writing for hsp files"""
    tempdir = _TempDir()
    points = np.loadtxt(hsp_fname, comments='%')

    dest = op.join(tempdir, 'test.txt')
    assert_raises(ValueError, write_polhemus_hsp, dest, points[:, :2])
    write_polhemus_hsp(dest, points)
    points1 = np.loadtxt(dest, comments='%')
    err = "Hsp points diverged after writing and reading."
    assert_array_equal(points, points1, err)


def test_add_dig_points():
    """Test application of Polhemus HSP to info"""
    dig_points = np.loadtxt(hsp_fname, comments='%')
    info = create_info(ch_names=['Test Ch'], sfreq=1000., ch_types=None)
    assert_false(info['dig'])

    add_dig_points(info, dig_points)
    assert_true(info['dig'])
    assert_array_equal(info['dig'][0]['r'], [-106.93, 99.80, 68.81])

    dig_points = np.loadtxt(elp_fname, comments='%')
    fids = dig_points[:3]
    info = create_info(ch_names=['Test Ch'], sfreq=1000., ch_types=None)
    assert_false(info['dig'])

    assert_raises(ValueError, add_dig_points, info, dig_points[0], ['nasion'])
    assert_raises(TypeError, add_dig_points, info, dig_points[0], 'nasion')
    add_dig_points(info, dig_points, ['nasion', 'lpa', 'rpa',
                                      '0', '1', '2', '3', '4'])
    assert_true(info['dig'])
    idx = [d['ident'] for d in info['dig']].index(FIFF.FIFFV_POINT_NASION)
    trans = get_ras_to_neuromag_trans(nasion=fids[0], lpa=fids[1], rpa=fids[2])
    assert_array_equal(info['dig'][idx]['r'],
                       apply_trans(trans, [1.3930, 13.1613, -4.6967]))
