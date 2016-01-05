# -*- coding: utf-8 -*-

import os.path as op

from nose.tools import assert_false, assert_equal, assert_raises, assert_true
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from mne import Epochs, read_events
from mne.io import (read_fiducials, write_fiducials, _coil_trans_to_loc,
                    _loc_to_coil_trans, Raw, read_info, write_info)
from mne.io.constants import FIFF
from mne.io.meas_info import (Info, create_info, _write_dig_points,
                              _read_dig_points, _make_dig_points, _merge_info)
from mne.utils import _TempDir, run_tests_if_main
from mne.channels.montage import read_montage, read_dig_montage

base_dir = op.join(op.dirname(__file__), 'data')
fiducials_fname = op.join(base_dir, 'fsaverage-fiducials.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
chpi_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')
kit_data_dir = op.join(op.dirname(__file__), '..', 'kit', 'tests', 'data')
hsp_fname = op.join(kit_data_dir, 'test_hsp.txt')
elp_fname = op.join(kit_data_dir, 'test_elp.txt')


def test_coil_trans():
    """Test loc<->coil_trans functions"""
    rng = np.random.RandomState(0)
    x = rng.randn(4, 4)
    x[3] = [0, 0, 0, 1]
    assert_allclose(_loc_to_coil_trans(_coil_trans_to_loc(x)), x)
    x = rng.randn(12)
    assert_allclose(_coil_trans_to_loc(_loc_to_coil_trans(x)), x)


def test_make_info():
    """Test some create_info properties
    """
    n_ch = 1
    info = create_info(n_ch, 1000., 'eeg')
    coil_types = set([ch['coil_type'] for ch in info['chs']])
    assert_true(FIFF.FIFFV_COIL_EEG in coil_types)

    assert_raises(TypeError, create_info, ch_names='Test Ch', sfreq=1000)
    assert_raises(ValueError, create_info, ch_names=['Test Ch'], sfreq=-1000)
    assert_raises(ValueError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types=['eeg', 'eeg'])
    assert_raises(TypeError, create_info, ch_names=[np.array([1])],
                  sfreq=1000)
    assert_raises(TypeError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types=np.array([1]))
    assert_raises(KeyError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types='awesome')
    assert_raises(TypeError, create_info, ['Test Ch'], sfreq=1000,
                  ch_types=None, montage=np.array([1]))
    m = read_montage('biosemi32')
    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg',
                       montage=m)
    ch_pos = [ch['loc'][:3] for ch in info['chs']]
    assert_array_equal(ch_pos, m.pos)

    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    d = read_dig_montage(hsp_fname, None, elp_fname, names, unit='m',
                         transform=False)
    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg',
                       montage=d)
    idents = [p['ident'] for p in info['dig']]
    assert_true(FIFF.FIFFV_POINT_NASION in idents)

    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg',
                       montage=[d, m])
    ch_pos = [ch['loc'][:3] for ch in info['chs']]
    assert_array_equal(ch_pos, m.pos)
    idents = [p['ident'] for p in info['dig']]
    assert_true(FIFF.FIFFV_POINT_NASION in idents)
    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg',
                       montage=[d, 'biosemi32'])
    ch_pos = [ch['loc'][:3] for ch in info['chs']]
    assert_array_equal(ch_pos, m.pos)
    idents = [p['ident'] for p in info['dig']]
    assert_true(FIFF.FIFFV_POINT_NASION in idents)


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
    raw = Raw(raw_fname)
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
    info = read_info(raw_fname)
    temp_file = op.join(tempdir, 'info.fif')
    # check for bug `#1198`
    info['dev_head_t']['trans'] = np.eye(4)
    t1 = info['dev_head_t']['trans']
    write_info(temp_file, info)
    info2 = read_info(temp_file)
    t2 = info2['dev_head_t']['trans']
    assert_true(len(info['chs']) == len(info2['chs']))
    assert_array_equal(t1, t2)
    # proc_history (e.g., GH#1875)
    creator = u'é'
    info = read_info(chpi_fname)
    info['proc_history'][0]['creator'] = creator
    info['hpi_meas'][0]['creator'] = creator
    info['subject_info']['his_id'] = creator
    write_info(temp_file, info)
    info = read_info(temp_file)
    assert_equal(info['proc_history'][0]['creator'], creator)
    assert_equal(info['hpi_meas'][0]['creator'], creator)
    assert_equal(info['subject_info']['his_id'], creator)


def test_io_dig_points():
    """Test Writing for dig files"""
    tempdir = _TempDir()
    points = _read_dig_points(hsp_fname)

    dest = op.join(tempdir, 'test.txt')
    dest_bad = op.join(tempdir, 'test.mne')
    assert_raises(ValueError, _write_dig_points, dest, points[:, :2])
    assert_raises(ValueError, _write_dig_points, dest_bad, points)
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


def test_merge_info():
    """Test merging of multiple Info objects"""
    info_a = create_info(ch_names=['a', 'b', 'c'], sfreq=1000., ch_types=None)
    info_b = create_info(ch_names=['d', 'e', 'f'], sfreq=1000., ch_types=None)
    info_merged = _merge_info([info_a, info_b])
    assert_equal(info_merged['nchan'], 6)
    assert_equal(info_merged['ch_names'], ['a', 'b', 'c', 'd', 'e', 'f'])
    assert_raises(ValueError, _merge_info, [info_a, info_a])


run_tests_if_main()
