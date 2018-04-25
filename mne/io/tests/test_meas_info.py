# -*- coding: utf-8 -*-
import hashlib
import os.path as op
import warnings

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy import sparse

from mne import Epochs, read_events, pick_info, pick_types
from mne.event import make_fixed_length_events
from mne.datasets import testing
from mne.io import (read_fiducials, write_fiducials, _coil_trans_to_loc,
                    _loc_to_coil_trans, read_raw_fif, read_info, write_info,
                    anonymize_info)
from mne.io.constants import FIFF
from mne.io.write import DATE_NONE
from mne.io.meas_info import (Info, create_info, _write_dig_points,
                              _read_dig_points, _make_dig_points, _merge_info,
                              _force_update_info, RAW_INFO_FIELDS,
                              _bad_chans_comp)
from mne.io import read_raw_ctf
from mne.utils import _TempDir, run_tests_if_main
from mne.channels.montage import read_montage, read_dig_montage

warnings.simplefilter("always")  # ensure we can verify expected warnings

base_dir = op.join(op.dirname(__file__), 'data')
fiducials_fname = op.join(base_dir, 'fsaverage-fiducials.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
chpi_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
event_name = op.join(base_dir, 'test-eve.fif')
kit_data_dir = op.join(op.dirname(__file__), '..', 'kit', 'tests', 'data')
hsp_fname = op.join(kit_data_dir, 'test_hsp.txt')
elp_fname = op.join(kit_data_dir, 'test_elp.txt')

data_path = testing.data_path(download=False)
sss_path = op.join(data_path, 'SSS')
pre = op.join(sss_path, 'test_move_anon_')
sss_ctc_fname = pre + 'crossTalk_raw_sss.fif'
ctf_fname = op.join(data_path, 'CTF', 'testdata_ctf.ds')


def test_coil_trans():
    """Test loc<->coil_trans functions."""
    rng = np.random.RandomState(0)
    x = rng.randn(4, 4)
    x[3] = [0, 0, 0, 1]
    assert_allclose(_loc_to_coil_trans(_coil_trans_to_loc(x)), x)
    x = rng.randn(12)
    assert_allclose(_coil_trans_to_loc(_loc_to_coil_trans(x)), x)


def test_make_info():
    """Test some create_info properties."""
    n_ch = 1
    info = create_info(n_ch, 1000., 'eeg')
    assert set(info.keys()) == set(RAW_INFO_FIELDS)

    coil_types = set([ch['coil_type'] for ch in info['chs']])
    assert FIFF.FIFFV_COIL_EEG in coil_types

    pytest.raises(TypeError, create_info, ch_names='Test Ch', sfreq=1000)
    pytest.raises(ValueError, create_info, ch_names=['Test Ch'], sfreq=-1000)
    pytest.raises(ValueError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types=['eeg', 'eeg'])
    pytest.raises(TypeError, create_info, ch_names=[np.array([1])],
                  sfreq=1000)
    pytest.raises(TypeError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types=np.array([1]))
    pytest.raises(KeyError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types='awesome')
    pytest.raises(TypeError, create_info, ['Test Ch'], sfreq=1000,
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
    assert FIFF.FIFFV_POINT_NASION in idents

    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg',
                       montage=[d, m])
    ch_pos = [ch['loc'][:3] for ch in info['chs']]
    assert_array_equal(ch_pos, m.pos)
    idents = [p['ident'] for p in info['dig']]
    assert (FIFF.FIFFV_POINT_NASION in idents)
    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg',
                       montage=[d, 'biosemi32'])
    ch_pos = [ch['loc'][:3] for ch in info['chs']]
    assert_array_equal(ch_pos, m.pos)
    idents = [p['ident'] for p in info['dig']]
    assert (FIFF.FIFFV_POINT_NASION in idents)
    assert info['meas_date'] is None


def test_fiducials_io():
    """Test fiducials i/o."""
    tempdir = _TempDir()
    pts, coord_frame = read_fiducials(fiducials_fname)
    assert pts[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI
    assert pts[0]['ident'] == FIFF.FIFFV_POINT_CARDINAL

    temp_fname = op.join(tempdir, 'test.fif')
    write_fiducials(temp_fname, pts, coord_frame)
    pts_1, coord_frame_1 = read_fiducials(temp_fname)
    assert coord_frame == coord_frame_1
    for pt, pt_1 in zip(pts, pts_1):
        assert pt['kind'] == pt_1['kind']
        assert pt['ident'] == pt_1['ident']
        assert pt['coord_frame'] == pt_1['coord_frame']
        assert_array_equal(pt['r'], pt_1['r'])

    # test safeguards
    pts[0]['coord_frame'] += 1
    pytest.raises(ValueError, write_fiducials, temp_fname, pts, coord_frame)


def test_info():
    """Test info object."""
    raw = read_raw_fif(raw_fname)
    event_id, tmin, tmax = 1, -0.2, 0.5
    events = read_events(event_name)
    event_id = int(events[0, 2])
    epochs = Epochs(raw, events[:1], event_id, tmin, tmax, picks=None,
                    baseline=(None, 0))

    evoked = epochs.average()

    # Test subclassing was successful.
    info = Info(a=7, b='aaaaa')
    assert ('a' in info)
    assert ('b' in info)
    info[42] = 'foo'
    assert (info[42] == 'foo')

    # Test info attribute in API objects
    for obj in [raw, epochs, evoked]:
        assert (isinstance(obj.info, Info))
        info_str = '%s' % obj.info
        assert len(info_str.split('\n')) == len(obj.info.keys()) + 2
        assert all(k in info_str for k in obj.info.keys())
        assert '2002-12-03 19:01:10 GMT' in repr(obj.info), repr(obj.info)

    # Test read-only fields
    info = raw.info.copy()
    nchan = len(info['chs'])
    ch_names = [ch['ch_name'] for ch in info['chs']]
    assert info['nchan'] == nchan
    assert list(info['ch_names']) == ch_names

    # Deleting of regular fields should work
    info['foo'] = 'bar'
    del info['foo']

    # Test updating of fields
    del info['chs'][-1]
    info._update_redundant()
    assert info['nchan'] == nchan - 1
    assert list(info['ch_names']) == ch_names[:-1]

    info['chs'][0]['ch_name'] = 'foo'
    info._update_redundant()
    assert info['ch_names'][0] == 'foo'

    # Test casting to and from a dict
    info_dict = dict(info)
    info2 = Info(info_dict)
    assert info == info2


def test_read_write_info():
    """Test IO of info."""
    tempdir = _TempDir()
    info = read_info(raw_fname)
    temp_file = op.join(tempdir, 'info.fif')
    # check for bug `#1198`
    info['dev_head_t']['trans'] = np.eye(4)
    t1 = info['dev_head_t']['trans']
    write_info(temp_file, info)
    info2 = read_info(temp_file)
    t2 = info2['dev_head_t']['trans']
    assert (len(info['chs']) == len(info2['chs']))
    assert_array_equal(t1, t2)
    # proc_history (e.g., GH#1875)
    creator = u'é'
    info = read_info(chpi_fname)
    info['proc_history'][0]['creator'] = creator
    info['hpi_meas'][0]['creator'] = creator
    info['subject_info']['his_id'] = creator
    info['subject_info']['weight'] = 11.1
    info['subject_info']['height'] = 2.3

    if info['gantry_angle'] is None:  # future testing data may include it
        info['gantry_angle'] = 0.  # Elekta supine position
    gantry_angle = info['gantry_angle']

    write_info(temp_file, info)
    info = read_info(temp_file)
    assert info['proc_history'][0]['creator'] == creator
    assert info['hpi_meas'][0]['creator'] == creator
    assert info['subject_info']['his_id'] == creator
    assert info['gantry_angle'] == gantry_angle
    assert info['subject_info']['height'] == 2.3
    assert info['subject_info']['weight'] == 11.1

    # Test that writing twice produces the same file
    m1 = hashlib.md5()
    with open(temp_file, 'rb') as fid:
        m1.update(fid.read())
    m1 = m1.hexdigest()
    temp_file_2 = op.join(tempdir, 'info2.fif')
    assert temp_file_2 != temp_file
    write_info(temp_file_2, info)
    m2 = hashlib.md5()
    with open(temp_file_2, 'rb') as fid:
        m2.update(fid.read())
    m2 = m2.hexdigest()
    assert m1 == m2


def test_io_dig_points():
    """Test Writing for dig files."""
    tempdir = _TempDir()
    points = _read_dig_points(hsp_fname)

    dest = op.join(tempdir, 'test.txt')
    dest_bad = op.join(tempdir, 'test.mne')
    pytest.raises(ValueError, _write_dig_points, dest, points[:, :2])
    pytest.raises(ValueError, _write_dig_points, dest_bad, points)
    _write_dig_points(dest, points)
    points1 = _read_dig_points(dest, unit='m')
    err = "Dig points diverged after writing and reading."
    assert_array_equal(points, points1, err)

    points2 = np.array([[-106.93, 99.80], [99.80, 68.81]])
    np.savetxt(dest, points2, delimiter='\t', newline='\n')
    pytest.raises(ValueError, _read_dig_points, dest)


def test_make_dig_points():
    """Test application of Polhemus HSP to info."""
    extra_points = _read_dig_points(hsp_fname)
    info = create_info(ch_names=['Test Ch'], sfreq=1000., ch_types=None)
    assert info['dig'] is None

    info['dig'] = _make_dig_points(extra_points=extra_points)
    assert (info['dig'])
    assert_allclose(info['dig'][0]['r'], [-.10693, .09980, .06881])

    elp_points = _read_dig_points(elp_fname)
    nasion, lpa, rpa = elp_points[:3]
    info = create_info(ch_names=['Test Ch'], sfreq=1000., ch_types=None)
    assert info['dig'] is None

    info['dig'] = _make_dig_points(nasion, lpa, rpa, elp_points[3:], None)
    assert (info['dig'])
    idx = [d['ident'] for d in info['dig']].index(FIFF.FIFFV_POINT_NASION)
    assert_array_equal(info['dig'][idx]['r'],
                       np.array([.0013930, .0131613, -.0046967]))
    pytest.raises(ValueError, _make_dig_points, nasion[:2])
    pytest.raises(ValueError, _make_dig_points, None, lpa[:2])
    pytest.raises(ValueError, _make_dig_points, None, None, rpa[:2])
    pytest.raises(ValueError, _make_dig_points, None, None, None,
                  elp_points[:, :2])
    pytest.raises(ValueError, _make_dig_points, None, None, None, None,
                  elp_points[:, :2])


def test_redundant():
    """Test some of the redundant properties of info."""
    # Indexing
    info = create_info(ch_names=['a', 'b', 'c'], sfreq=1000., ch_types=None)
    assert info['ch_names'][0] == 'a'
    assert info['ch_names'][1] == 'b'
    assert info['ch_names'][2] == 'c'

    # Equality
    assert info['ch_names'] == info['ch_names']
    assert info['ch_names'] == ['a', 'b', 'c']

    # No channels in info
    info = create_info(ch_names=[], sfreq=1000., ch_types=None)
    assert info['ch_names'] == []

    # List should be read-only
    info = create_info(ch_names=['a', 'b', 'c'], sfreq=1000., ch_types=None)


def test_merge_info():
    """Test merging of multiple Info objects."""
    info_a = create_info(ch_names=['a', 'b', 'c'], sfreq=1000., ch_types=None)
    info_b = create_info(ch_names=['d', 'e', 'f'], sfreq=1000., ch_types=None)
    info_merged = _merge_info([info_a, info_b])
    assert info_merged['nchan'], 6
    assert info_merged['ch_names'], ['a', 'b', 'c', 'd', 'e', 'f']
    pytest.raises(ValueError, _merge_info, [info_a, info_a])

    # Testing for force updates before merging
    info_c = create_info(ch_names=['g', 'h', 'i'], sfreq=500., ch_types=None)
    # This will break because sfreq is not equal
    pytest.raises(RuntimeError, _merge_info, [info_a, info_c])
    _force_update_info(info_a, info_c)
    assert (info_c['sfreq'] == info_a['sfreq'])
    assert (info_c['ch_names'][0] != info_a['ch_names'][0])
    # Make sure it works now
    _merge_info([info_a, info_c])
    # Check that you must supply Info
    pytest.raises(ValueError, _force_update_info, info_a,
                  dict([('sfreq', 1000.)]))
    # KIT System-ID
    info_a['kit_system_id'] = 50
    assert _merge_info((info_a, info_b))['kit_system_id'] == 50
    info_b['kit_system_id'] = 50
    assert _merge_info((info_a, info_b))['kit_system_id'] == 50
    info_b['kit_system_id'] = 60
    pytest.raises(ValueError, _merge_info, (info_a, info_b))


def test_check_consistency():
    """Test consistency check of Info objects."""
    info = create_info(ch_names=['a', 'b', 'c'], sfreq=1000.)

    # This should pass
    info._check_consistency()

    # Info without any channels
    info_empty = create_info(ch_names=[], sfreq=1000.)
    info_empty._check_consistency()

    # Bad channels that are not in the info object
    info2 = info.copy()
    info2['bads'] = ['b', 'foo', 'bar']
    pytest.raises(RuntimeError, info2._check_consistency)

    # Bad data types
    info2 = info.copy()
    info2['sfreq'] = 'foo'
    pytest.raises(ValueError, info2._check_consistency)

    info2 = info.copy()
    info2['highpass'] = 'foo'
    pytest.raises(ValueError, info2._check_consistency)

    info2 = info.copy()
    info2['lowpass'] = 'foo'
    pytest.raises(ValueError, info2._check_consistency)

    info2 = info.copy()
    info2['filename'] = 'foo'
    with warnings.catch_warnings(record=True) as w:
        info2._check_consistency()
    assert len(w) == 1
    assert (all('filename' in str(ww.message) for ww in w))

    # Silent type conversion to float
    info2 = info.copy()
    info2['sfreq'] = 1
    info2['highpass'] = 2
    info2['lowpass'] = 2
    info2._check_consistency()
    assert (isinstance(info2['sfreq'], float))
    assert (isinstance(info2['highpass'], float))
    assert (isinstance(info2['lowpass'], float))

    # Duplicate channel names
    info2 = info.copy()
    info2['chs'][2]['ch_name'] = 'b'
    pytest.raises(RuntimeError, info2._check_consistency)

    # Duplicates appended with running numbers
    with warnings.catch_warnings(record=True) as w:
        info3 = create_info(ch_names=['a', 'b', 'b', 'c', 'b'], sfreq=1000.)
    assert len(w) == 1
    assert (all('Channel names are not' in '%s' % ww.message for ww in w))
    assert_array_equal(info3['ch_names'], ['a', 'b-0', 'b-1', 'c', 'b-2'])


def test_anonymize():
    """Test that sensitive information can be anonymized."""
    pytest.raises(ValueError, anonymize_info, 'foo')

    # Fake some subject data
    raw = read_raw_fif(raw_fname)
    raw.info['subject_info'] = dict(id=1, his_id='foobar', last_name='bar',
                                    first_name='bar', birthday=(1987, 4, 8),
                                    sex=0, hand=1)

    orig_file_id = raw.info['file_id']['secs']
    orig_meas_id = raw.info['meas_id']['secs']
    # Test instance method
    events = read_events(event_name)
    epochs = Epochs(raw, events[:1], 2, 0., 0.1)
    for inst in [raw, epochs]:
        assert 'subject_info' in inst.info.keys()
        assert inst.info['subject_info'] is not None
        for key in ('file_id', 'meas_id'):
            assert inst.info[key]['secs'] != DATE_NONE[0]
            assert inst.info[key]['usecs'] != DATE_NONE[1]
        assert tuple(inst.info['meas_date']) != DATE_NONE
        inst.anonymize()
        assert 'subject_info' not in inst.info.keys()
        for key in ('file_id', 'meas_id'):
            assert inst.info[key]['secs'] == DATE_NONE[0]
            assert inst.info[key]['usecs'] == DATE_NONE[1]
        assert inst.info['meas_date'] is None

    # When we write out with raw.save, these get overwritten with the
    # new save time
    tempdir = _TempDir()
    out_fname = op.join(tempdir, 'test_subj_info_raw.fif')
    raw.save(out_fname, overwrite=True)
    raw = read_raw_fif(out_fname)
    assert raw.info.get('subject_info') is None
    assert raw.info['meas_date'] is None
    # XXX mne.io.write.write_id necessarily writes secs
    assert raw.info['file_id']['secs'] != orig_file_id
    assert raw.info['meas_id']['secs'] != orig_meas_id

    # Test no error for incomplete info
    info = raw.info.copy()
    info.pop('file_id')
    anonymize_info(info)


@testing.requires_testing_data
def test_csr_csc():
    """Test CSR and CSC."""
    info = read_info(sss_ctc_fname)
    info = pick_info(info, pick_types(info, meg=True, exclude=[]))
    sss_ctc = info['proc_history'][0]['max_info']['sss_ctc']
    ct = sss_ctc['decoupler'].copy()
    # CSC
    assert isinstance(ct, sparse.csc_matrix)
    tempdir = _TempDir()
    fname = op.join(tempdir, 'test.fif')
    write_info(fname, info)
    info_read = read_info(fname)
    ct_read = info_read['proc_history'][0]['max_info']['sss_ctc']['decoupler']
    assert isinstance(ct_read, sparse.csc_matrix)
    assert_array_equal(ct_read.toarray(), ct.toarray())
    # Now CSR
    csr = ct.tocsr()
    assert isinstance(csr, sparse.csr_matrix)
    assert_array_equal(csr.toarray(), ct.toarray())
    info['proc_history'][0]['max_info']['sss_ctc']['decoupler'] = csr
    fname = op.join(tempdir, 'test1.fif')
    write_info(fname, info)
    info_read = read_info(fname)
    ct_read = info_read['proc_history'][0]['max_info']['sss_ctc']['decoupler']
    assert isinstance(ct_read, sparse.csc_matrix)  # this gets cast to CSC
    assert_array_equal(ct_read.toarray(), ct.toarray())


@testing.requires_testing_data
def test_check_compensation_consistency():
    """Test check picks compensation."""
    raw = read_raw_ctf(ctf_fname, preload=False)
    events = make_fixed_length_events(raw, 99999)
    picks = pick_types(raw.info, meg=True, exclude=[], ref_meg=True)
    pick_ch_names = [raw.info['ch_names'][idx] for idx in picks]
    for (comp, expected_result) in zip([0, 1], [False, False]):
        raw.apply_gradient_compensation(comp)
        ret, missing = _bad_chans_comp(raw.info, pick_ch_names)
        assert ret == expected_result
        assert len(missing) == 0
        Epochs(raw, events, None, -0.2, 0.2, preload=False, picks=picks)

    picks = pick_types(raw.info, meg=True, exclude=[], ref_meg=False)
    pick_ch_names = [raw.info['ch_names'][idx] for idx in picks]

    for (comp, expected_result) in zip([0, 1], [False, True]):
        raw.apply_gradient_compensation(comp)
        ret, missing = _bad_chans_comp(raw.info, pick_ch_names)
        assert ret == expected_result
        assert len(missing) == 17
        if comp != 0:
            with pytest.raises(RuntimeError,
                               match='Compensation grade 1 has been applied'):
                Epochs(raw, events, None, -0.2, 0.2, preload=False,
                       picks=picks)
        else:
            Epochs(raw, events, None, -0.2, 0.2, preload=False, picks=picks)


run_tests_if_main()
