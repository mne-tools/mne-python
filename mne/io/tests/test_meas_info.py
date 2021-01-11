# -*- coding: utf-8 -*-
# # Authors: MNE Developers
#            Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import hashlib
import os.path as op
from datetime import datetime, timedelta, timezone, date

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy import sparse

from mne import Epochs, read_events, pick_info, pick_types, Annotations
from mne.channels import read_polhemus_fastscan
from mne.event import make_fixed_length_events
from mne.datasets import testing
from mne.io import (read_fiducials, write_fiducials, _coil_trans_to_loc,
                    _loc_to_coil_trans, read_raw_fif, read_info, write_info)
from mne.io.constants import FIFF
from mne.io.write import _generate_meas_id, DATE_NONE
from mne.io.meas_info import (Info, create_info, _merge_info,
                              _force_update_info, RAW_INFO_FIELDS,
                              _bad_chans_comp, _get_valid_units,
                              anonymize_info, _stamp_to_dt, _dt_to_stamp,
                              _add_timedelta_to_stamp)
from mne.io._digitization import _write_dig_points, _make_dig_points
from mne.io import read_raw_ctf
from mne.transforms import Transform
from mne.utils import catch_logging, assert_object_equal
from mne.channels import make_standard_montage, equalize_channels

fiducials_fname = op.join(op.dirname(__file__), '..', '..', 'data',
                          'fsaverage', 'fsaverage-fiducials.fif')
base_dir = op.join(op.dirname(__file__), 'data')
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
raw_invalid_bday_fname = op.join(data_path, 'misc',
                                 'sample_invalid_birthday_raw.fif')


@pytest.mark.parametrize('kwargs, want', [
    (dict(meg=False, eeg=True), [0]),
    (dict(meg=False, fnirs=True), [5]),
    (dict(meg=False, fnirs='hbo'), [5]),
    (dict(meg=False, fnirs='hbr'), []),
    (dict(meg=False, misc=True), [1]),
    (dict(meg=True), [2, 3, 4]),
    (dict(meg='grad'), [2, 3]),
    (dict(meg='planar1'), [2]),
    (dict(meg='planar2'), [3]),
    (dict(meg='mag'), [4]),
])
def test_create_info_grad(kwargs, want):
    """Test create_info behavior with grad coils."""
    info = create_info(6, 256, ["eeg", "misc", "grad", "grad", "mag", "hbo"])
    # Put these in an order such that grads get named "2" and "3", since
    # they get picked based first on coil_type then ch_name...
    assert [ch['ch_name'] for ch in info['chs']
            if ch['coil_type'] == FIFF.FIFFV_COIL_VV_PLANAR_T1] == ['2', '3']
    picks = pick_types(info, **kwargs)
    assert_array_equal(picks, want)


def test_get_valid_units():
    """Test the valid units."""
    valid_units = _get_valid_units()
    assert isinstance(valid_units, tuple)
    assert all(isinstance(unit, str) for unit in valid_units)
    assert "n/a" in valid_units


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
    n_ch = np.longlong(1)
    info = create_info(n_ch, 1000., 'eeg')
    assert set(info.keys()) == set(RAW_INFO_FIELDS)

    coil_types = {ch['coil_type'] for ch in info['chs']}
    assert FIFF.FIFFV_COIL_EEG in coil_types

    pytest.raises(TypeError, create_info, ch_names='Test Ch', sfreq=1000)
    pytest.raises(ValueError, create_info, ch_names=['Test Ch'], sfreq=-1000)
    pytest.raises(ValueError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types=['eeg', 'eeg'])
    pytest.raises(TypeError, create_info, ch_names=[np.array([1])],
                  sfreq=1000)
    pytest.raises(KeyError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types=np.array([1]))
    pytest.raises(KeyError, create_info, ch_names=['Test Ch'], sfreq=1000,
                  ch_types='awesome')
    pytest.raises(TypeError, create_info, ['Test Ch'], sfreq=1000,
                  montage=np.array([1]))
    m = make_standard_montage('biosemi32')
    info = create_info(ch_names=m.ch_names, sfreq=1000., ch_types='eeg')
    info.set_montage(m)
    ch_pos = [ch['loc'][:3] for ch in info['chs']]
    ch_pos_mon = m._get_ch_pos()
    ch_pos_mon = np.array(
        [ch_pos_mon[ch_name] for ch_name in info['ch_names']])
    # transform to head
    ch_pos_mon += (0., 0., 0.04014)
    assert_allclose(ch_pos, ch_pos_mon, atol=1e-5)


def test_duplicate_name_correction():
    """Test duplicate channel names with running number."""
    # When running number is possible
    info = create_info(['A', 'A', 'A'], 1000., verbose='error')
    assert info['ch_names'] == ['A-0', 'A-1', 'A-2']

    # When running number is not possible
    with pytest.raises(ValueError, match='Adding a running number'):
        create_info(['A', 'A', 'A-0'], 1000., verbose='error')


def test_fiducials_io(tmpdir):
    """Test fiducials i/o."""
    pts, coord_frame = read_fiducials(fiducials_fname)
    assert pts[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI
    assert pts[0]['ident'] == FIFF.FIFFV_POINT_CARDINAL

    temp_fname = tmpdir.join('test.fif')
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
    epochs = Epochs(raw, events[:1], event_id, tmin, tmax, picks=None)

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
        rep = repr(obj.info)
        assert '2002-12-03 19:01:10 UTC' in rep, rep
        assert '146 items (3 Cardinal, 4 HPI, 61 EEG, 78 Extra)' in rep
        dig_rep = repr(obj.info['dig'][0])
        assert 'LPA' in dig_rep, dig_rep
        assert '(-71.4, 0.0, 0.0) mm' in dig_rep, dig_rep
        assert 'head frame' in dig_rep, dig_rep
        # Test our BunchConstNamed support
        for func in (str, repr):
            assert '4 (FIFFV_COORD_HEAD)' == \
                func(obj.info['dig'][0]['coord_frame'])

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


def test_read_write_info(tmpdir):
    """Test IO of info."""
    info = read_info(raw_fname)
    temp_file = str(tmpdir.join('info.fif'))
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

    meas_id = info['meas_id']
    write_info(temp_file, info)
    info = read_info(temp_file)
    assert info['proc_history'][0]['creator'] == creator
    assert info['hpi_meas'][0]['creator'] == creator
    assert info['subject_info']['his_id'] == creator
    assert info['gantry_angle'] == gantry_angle
    assert info['subject_info']['height'] == 2.3
    assert info['subject_info']['weight'] == 11.1
    for key in ['secs', 'usecs', 'version']:
        assert info['meas_id'][key] == meas_id[key]
    assert_array_equal(info['meas_id']['machid'], meas_id['machid'])

    # Test that writing twice produces the same file
    m1 = hashlib.md5()
    with open(temp_file, 'rb') as fid:
        m1.update(fid.read())
    m1 = m1.hexdigest()
    temp_file_2 = tmpdir.join('info2.fif')
    assert temp_file_2 != temp_file
    write_info(temp_file_2, info)
    m2 = hashlib.md5()
    with open(str(temp_file_2), 'rb') as fid:
        m2.update(fid.read())
    m2 = m2.hexdigest()
    assert m1 == m2

    info = read_info(raw_fname)
    info['meas_date'] = None
    anonymize_info(info, verbose='error')
    assert info['meas_date'] is None
    tmp_fname_3 = tmpdir.join('info3.fif')
    write_info(tmp_fname_3, info)
    assert info['meas_date'] is None
    info2 = read_info(tmp_fname_3)
    assert info2['meas_date'] is None

    # Check that having a very old date in fine until you try to save it to fif
    info['meas_date'] = datetime(1800, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    info._check_consistency()
    fname = tmpdir.join('test.fif')
    with pytest.raises(RuntimeError, match='must be between '):
        write_info(fname, info)


def test_io_dig_points(tmpdir):
    """Test Writing for dig files."""
    points = read_polhemus_fastscan(hsp_fname, on_header_missing='ignore')

    dest = str(tmpdir.join('test.txt'))
    dest_bad = str(tmpdir.join('test.mne'))
    with pytest.raises(ValueError, match='must be of shape'):
        _write_dig_points(dest, points[:, :2])
    with pytest.raises(ValueError, match='extension'):
        _write_dig_points(dest_bad, points)
    _write_dig_points(dest, points)
    points1 = read_polhemus_fastscan(
        dest, unit='m', on_header_missing='ignore')
    err = "Dig points diverged after writing and reading."
    assert_array_equal(points, points1, err)

    points2 = np.array([[-106.93, 99.80], [99.80, 68.81]])
    np.savetxt(dest, points2, delimiter='\t', newline='\n')
    with pytest.raises(ValueError, match='must be of shape'):
        with pytest.warns(RuntimeWarning, match='FastSCAN header'):
            read_polhemus_fastscan(dest, on_header_missing='warn')


def test_make_dig_points():
    """Test application of Polhemus HSP to info."""
    extra_points = read_polhemus_fastscan(
        hsp_fname, on_header_missing='ignore')
    info = create_info(ch_names=['Test Ch'], sfreq=1000.)
    assert info['dig'] is None

    info['dig'] = _make_dig_points(extra_points=extra_points)
    assert (info['dig'])
    assert_allclose(info['dig'][0]['r'], [-.10693, .09980, .06881])

    elp_points = read_polhemus_fastscan(elp_fname, on_header_missing='ignore')
    nasion, lpa, rpa = elp_points[:3]
    info = create_info(ch_names=['Test Ch'], sfreq=1000.)
    assert info['dig'] is None

    info['dig'] = _make_dig_points(nasion, lpa, rpa, elp_points[3:], None)
    assert (info['dig'])
    idx = [d['ident'] for d in info['dig']].index(FIFF.FIFFV_POINT_NASION)
    assert_allclose(info['dig'][idx]['r'], [.0013930, .0131613, -.0046967])
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
    info = create_info(ch_names=['a', 'b', 'c'], sfreq=1000.)
    assert info['ch_names'][0] == 'a'
    assert info['ch_names'][1] == 'b'
    assert info['ch_names'][2] == 'c'

    # Equality
    assert info['ch_names'] == info['ch_names']
    assert info['ch_names'] == ['a', 'b', 'c']

    # No channels in info
    info = create_info(ch_names=[], sfreq=1000.)
    assert info['ch_names'] == []

    # List should be read-only
    info = create_info(ch_names=['a', 'b', 'c'], sfreq=1000.)


def test_merge_info():
    """Test merging of multiple Info objects."""
    info_a = create_info(ch_names=['a', 'b', 'c'], sfreq=1000.)
    info_b = create_info(ch_names=['d', 'e', 'f'], sfreq=1000.)
    info_merged = _merge_info([info_a, info_b])
    assert info_merged['nchan'], 6
    assert info_merged['ch_names'], ['a', 'b', 'c', 'd', 'e', 'f']
    pytest.raises(ValueError, _merge_info, [info_a, info_a])

    # Testing for force updates before merging
    info_c = create_info(ch_names=['g', 'h', 'i'], sfreq=500.)
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

    # hpi infos
    info_d = create_info(ch_names=['d', 'e', 'f'], sfreq=1000.)
    info_merged = _merge_info([info_a, info_d])
    assert not info_merged['hpi_meas']
    assert not info_merged['hpi_results']
    info_a['hpi_meas'] = [{'f1': 3, 'f2': 4}]
    assert _merge_info([info_a, info_d])['hpi_meas'] == info_a['hpi_meas']
    info_d['hpi_meas'] = [{'f1': 3, 'f2': 4}]
    assert _merge_info([info_a, info_d])['hpi_meas'] == info_d['hpi_meas']
    # This will break because of inconsistency
    info_d['hpi_meas'] = [{'f1': 3, 'f2': 5}]
    pytest.raises(ValueError, _merge_info, [info_a, info_d])

    info_0 = read_info(raw_fname)
    info_0['bads'] = ['MEG 2443', 'EEG 053']
    assert len(info_0['chs']) == 376
    assert len(info_0['dig']) == 146
    info_1 = create_info(["STI XXX"], info_0['sfreq'], ['stim'])
    assert info_1['bads'] == []
    info_out = _merge_info([info_0, info_1], force_update_to_first=True)
    assert len(info_out['chs']) == 377
    assert len(info_out['bads']) == 2
    assert len(info_out['dig']) == 146
    assert len(info_0['chs']) == 376
    assert len(info_0['bads']) == 2
    assert len(info_0['dig']) == 146


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
    with pytest.warns(RuntimeWarning, match='filename'):
        info2._check_consistency()

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
    with pytest.warns(RuntimeWarning, match='Channel names are not'):
        info3 = create_info(ch_names=['a', 'b', 'b', 'c', 'b'], sfreq=1000.)
    assert_array_equal(info3['ch_names'], ['a', 'b-0', 'b-1', 'c', 'b-2'])

    # a few bad ones
    idx = 0
    ch = info['chs'][idx]
    for key, bad, match in (('ch_name', 1., 'not a string'),
                            ('loc', np.zeros(15), '12 elements'),
                            ('cal', np.ones(1), 'float or int')):
        info._check_consistency()  # okay
        old = ch[key]
        ch[key] = bad
        if key == 'ch_name':
            info['ch_names'][idx] = bad
        with pytest.raises(TypeError, match=match):
            info._check_consistency()
        ch[key] = old
        if key == 'ch_name':
            info['ch_names'][idx] = old

    # bad channel entries
    info2 = info.copy()
    info2['chs'][0]['foo'] = 'bar'
    with pytest.raises(KeyError, match='key errantly present'):
        info2._check_consistency()
    info2 = info.copy()
    del info2['chs'][0]['loc']
    with pytest.raises(KeyError, match='key missing'):
        info2._check_consistency()


def _test_anonymize_info(base_info):
    """Test that sensitive information can be anonymized."""
    pytest.raises(TypeError, anonymize_info, 'foo')

    default_anon_dos = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    default_str = "mne_anonymize"
    default_subject_id = 0
    default_desc = ("Anonymized using a time shift" +
                    " to preserve age at acquisition")

    # Test no error for incomplete info
    info = base_info.copy()
    info.pop('file_id')
    anonymize_info(info)

    # Fake some subject data
    meas_date = datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    base_info['meas_date'] = meas_date

    base_info['subject_info'] = dict(id=1, his_id='foobar', last_name='bar',
                                     first_name='bar', birthday=(1987, 4, 8),
                                     sex=0, hand=1)

    # generate expected info...
    # first expected result with no options.
    # will move DOS from 2010/1/1 to 2000/1/1 which is 3653 days.
    exp_info = base_info.copy()
    exp_info['description'] = default_desc
    exp_info['experimenter'] = default_str
    exp_info['proj_name'] = default_str
    exp_info['proj_id'] = np.array([0])
    exp_info['subject_info']['first_name'] = default_str
    exp_info['subject_info']['last_name'] = default_str
    exp_info['subject_info']['id'] = default_subject_id
    exp_info['subject_info']['his_id'] = str(default_subject_id)
    # this bday is 3653 days different. the change in day is due to a
    # different number of leap days between 1987 and 1977 than between
    # 2010 and 2000.
    exp_info['subject_info']['birthday'] = (1977, 4, 7)
    exp_info['meas_date'] = default_anon_dos

    # make copies
    exp_info_3 = exp_info.copy()

    # adjust each expected outcome
    delta_t = timedelta(days=3653)
    for key in ('file_id', 'meas_id'):
        value = exp_info.get(key)
        if value is not None:
            assert 'msecs' not in value
            tmp = _add_timedelta_to_stamp(
                (value['secs'], value['usecs']), -delta_t)
            value['secs'] = tmp[0]
            value['usecs'] = tmp[1]
            value['machid'][:] = 0

    # exp 2 tests the keep_his option
    exp_info_2 = exp_info.copy()
    exp_info_2['subject_info']['his_id'] = 'foobar'

    # exp 3 tests is a supplied daysback
    delta_t_2 = timedelta(days=43)
    exp_info_3['subject_info']['birthday'] = (1987, 2, 24)
    exp_info_3['meas_date'] = meas_date - delta_t_2
    for key in ('file_id', 'meas_id'):
        value = exp_info_3.get(key)
        if value is not None:
            assert 'msecs' not in value
            tmp = _add_timedelta_to_stamp(
                (value['secs'], value['usecs']), -delta_t_2)
            value['secs'] = tmp[0]
            value['usecs'] = tmp[1]
            value['machid'][:] = 0

    # exp 4 tests is a supplied daysback
    delta_t_3 = timedelta(days=223 + 364 * 500)

    new_info = anonymize_info(base_info.copy())
    assert_object_equal(new_info, exp_info)

    new_info = anonymize_info(base_info.copy(), keep_his=True)
    assert_object_equal(new_info, exp_info_2)

    new_info = anonymize_info(base_info.copy(), daysback=delta_t_2.days)
    assert_object_equal(new_info, exp_info_3)

    with pytest.raises(RuntimeError, match='anonymize_info generated'):
        anonymize_info(base_info.copy(), daysback=delta_t_3.days)
    # assert_object_equal(new_info, exp_info_4)

    # test with meas_date = None
    base_info['meas_date'] = None
    exp_info_3['meas_date'] = None
    exp_info_3['file_id']['secs'] = DATE_NONE[0]
    exp_info_3['file_id']['usecs'] = DATE_NONE[1]
    exp_info_3['meas_id']['secs'] = DATE_NONE[0]
    exp_info_3['meas_id']['usecs'] = DATE_NONE[1]
    exp_info_3['subject_info'].pop('birthday', None)

    if base_info['meas_date'] is None:
        with pytest.warns(RuntimeWarning, match='all information'):
            new_info = anonymize_info(base_info.copy(),
                                      daysback=delta_t_2.days)
    else:
        new_info = anonymize_info(base_info.copy(), daysback=delta_t_2.days)
    assert_object_equal(new_info, exp_info_3)

    with pytest.warns(None):  # meas_date is None
        new_info = anonymize_info(base_info.copy())
    assert_object_equal(new_info, exp_info_3)


@pytest.mark.parametrize('stamp, dt', [
    [(1346981585, 835782), (2012, 9, 7, 1, 33, 5, 835782)],
    # test old dates for BIDS anonymization
    [(-1533443343, 24382), (1921, 5, 29, 19, 30, 57, 24382)],
    # gh-7116
    [(-908196946, 988669), (1941, 3, 22, 11, 4, 14, 988669)],
])
def test_meas_date_convert(stamp, dt):
    """Test conversions of meas_date to datetime objects."""
    meas_datetime = _stamp_to_dt(stamp)
    stamp2 = _dt_to_stamp(meas_datetime)
    assert stamp == stamp2
    assert meas_datetime == datetime(*dt, tzinfo=timezone.utc)
    # smoke test for info __repr__
    info = create_info(1, 1000., 'eeg')
    info['meas_date'] = meas_datetime
    assert str(dt[0]) in repr(info)


def test_anonymize(tmpdir):
    """Test that sensitive information can be anonymized."""
    pytest.raises(TypeError, anonymize_info, 'foo')

    # Fake some subject data
    raw = read_raw_fif(raw_fname)
    raw.set_annotations(Annotations(onset=[0, 1],
                                    duration=[1, 1],
                                    description='dummy',
                                    orig_time=None))
    first_samp = raw.first_samp
    expected_onset = np.arange(2) + raw._first_time
    assert raw.first_samp == first_samp
    assert_allclose(raw.annotations.onset, expected_onset)

    # Test instance method
    events = read_events(event_name)
    epochs = Epochs(raw, events[:1], 2, 0., 0.1, baseline=None)

    _test_anonymize_info(raw.info.copy())
    _test_anonymize_info(epochs.info.copy())

    # test that annotations are correctly zeroed
    raw.anonymize()
    assert raw.first_samp == first_samp
    assert_allclose(raw.annotations.onset, expected_onset)
    assert raw.annotations.orig_time == raw.info['meas_date']
    stamp = _dt_to_stamp(raw.info['meas_date'])
    assert raw.annotations.orig_time == _stamp_to_dt(stamp)

    raw.info['meas_date'] = None
    with pytest.warns(RuntimeWarning, match='None'):
        raw.anonymize()
    assert raw.annotations.orig_time is None
    assert raw.first_samp == first_samp
    assert_allclose(raw.annotations.onset, expected_onset)


def test_anonymize_with_io(tmpdir):
    """Test that IO does not break anonymization."""
    raw = read_raw_fif(raw_fname)

    temp_path = tmpdir.join('tmp_raw.fif')
    raw.save(temp_path)

    raw2 = read_raw_fif(temp_path)

    daysback = (raw2.info['meas_date'].date() - date(1924, 1, 1)).days
    raw2.anonymize(daysback=daysback)


@testing.requires_testing_data
def test_csr_csc(tmpdir):
    """Test CSR and CSC."""
    info = read_info(sss_ctc_fname)
    info = pick_info(info, pick_types(info, meg=True, exclude=[]))
    sss_ctc = info['proc_history'][0]['max_info']['sss_ctc']
    ct = sss_ctc['decoupler'].copy()
    # CSC
    assert isinstance(ct, sparse.csc_matrix)
    fname = tmpdir.join('test.fif')
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
    fname = tmpdir.join('test1.fif')
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
        with catch_logging() as log:
            Epochs(raw, events, None, -0.2, 0.2, preload=False,
                   picks=picks, verbose=True)
            assert'Removing 5 compensators' in log.getvalue()


def test_field_round_trip(tmpdir):
    """Test round-trip for new fields."""
    info = create_info(1, 1000., 'eeg')
    for key in ('file_id', 'meas_id'):
        info[key] = _generate_meas_id()
    info['device_info'] = dict(
        type='a', model='b', serial='c', site='d')
    info['helium_info'] = dict(
        he_level_raw=1., helium_level=2., orig_file_guid='e', meas_date=(1, 2))
    fname = tmpdir.join('temp-info.fif')
    write_info(fname, info)
    info_read = read_info(fname)
    info_read['dig'] = None  # XXX eventually this should go away
    assert_object_equal(info, info_read)


def test_equalize_channels():
    """Test equalization of channels for instances of Info."""
    info1 = create_info(['CH1', 'CH2', 'CH3'], sfreq=1.)
    info2 = create_info(['CH4', 'CH2', 'CH1'], sfreq=1.)
    info1, info2 = equalize_channels([info1, info2])

    assert info1.ch_names == ['CH1', 'CH2']
    assert info2.ch_names == ['CH1', 'CH2']


def test_repr():
    """Test Info repr."""
    info = create_info(1, 1000, 'eeg')
    assert '7 non-empty values' in repr(info)

    t = Transform(1, 2, np.ones((4, 4)))
    info['dev_head_t'] = t
    assert 'dev_head_t: MEG device -> isotrak transform' in repr(info)


@testing.requires_testing_data
def test_invalid_subject_birthday():
    """Test handling of an invalid birthday in the raw file."""
    with pytest.warns(RuntimeWarning, match='No birthday will be set'):
        raw = read_raw_fif(raw_invalid_bday_fname)
    assert 'birthday' not in raw.info['subject_info']
