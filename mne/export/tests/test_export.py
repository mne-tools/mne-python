# -*- coding: utf-8 -*-
"""Test exporting functions."""
# Authors: MNE Developers
#
# License: BSD-3-Clause

from datetime import datetime, timezone
from mne.io import RawArray
from mne.io.meas_info import create_info
from pathlib import Path
import os.path as op

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)

from mne import (read_epochs_eeglab, Epochs, read_evokeds, read_evokeds_mff,
                 Annotations)
from mne.datasets import testing, misc
from mne.export import export_evokeds, export_evokeds_mff
from mne.io import read_raw_fif, read_raw_eeglab, read_raw_edf
from mne.utils import (_check_eeglabio_installed, requires_version,
                       object_diff, _check_edflib_installed, _resource_path)
from mne.tests.test_epochs import _get_data

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')

data_path = testing.data_path(download=False)
egi_evoked_fname = op.join(data_path, 'EGI', 'test_egi_evoked.mff')


@pytest.mark.skipif(not _check_eeglabio_installed(strict=False),
                    reason='eeglabio not installed')
def test_export_raw_eeglab(tmp_path):
    """Test saving a Raw instance to EEGLAB's set format."""
    fname = (Path(__file__).parent.parent.parent /
             "io" / "tests" / "data" / "test_raw.fif")
    raw = read_raw_fif(fname, preload=True)
    raw.apply_proj()
    temp_fname = op.join(str(tmp_path), 'test.set')
    raw.export(temp_fname)
    raw.drop_channels([ch for ch in ['epoc']
                       if ch in raw.ch_names])
    raw_read = read_raw_eeglab(temp_fname, preload=True)
    assert raw.ch_names == raw_read.ch_names
    cart_coords = np.array([d['loc'][:3] for d in raw.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3] for d in raw_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert_allclose(raw.times, raw_read.times)
    assert_allclose(raw.get_data(), raw_read.get_data())

    # test overwrite
    with pytest.raises(FileExistsError, match='Destination file exists'):
        raw.export(temp_fname, overwrite=False)
    raw.export(temp_fname, overwrite=True)

    # test pathlib.Path files
    raw.export(Path(temp_fname), overwrite=True)

    # test warning with unapplied projectors
    raw = read_raw_fif(fname, preload=True)
    with pytest.warns(RuntimeWarning,
                      match='Raw instance has unapplied projectors.'):
        raw.export(temp_fname, overwrite=True)


@pytest.mark.skipif(not _check_edflib_installed(strict=False),
                    reason='edflib-python not installed')
def test_double_export_edf(tmp_path):
    """Test exporting an EDF file multiple times."""
    rng = np.random.RandomState(123456)
    format = 'edf'
    ch_types = ['eeg', 'eeg', 'stim', 'ecog', 'ecog', 'seeg', 'eog', 'ecg',
                'emg', 'dbs', 'bio']
    info = create_info(len(ch_types), sfreq=1000, ch_types=ch_types)
    data = rng.random(size=(len(ch_types), 1000)) * 1e-5

    # include subject info and measurement date
    info['subject_info'] = dict(first_name='mne', last_name='python',
                                birthday=(1992, 1, 20), sex=1, hand=3)
    raw = RawArray(data, info)

    # export once
    temp_fname = tmp_path / f'test.{format}'
    raw.export(temp_fname, add_ch_type=True)
    raw_read = read_raw_edf(temp_fname, infer_types=True, preload=True)

    # export again
    raw_read.load_data()
    raw_read.export(temp_fname, add_ch_type=True, overwrite=True)
    raw_read = read_raw_edf(temp_fname, infer_types=True, preload=True)

    # stim channel should be dropped
    raw.drop_channels('2')

    assert raw.ch_names == raw_read.ch_names
    # only compare the original length, since extra zeros are appended
    orig_raw_len = len(raw)
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=4)
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)

    # check channel types except for 'bio', which loses its type
    orig_ch_types = raw.get_channel_types()
    read_ch_types = raw_read.get_channel_types()
    assert_array_equal(orig_ch_types, read_ch_types)


@pytest.mark.skipif(not _check_edflib_installed(strict=False),
                    reason='edflib-python not installed')
def test_export_edf_annotations(tmp_path):
    """Test that exporting EDF preserves annotations."""
    rng = np.random.RandomState(123456)
    format = 'edf'
    ch_types = ['eeg', 'eeg', 'stim', 'ecog', 'ecog', 'seeg',
                'eog', 'ecg', 'emg', 'dbs', 'bio']
    ch_names = np.arange(len(ch_types)).astype(str).tolist()
    info = create_info(ch_names, sfreq=1000,
                       ch_types=ch_types)
    data = rng.random(size=(len(ch_names), 2000)) * 1.e-5
    raw = RawArray(data, info)

    annotations = Annotations(
        onset=[0.01, 0.05, 0.90, 1.05], duration=[0, 1, 0, 0],
        description=['test1', 'test2', 'test3', 'test4'])
    raw.set_annotations(annotations)

    # export
    temp_fname = op.join(str(tmp_path), f'test.{format}')
    raw.export(temp_fname)

    # read in the file
    raw_read = read_raw_edf(temp_fname, preload=True)
    assert_array_equal(raw.annotations.onset, raw_read.annotations.onset)
    assert_array_equal(raw.annotations.duration, raw_read.annotations.duration)
    assert_array_equal(raw.annotations.description,
                       raw_read.annotations.description)


@pytest.mark.skipif(not _check_edflib_installed(strict=False),
                    reason='edflib-python not installed')
def test_rawarray_edf(tmp_path):
    """Test saving a Raw array with integer sfreq to EDF."""
    rng = np.random.RandomState(12345)
    format = 'edf'
    ch_types = ['eeg', 'eeg', 'stim', 'ecog', 'seeg', 'eog', 'ecg', 'emg',
                'dbs', 'bio']
    ch_names = np.arange(len(ch_types)).astype(str).tolist()
    info = create_info(ch_names, sfreq=1000,
                       ch_types=ch_types)
    data = rng.random(size=(len(ch_names), 1000)) * 1e-5

    # include subject info and measurement date
    subject_info = dict(first_name='mne', last_name='python',
                        birthday=(1992, 1, 20), sex=1, hand=3)
    info['subject_info'] = subject_info
    raw = RawArray(data, info)
    time_now = datetime.now()
    meas_date = datetime(year=time_now.year, month=time_now.month,
                         day=time_now.day, hour=time_now.hour,
                         minute=time_now.minute, second=time_now.second,
                         tzinfo=timezone.utc)
    raw.set_meas_date(meas_date)
    temp_fname = op.join(str(tmp_path), f'test.{format}')

    raw.export(temp_fname, add_ch_type=True)
    raw_read = read_raw_edf(temp_fname, infer_types=True, preload=True)

    # stim channel should be dropped
    raw.drop_channels('2')

    assert raw.ch_names == raw_read.ch_names
    # only compare the original length, since extra zeros are appended
    orig_raw_len = len(raw)
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=4)
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)

    # check channel types except for 'bio', which loses its type
    orig_ch_types = raw.get_channel_types()
    read_ch_types = raw_read.get_channel_types()
    assert_array_equal(orig_ch_types, read_ch_types)
    assert raw.info['meas_date'] == raw_read.info['meas_date']

    # channel name can't be longer than 16 characters with the type added
    raw_bad = raw.copy()
    raw_bad.rename_channels({'1': 'abcdefghijklmnopqrstuvwxyz'})
    with pytest.raises(RuntimeError, match='Signal label'), \
            pytest.warns(RuntimeWarning, match='Data has a non-integer'):
        raw_bad.export(temp_fname, overwrite=True)

    # include bad birthday that is non-EDF compliant
    bad_info = info.copy()
    bad_info['subject_info']['birthday'] = (1700, 1, 20)
    raw = RawArray(data, bad_info)
    with pytest.raises(RuntimeError, match='Setting patient birth date'):
        raw.export(temp_fname, overwrite=True)

    # include bad measurement date that is non-EDF compliant
    raw = RawArray(data, info)
    meas_date = datetime(year=1984, month=1, day=1, tzinfo=timezone.utc)
    raw.set_meas_date(meas_date)
    with pytest.raises(RuntimeError, match='Setting start date time'):
        raw.export(temp_fname, overwrite=True)

    # test that warning is raised if there are non-voltage based channels
    raw = RawArray(data, info)
    with pytest.warns(RuntimeWarning, match='The unit'):
        raw.set_channel_types({'9': 'hbr'})
    with pytest.warns(RuntimeWarning, match='Non-voltage channels'):
        raw.export(temp_fname, overwrite=True)

    # data should match up to the non-accepted channel
    raw_read = read_raw_edf(temp_fname, preload=True)
    orig_raw_len = len(raw)
    assert_array_almost_equal(
        raw.get_data()[:-1, :], raw_read.get_data()[:, :orig_raw_len],
        decimal=4)
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)

    # the data should still match though
    raw_read = read_raw_edf(temp_fname, preload=True)
    raw.drop_channels('2')
    assert raw.ch_names == raw_read.ch_names
    orig_raw_len = len(raw)
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=4)
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)


@pytest.mark.skipif(not _check_edflib_installed(strict=False),
                    reason='edflib-python not installed')
@pytest.mark.parametrize(
    ['dataset', 'format'], [
        ['test', 'edf'],
        pytest.param('misc', 'edf', marks=pytest.mark.slowtest),
    ])
def test_export_raw_edf(tmp_path, dataset, format):
    """Test saving a Raw instance to EDF format."""
    if dataset == 'test':
        fname = _resource_path('mne.io.tests.data', 'test_raw.fif')
        raw = read_raw_fif(fname)
    elif dataset == 'misc':
        fname = op.join(misc.data_path(), 'ecog', 'sample_ecog_ieeg.fif')
        raw = read_raw_fif(fname)

    # only test with EEG channels
    raw.pick_types(eeg=True, ecog=True, seeg=True)
    raw.load_data()
    orig_ch_names = raw.ch_names
    temp_fname = op.join(str(tmp_path), f'test.{format}')

    # test runtime errors
    with pytest.raises(RuntimeError, match='The maximum'), \
            pytest.warns(RuntimeWarning, match='Data has a non-integer'):
        raw.export(temp_fname, physical_range=(-1e6, 0))
    with pytest.raises(RuntimeError, match='The minimum'), \
            pytest.warns(RuntimeWarning, match='Data has a non-integer'):
        raw.export(temp_fname, physical_range=(0, 1e6))

    if dataset == 'test':
        with pytest.warns(RuntimeWarning, match='Data has a non-integer'):
            raw.export(temp_fname)
    elif dataset == 'misc':
        with pytest.warns(RuntimeWarning, match='EDF format requires'):
            raw.export(temp_fname)

    if 'epoc' in raw.ch_names:
        raw.drop_channels(['epoc'])

    raw_read = read_raw_edf(temp_fname, preload=True)
    assert orig_ch_names == raw_read.ch_names
    # only compare the original length, since extra zeros are appended
    orig_raw_len = len(raw)

    # assert data and times are not different
    # Due to the physical range of the data, reading and writing is
    # not lossless. For example, a physical min/max of -/+ 3200 uV
    # will result in a resolution of 0.09 uV. This resolution
    # though is acceptable for most EEG manufacturers.
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=4)

    # Due to the data record duration limitations of EDF files, one
    # cannot store arbitrary float sampling rate exactly. Usually this
    # results in two sampling rates that are off by very low number of
    # decimal points. This for practical purposes does not matter
    # but will result in an error when say the number of time points
    # is very very large.
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)


@pytest.mark.skipif(not _check_eeglabio_installed(strict=False),
                    reason='eeglabio not installed')
@pytest.mark.parametrize('preload', (True, False))
def test_export_epochs_eeglab(tmp_path, preload):
    """Test saving an Epochs instance to EEGLAB's set format."""
    raw, events = _get_data()[:2]
    raw.load_data()
    epochs = Epochs(raw, events, preload=preload)
    temp_fname = op.join(str(tmp_path), 'test.set')
    epochs.export(temp_fname)
    epochs.drop_channels([ch for ch in ['epoc', 'STI 014']
                          if ch in epochs.ch_names])
    epochs_read = read_epochs_eeglab(temp_fname)
    assert epochs.ch_names == epochs_read.ch_names
    cart_coords = np.array([d['loc'][:3]
                           for d in epochs.info['chs']])  # just xyz
    cart_coords_read = np.array([d['loc'][:3]
                                for d in epochs_read.info['chs']])
    assert_allclose(cart_coords, cart_coords_read)
    assert_array_equal(epochs.events[:, 0],
                       epochs_read.events[:, 0])  # latency
    assert epochs.event_id.keys() == epochs_read.event_id.keys()  # just keys
    assert_allclose(epochs.times, epochs_read.times)
    assert_allclose(epochs.get_data(), epochs_read.get_data())

    # test overwrite
    with pytest.raises(FileExistsError, match='Destination file exists'):
        epochs.export(temp_fname, overwrite=False)
    epochs.export(temp_fname, overwrite=True)

    # test pathlib.Path files
    epochs.export(Path(temp_fname), overwrite=True)

    # test warning with unapplied projectors
    epochs = Epochs(raw, events, preload=preload, proj=False)
    with pytest.warns(RuntimeWarning,
                      match='Epochs instance has unapplied projectors.'):
        epochs.export(Path(temp_fname), overwrite=True)


@requires_version('mffpy', '0.5.7')
@testing.requires_testing_data
@pytest.mark.parametrize('fmt', ('auto', 'mff'))
@pytest.mark.parametrize('do_history', (True, False))
def test_export_evokeds_to_mff(tmp_path, fmt, do_history):
    """Test exporting evoked dataset to MFF."""
    evoked = read_evokeds_mff(egi_evoked_fname)
    export_fname = op.join(str(tmp_path), 'evoked.mff')
    history = [
        {
            'name': 'Test Segmentation',
            'method': 'Segmentation',
            'settings': ['Setting 1', 'Setting 2'],
            'results': ['Result 1', 'Result 2']
        },
        {
            'name': 'Test Averaging',
            'method': 'Averaging',
            'settings': ['Setting 1', 'Setting 2'],
            'results': ['Result 1', 'Result 2']
        }
    ]
    if do_history:
        export_evokeds_mff(export_fname, evoked, history=history)
    else:
        export_evokeds(export_fname, evoked)
    # Drop non-EEG channels
    evoked = [ave.drop_channels(['ECG', 'EMG']) for ave in evoked]
    evoked_exported = read_evokeds_mff(export_fname)
    assert len(evoked) == len(evoked_exported)
    for ave, ave_exported in zip(evoked, evoked_exported):
        # Compare infos
        assert object_diff(ave_exported.info, ave.info) == ''
        # Compare data
        assert_allclose(ave_exported.data, ave.data)
        # Compare properties
        assert ave_exported.nave == ave.nave
        assert ave_exported.kind == ave.kind
        assert ave_exported.comment == ave.comment
        assert_allclose(ave_exported.times, ave.times)

    # test overwrite
    with pytest.raises(FileExistsError, match='Destination file exists'):
        if do_history:
            export_evokeds_mff(export_fname, evoked, history=history,
                               overwrite=False)
        else:
            export_evokeds(export_fname, evoked, overwrite=False)

    if do_history:
        export_evokeds_mff(export_fname, evoked, history=history,
                           overwrite=True)
    else:
        export_evokeds(export_fname, evoked, overwrite=True)


@requires_version('mffpy', '0.5.7')
@testing.requires_testing_data
def test_export_to_mff_no_device():
    """Test no device type throws ValueError."""
    evoked = read_evokeds_mff(egi_evoked_fname, condition='Category 1')
    evoked.info['device_info'] = None
    with pytest.raises(ValueError, match='No device type.'):
        export_evokeds('output.mff', evoked)


@requires_version('mffpy', '0.5.7')
def test_export_to_mff_incompatible_sfreq():
    """Test non-whole number sampling frequency throws ValueError."""
    evoked = read_evokeds(fname_evoked)
    with pytest.raises(ValueError, match=f'sfreq: {evoked[0].info["sfreq"]}'):
        export_evokeds('output.mff', evoked)


@pytest.mark.parametrize('fmt,ext', [
    ('EEGLAB', 'set'),
    ('EDF', 'edf'),
    ('BrainVision', 'eeg')
])
def test_export_evokeds_unsupported_format(fmt, ext):
    """Test exporting evoked dataset to non-supported formats."""
    evoked = read_evokeds(fname_evoked)
    with pytest.raises(NotImplementedError, match=f'Export to {fmt} not imp'):
        export_evokeds(f'output.{ext}', evoked)
