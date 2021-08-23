# -*- coding: utf-8 -*-
"""Test exporting functions."""
# Authors: MNE Developers
#
# License: BSD-3-Clause

from mne.io import RawArray
from mne.io.meas_info import create_info
from mne.io.edf.edf import read_raw_bdf
from pathlib import Path
import os.path as op

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_equal)

from mne import read_epochs_eeglab, Epochs, read_evokeds, read_evokeds_mff
from mne.datasets import testing, misc
from mne.export import export_evokeds, export_evokeds_mff
from mne.io import read_raw_fif, read_raw_eeglab, read_raw_edf
from mne.utils import (_check_eeglabio_installed, requires_version,
                       object_diff, _check_edflib_installed)
from mne.tests.test_epochs import _get_data

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')

data_path = testing.data_path(download=False)
egi_evoked_fname = op.join(data_path, 'EGI', 'test_egi_evoked.mff')


@pytest.mark.skipif(not _check_eeglabio_installed(strict=False),
                    reason='eeglabio not installed')
def test_export_raw_eeglab(tmpdir):
    """Test saving a Raw instance to EEGLAB's set format."""
    fname = (Path(__file__).parent.parent.parent /
             "io" / "tests" / "data" / "test_raw.fif")
    raw = read_raw_fif(fname)
    raw.load_data()
    temp_fname = op.join(str(tmpdir), 'test.set')
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


@pytest.mark.skipif(not _check_edflib_installed(strict=False),
                    reason='edflib-python not installed')
def test_integer_sfreq_edf_and_bdf(tmp_path, format):
    """Test saving a Raw array with integer sfreq to EDF."""
    info = create_info(['1', '2', '3'], sfreq=1000, ch_types='eeg')
    data = np.random.random(size=(3, 1000))
    raw = RawArray(data, info)

    temp_fname = op.join(str(tmp_path), f'test.{format}')

    raw.export(temp_fname)
    raw_read = read_raw_edf(temp_fname, preload=True)

    assert raw.ch_names == raw_read.ch_names
    # only compare the original length, since extra zeros are appended
    orig_raw_len = len(raw)
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=3)
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)


@pytest.mark.skipif(not _check_edflib_installed(strict=False),
                    reason='edflib-python not installed')
@pytest.mark.parametrize(
    ['dataset', 'format'], [
        ['test', 'edf'],
        ['misc', 'edf'],
    ])
def test_export_raw_edf_and_bdf(tmp_path, dataset, format):
    """Test saving a Raw instance to EDF format."""
    if dataset == 'test':
        try:
            import importlib_resources as imp_resrc  # py 3.7
        except ImportError:
            import importlib.resources as imp_resrc  # py 3.8+
        import mne.io.tests.data
        with imp_resrc.path(mne.io.tests.data, 'test_raw.fif') as fname:
            raw = read_raw_fif(fname)
    elif dataset == 'misc':
        fname = op.join(misc.data_path(),
                        'ecog', 'sample_ecog.edf')
        raw = read_raw_edf(fname)

    # only test with EEG channels
    if dataset == 'misc':
        raw = raw.set_channel_types({'EKGL': 'ecg', 'EKGR': 'ecg'})
    raw.pick_types(eeg=True, ecog=True, seeg=True,
                   eog=True, ecg=True, emg=True)
    raw.load_data()
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

    if format == 'edf':
        raw_read = read_raw_edf(temp_fname, preload=True)

    assert raw.ch_names == raw_read.ch_names
    # only compare the original length, since extra zeros are appended
    orig_raw_len = len(raw)
    assert_array_almost_equal(
        raw.get_data(), raw_read.get_data()[:, :orig_raw_len], decimal=3)
    assert_allclose(
        raw.times, raw_read.times[:orig_raw_len], rtol=0, atol=1e-5)


@pytest.mark.skipif(not _check_eeglabio_installed(strict=False),
                    reason='eeglabio not installed')
@pytest.mark.parametrize('preload', (True, False))
def test_export_epochs_eeglab(tmpdir, preload):
    """Test saving an Epochs instance to EEGLAB's set format."""
    raw, events = _get_data()[:2]
    raw.load_data()
    epochs = Epochs(raw, events, preload=preload)
    temp_fname = op.join(str(tmpdir), 'test.set')
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


@requires_version('mffpy', '0.5.7')
@testing.requires_testing_data
@pytest.mark.parametrize('fmt', ('auto', 'mff'))
@pytest.mark.parametrize('do_history', (True, False))
def test_export_evokeds_to_mff(tmpdir, fmt, do_history):
    """Test exporting evoked dataset to MFF."""
    evoked = read_evokeds_mff(egi_evoked_fname)
    export_fname = op.join(str(tmpdir), 'evoked.mff')
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
