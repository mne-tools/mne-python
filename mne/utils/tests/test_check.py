"""Test check utilities."""
# Authors: MNE Developers
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause
import os
import os.path as op
import sys

import numpy as np
import pytest
from pathlib import Path

import mne
from mne import read_vectorview_selection
from mne.datasets import testing
from mne.io.pick import pick_channels_cov, _picks_to_idx
from mne.utils import (check_random_state, _check_fname, check_fname,
                       _check_subject, requires_mayavi, traits_test,
                       _check_mayavi_version, _check_info_inv, _check_option,
                       check_version, _path_like, _validate_type,
                       _suggest, _on_missing, requires_nibabel, _safe_input,
                       _check_ch_locs)

data_path = testing.data_path(download=False)
base_dir = op.join(data_path, 'MEG', 'sample')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_event = op.join(base_dir, 'sample_audvis_trunc_raw-eve.fif')
fname_fwd = op.join(base_dir, 'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_mgz = op.join(data_path, 'subjects', 'sample', 'mri', 'aseg.mgz')
reject = dict(grad=4000e-13, mag=4e-12)


@testing.requires_testing_data
def test_check(tmp_path):
    """Test checking functions."""
    pytest.raises(ValueError, check_random_state, 'foo')
    pytest.raises(TypeError, _check_fname, 1)
    _check_fname(Path('./foo'))
    fname = tmp_path / 'foo'
    with open(fname, 'wb'):
        pass
    assert op.isfile(fname)
    _check_fname(fname, overwrite='read', must_exist=True)
    orig_perms = os.stat(fname).st_mode
    os.chmod(fname, 0)
    if not sys.platform.startswith('win'):
        with pytest.raises(PermissionError, match='read permissions'):
            _check_fname(fname, overwrite='read', must_exist=True)
    os.chmod(fname, orig_perms)
    os.remove(fname)
    assert not op.isfile(fname)
    pytest.raises(IOError, check_fname, 'foo', 'tets-dip.x', (), ('.fif',))
    pytest.raises(ValueError, _check_subject, None, None)
    pytest.raises(TypeError, _check_subject, None, 1)
    pytest.raises(TypeError, _check_subject, 1, None)
    # smoke tests for permitted types
    check_random_state(None).choice(1)
    check_random_state(0).choice(1)
    check_random_state(np.random.RandomState(0)).choice(1)
    if check_version('numpy', '1.17'):
        check_random_state(np.random.default_rng(0)).choice(1)


@testing.requires_testing_data
@pytest.mark.parametrize('suffix',
                         ('_meg.fif', '_eeg.fif', '_ieeg.fif',
                          '_meg.fif.gz', '_eeg.fif.gz', '_ieeg.fif.gz'))
def test_check_fname_suffixes(suffix, tmp_path):
    """Test checking for valid filename suffixes."""
    new_fname = tmp_path / op.basename(fname_raw).replace('_raw.fif', suffix)
    raw = mne.io.read_raw_fif(fname_raw).crop(0, 0.1)
    raw.save(new_fname)
    mne.io.read_raw_fif(new_fname)


@requires_mayavi
@traits_test
def test_check_mayavi():
    """Test mayavi version check."""
    pytest.raises(RuntimeError, _check_mayavi_version, '100.0.0')


def _get_data():
    """Read in data used in tests."""
    # read forward model
    forward = mne.read_forward_solution(fname_fwd)
    # read data
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    events = mne.read_events(fname_event)
    event_id, tmin, tmax = 1, -0.1, 0.15

    # decimate for speed
    left_temporal_channels = read_vectorview_selection('Left-temporal')
    picks = mne.pick_types(raw.info, meg=True,
                           selection=left_temporal_channels)
    picks = picks[::2]
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    del picks

    raw.info.normalize_proj()  # avoid projection warnings

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(None, 0.), preload=True, reject=reject)

    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.)

    data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.15)

    return epochs, data_cov, noise_cov, forward


@testing.requires_testing_data
def test_check_info_inv():
    """Test checks for common channels across fwd model and cov matrices."""
    epochs, data_cov, noise_cov, forward = _get_data()

    # make sure same channel lists exist in data to make testing life easier
    assert epochs.info['ch_names'] == data_cov.ch_names
    assert epochs.info['ch_names'] == noise_cov.ch_names

    # check whether bad channels get excluded from the channel selection
    # info
    info_bads = epochs.info.copy()
    info_bads['bads'] = info_bads['ch_names'][1:3]  # include two bad channels
    picks = _check_info_inv(info_bads, forward, noise_cov=noise_cov)
    assert [1, 2] not in picks
    # covariance matrix
    data_cov_bads = data_cov.copy()
    data_cov_bads['bads'] = data_cov_bads.ch_names[0]
    picks = _check_info_inv(epochs.info, forward, data_cov=data_cov_bads)
    assert 0 not in picks
    # noise covariance matrix
    noise_cov_bads = noise_cov.copy()
    noise_cov_bads['bads'] = noise_cov_bads.ch_names[1]
    picks = _check_info_inv(epochs.info, forward, noise_cov=noise_cov_bads)
    assert 1 not in picks

    # test whether reference channels get deleted
    info_ref = epochs.info.copy()
    info_ref['chs'][0]['kind'] = 301  # pretend to have a ref channel
    picks = _check_info_inv(info_ref, forward, noise_cov=noise_cov)
    assert 0 not in picks

    # pick channels in all inputs and make sure common set is returned
    epochs.pick_channels([epochs.ch_names[ii] for ii in range(10)])
    data_cov = pick_channels_cov(data_cov, include=[data_cov.ch_names[ii]
                                                    for ii in range(5, 20)])
    noise_cov = pick_channels_cov(noise_cov, include=[noise_cov.ch_names[ii]
                                                      for ii in range(7, 12)])
    picks = _check_info_inv(epochs.info, forward, noise_cov=noise_cov,
                            data_cov=data_cov)
    assert list(range(7, 10)) == picks


def test_check_option():
    """Test checking the value of a parameter against a list of options."""
    allowed_values = ['valid', 'good', 'ok']

    # Value is allowed
    assert _check_option('option', 'valid', allowed_values)
    assert _check_option('option', 'good', allowed_values)
    assert _check_option('option', 'ok', allowed_values)
    assert _check_option('option', 'valid', ['valid'])

    # Check error message for invalid value
    msg = ("Invalid value for the 'option' parameter. Allowed values are "
           "'valid', 'good', and 'ok', but got 'bad' instead.")
    with pytest.raises(ValueError, match=msg):
        assert _check_option('option', 'bad', allowed_values)

    # Special error message if only one value is allowed
    msg = ("Invalid value for the 'option' parameter. The only allowed value "
           "is 'valid', but got 'bad' instead.")
    with pytest.raises(ValueError, match=msg):
        assert _check_option('option', 'bad', ['valid'])


def test_path_like():
    """Test _path_like()."""
    str_path = str(base_dir)
    pathlib_path = Path(base_dir)
    no_path = dict(foo='bar')

    assert _path_like(str_path) is True
    assert _path_like(pathlib_path) is True
    assert _path_like(no_path) is False


def test_validate_type():
    """Test _validate_type."""
    _validate_type(1, 'int-like')
    with pytest.raises(TypeError, match='int-like'):
        _validate_type(False, 'int-like')


@requires_nibabel()
@testing.requires_testing_data
def test_suggest():
    """Test suggestions."""
    names = mne.get_volume_labels_from_aseg(fname_mgz)
    sug = _suggest('', names)
    assert sug == ''  # nothing
    sug = _suggest('Left-cerebellum', names)
    assert sug == " Did you mean 'Left-Cerebellum-Cortex'?"
    sug = _suggest('Cerebellum-Cortex', names)
    assert sug == " Did you mean one of ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex', 'Left-Cerebral-Cortex']?"  # noqa: E501


def test_on_missing():
    """Test _on_missing."""
    msg = 'test'
    with pytest.raises(ValueError, match=msg):
        _on_missing('raise', msg)
    with pytest.warns(RuntimeWarning, match=msg):
        _on_missing('warn', msg)
    _on_missing('ignore', msg)

    with pytest.raises(ValueError,
                       match='Invalid value for the \'on_missing\' parameter'):
        _on_missing('foo', msg)


def _matlab_input(msg):
    raise EOFError()


def test_safe_input(monkeypatch):
    """Test _safe_input."""
    monkeypatch.setattr(mne.utils.check, 'input', _matlab_input)
    with pytest.raises(RuntimeError, match='Could not use input'):
        _safe_input('whatever', alt='nothing')
    assert _safe_input('whatever', use='nothing') == 'nothing'


@testing.requires_testing_data
def test_check_ch_locs():
    """Test _check_ch_locs behavior."""
    info = mne.io.read_info(fname_raw)
    assert _check_ch_locs(info=info)

    for picks in ([0], [0, 1], None):
        assert _check_ch_locs(info=info, picks=picks)

    for ch_type in ('meg', 'mag', 'grad', 'eeg'):
        assert _check_ch_locs(info=info, ch_type=ch_type)

    # drop locations for EEG
    picks_eeg = _picks_to_idx(info=info, picks='eeg')
    for idx in picks_eeg:
        info['chs'][idx]['loc'][:3] = np.nan

    # EEG tests should fail now
    assert _check_ch_locs(info=info, picks=picks_eeg) is False
    assert _check_ch_locs(info=info, ch_type='eeg') is False

    # tests for other (and "all") channels should still pass
    assert _check_ch_locs(info=info)
    assert _check_ch_locs(info=info, ch_type='mag')
