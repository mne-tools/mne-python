from copy import deepcopy
import os.path as op
import pytest

import mne
from mne.datasets import testing
from mne.utils import (check_random_state, _check_fname, check_fname,
                       _check_subject, requires_mayavi, traits_test,
                       _check_mayavi_version)

data_path = testing.data_path(download=False)
base_dir = op.join(data_path, 'MEG', 'sample')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_event = op.join(base_dir, 'sample_audvis_trunc_raw-eve.fif')
fname_fwd = op.join(base_dir, 'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_cov = op.join(base_dir, 'sample_audivis_trunc_cov.fif')
reject = dict(grad=4000e-13, mag=4e-12)


def test_check():
    """Test checking functions."""
    pytest.raises(ValueError, check_random_state, 'foo')
    pytest.raises(TypeError, _check_fname, 1)
    pytest.raises(IOError, check_fname, 'foo', 'tets-dip.x', (), ('.fif',))
    pytest.raises(ValueError, _check_subject, None, None)
    pytest.raises(TypeError, _check_subject, None, 1)
    pytest.raises(TypeError, _check_subject, 1, None)


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
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, selection=left_temporal_channels)
    picks = picks[::2]
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    del picks

    raw.info.normalize_proj()  # avoid projection warnings

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=(None, 0.), preload=True, reject=reject)

    noise_cov = mne.read_cov(fname_cov)
    noise_cov['projs'] = []

    data_cov = mne.compute_covariance(epochs, tin=0.01, tmax=0.15)

    return epochs, data_cov, noise_cov, forward


@testing.requires_testing_data
def test_check_info_inv():
    """Test checks for common channels acros fwd model and cov matrices."""
    epochs, data_cov, noise_cov, forward = _get_data()

    # test whether reference channels get deleted
    info_ref = deepcopy(epochs.info)
    info_ref['chs']
