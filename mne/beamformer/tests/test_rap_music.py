import os.path as op

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import warnings

import mne
from mne.datasets import testing
from mne.beamformer import rap_music
from mne.externals.six import advance_iterator
from mne.utils import run_tests_if_main, slow_test


data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def read_forward_solution_meg(*args, **kwargs):
    fwd = mne.read_forward_solution(*args, **kwargs)
    return mne.pick_types_forward(fwd, meg=True, eeg=False)


def _get_data(tmin=-0.1, tmax=0.15, all_forward=True):
    """Read in data used in tests
    """
    events = mne.read_events(fname_event)
    raw = mne.io.Raw(fname_raw, preload=True)
    forward = mne.read_forward_solution(fname_fwd)
    if all_forward:
        forward_surf_ori = read_forward_solution_meg(fname_fwd, surf_ori=True)
        forward_fixed = read_forward_solution_meg(fname_fwd, force_fixed=True,
                                                  surf_ori=True)
    else:
        forward_surf_ori = None
        forward_fixed = None

    event_id, tmin, tmax = 1, tmin, tmax

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True,
                           eog=True, ref_meg=False, exclude='bads')

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0),
                        preload=False,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

    evoked = epochs.average()
    info = evoked.info

    noise_cov = mne.read_cov(fname_cov)
    noise_cov = mne.cov.regularize(noise_cov, info, mag=0.05, grad=0.05,
                                   eeg=0.1, proj=True)

    return evoked, noise_cov, forward, forward_surf_ori, forward_fixed


@slow_test
@testing.requires_testing_data
def test_rap_music():
    """Test RAP-MUSIC with evoked data
    """
    evoked, noise_cov, forward, forward_surf_ori, forward_fixed =\
        _get_data()

    n_sources = 4
    stc = rap_music(evoked, forward, noise_cov, n_sources=n_sources)
    stc.crop(0.02, None)

    assert_true(stc.data.shape[0], n_sources)
    assert_true(stc.vertices[0].shape > 0)
    assert_true(stc.vertices[1].shape > 0)

    stc_pow = np.sum(stc.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = np.abs(stc.data[idx])
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.09 < tmax < 0.105, tmax)

    # Test picking normal orientation (surface source space only)
    stc_normal = rap_music(evoked, forward_surf_ori, noise_cov,
                           n_sources=n_sources, pick_ori="normal")
    stc_normal.crop(0.02, None)

    assert_true(stc.data.shape[0], n_sources)
    assert_true(stc.vertices[0].shape > 0)
    assert_true(stc.vertices[1].shape > 0)

    stc_pow = np.sum(np.abs(stc_normal.data), axis=1)
    idx = np.argmax(stc_pow)
    max_stc = np.abs(stc_normal.data[idx])
    tmax = stc_normal.times[np.argmax(max_stc)]

    assert_true(0.04 < tmax < 0.11, tmax)

    # Test if fixed forward operator is detected when picking normal
    assert_raises(ValueError, rap_music, evoked, forward_fixed, noise_cov,
                  pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, rap_music, evoked, forward, noise_cov,
                  pick_ori="normal")

    # Test the residual times
    stc_normal, res = rap_music(evoked, forward_surf_ori, noise_cov,
                                n_sources=n_sources, return_residual=True)
    assert_array_almost_equal(evoked.times, res.times)
