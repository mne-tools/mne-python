# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op

import warnings
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal


import mne
from mne.datasets import testing
from mne.beamformer import rap_music
from mne.utils import slow_test


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


def _get_data(tmin=-0.1, tmax=0.15, event_id=1):
    """Read in data used in tests
    """
    events = mne.read_events(fname_event)
    raw = mne.io.Raw(fname_raw, preload=True)
    forward = mne.read_forward_solution(fname_fwd)

    forward_surf_ori = read_forward_solution_meg(fname_fwd, surf_ori=True)
    forward_fixed = read_forward_solution_meg(fname_fwd, force_fixed=True,
                                              surf_ori=True)

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True,
                           eog=True, ref_meg=False, exclude='bads')

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=False,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

    evoked = epochs.average()
    info = evoked.info

    noise_cov = mne.read_cov(fname_cov)

    return evoked, noise_cov, forward, forward_surf_ori, forward_fixed


@slow_test
@testing.requires_testing_data
def test_rap_music():
    """Test RAP-MUSIC with evoked data
    """
    evoked, noise_cov, forward, forward_surf_ori, forward_fixed =\
        _get_data()

    def _check_dipoles(dipoles):
        assert_true(len(dipoles), n_dipoles)
        assert_true(len(dipoles[0]['pos']), 3)
        assert_true(len(dipoles[0]['ori']), 3)

    n_dipoles = 2

    dipoles = rap_music(evoked, forward, noise_cov, n_dipoles=n_dipoles)
    _check_dipoles(dipoles)

    # Test with fixed forward
    dipoles_fixed, res = rap_music(evoked, forward_surf_ori, noise_cov,
                                   n_dipoles=n_dipoles,
                                   return_residual=True)
    _check_dipoles(dipoles_fixed)

    # Test the residual times
    assert_array_almost_equal(evoked.times, res.times)
