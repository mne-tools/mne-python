# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

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


def read_forward_solution_meg(fname_fwd, ch_names, **kwargs):
    fwd = mne.read_forward_solution(fname_fwd, **kwargs)
    return mne.pick_types_forward(fwd, eeg=False, include=ch_names,
                                  exclude=['MEG 2443'])


def _get_data(tmin=-0.1, tmax=0.15, event_id=1):
    """Read in data used in tests
    """
    events = mne.read_events(fname_event)
    raw = mne.io.Raw(fname_raw, preload=True)

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False,
                           eog=True, ref_meg=False, exclude='bads')

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=False,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))

    evoked = epochs.average()

    ch_names = evoked.info['ch_names']
    forward = mne.read_forward_solution(fname_fwd, ch_names)

    forward_surf_ori = read_forward_solution_meg(fname_fwd, ch_names,
                                                 surf_ori=True)
    forward_fixed = read_forward_solution_meg(fname_fwd, ch_names,
                                              force_fixed=True,
                                              surf_ori=True)

    noise_cov = mne.read_cov(fname_cov)

    return evoked, noise_cov, forward, forward_surf_ori, forward_fixed


def simu_data(evoked, forward, noise_cov, n_dipoles, times):
    # Generate the two dipoles data
    mu, sigma = 0.1, 0.005
    s1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 /
                                                   (2 * sigma ** 2))

    mu, sigma = 0.075, 0.008
    s2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 /
                                                   (2 * sigma ** 2))
    data = np.array([s1, s2]) * 10e-10

    src = forward['src']
    rndi = np.random.randint(len(src[0]['vertno']))
    lh_vertno = src[0]['vertno'][[rndi]]
    rndi = np.random.randint(len(src[1]['vertno']))
    rh_vertno = src[1]['vertno'][[rndi]]

    vertices = [lh_vertno, rh_vertno]
    tmin, tstep = times.min(), 1 / evoked.info['sfreq']
    stc = mne.SourceEstimate(data, vertices=vertices, tmin=tmin, tstep=tstep)

    rng = np.random.RandomState(0)
    sim_evoked = mne.simulation.generate_evoked(forward, stc, evoked,
                                                noise_cov, snr=20,
                                                random_state=rng)

    return sim_evoked, stc


@slow_test
@testing.requires_testing_data
def test_rap_music():
    """Test RAP-MUSIC with evoked data
    """
    evoked, noise_cov, forward, forward_surf_ori, forward_fixed =\
        _get_data()

    def _check_dipoles(dipoles):
        assert_true(len(dipoles), n_dipoles)
        n_times = len(dipoles[0].times)
        assert_true(dipoles[0].pos.shape[0], n_times)
        assert_true(dipoles[0].ori.shape[0], n_times)

        assert_true(dipoles[0].pos.shape[1], 3)
        assert_true(dipoles[0].ori.shape[1], 3)

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


@slow_test
@testing.requires_testing_data
def test_rap_music_simulated():
    """Test RAP-MUSIC with simulated evoked
    """
    evoked, noise_cov, forward, forward_surf_ori, forward_fixed =\
        _get_data()

    n_dipoles = 2
    sim_evoked, stc = simu_data(evoked, forward_fixed, noise_cov, n_dipoles,
                                evoked.times)

    def _check_dipoles(dipoles, ori=False):
        src = forward_fixed['src']
        pos1 = forward_fixed['source_rr'][np.where(src[0]['vertno'] ==
                                          stc.vertices[0])]
        pos2 = forward_fixed['source_rr'][np.where(src[1]['vertno'] ==
                                          stc.vertices[1])[0] +
                                          len(src[0]['vertno'])]

        # Check the position of the two dipoles
        assert_true(dipoles[0].pos[0] in np.array([pos1, pos2]))
        assert_true(dipoles[1].pos[0] in np.array([pos1, pos2]))

        ori1 = forward_fixed['source_nn'][np.where(src[0]['vertno'] ==
                                          stc.vertices[0])]
        ori2 = forward_fixed['source_nn'][np.where(src[1]['vertno'] ==
                                          stc.vertices[1])[0] +
                                          len(src[0]['vertno'])]

        if ori:
            # Check the orientation of the two dipoles
            assert_true(dipoles[0].ori[0] in np.array([ori1, ori2]))
            assert_true(dipoles[1].ori[0] in np.array([ori1, ori2]))

    # Check dipoles for fixed ori
    dipoles = rap_music(sim_evoked, forward_fixed, noise_cov,
                        n_dipoles=n_dipoles)
    _check_dipoles(dipoles, ori=True)

    # Check dipoles for free ori
    dipoles = rap_music(sim_evoked, forward_fixed, noise_cov,
                        n_dipoles=n_dipoles)
    _check_dipoles(dipoles)
