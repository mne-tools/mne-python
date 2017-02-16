# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from scipy import linalg

import warnings
from nose.tools import assert_true

import mne
from mne.datasets import testing
from mne.beamformer import rap_music
from mne.utils import run_tests_if_main


data_path = testing.data_path(download=False)
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')

warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _read_forward_solution_meg(fname_fwd, **kwargs):
    fwd = mne.read_forward_solution(fname_fwd, **kwargs)
    return mne.pick_types_forward(fwd, meg=True, eeg=False,
                                  exclude=['MEG 2443'])


def _get_data(event_id=1):
    """Read in data used in tests
    """
    # Read evoked
    evoked = mne.read_evokeds(fname_ave, event_id)
    evoked.pick_types(meg=True, eeg=False)
    evoked.crop(0, 0.3)

    forward = mne.read_forward_solution(fname_fwd)

    forward_surf_ori = _read_forward_solution_meg(fname_fwd, surf_ori=True)
    forward_fixed = _read_forward_solution_meg(fname_fwd, force_fixed=True,
                                               surf_ori=True)

    noise_cov = mne.read_cov(fname_cov)

    return evoked, noise_cov, forward, forward_surf_ori, forward_fixed


def simu_data(evoked, forward, noise_cov, n_dipoles, times):
    """Simulate an evoked dataset with 2 sources

    One source is put in each hemisphere.
    """
    # Generate the two dipoles data
    mu, sigma = 0.1, 0.005
    s1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 /
                                                   (2 * sigma ** 2))

    mu, sigma = 0.075, 0.008
    s2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 /
                                                   (2 * sigma ** 2))
    data = np.array([s1, s2]) * 1e-9

    src = forward['src']
    rng = np.random.RandomState(42)

    rndi = rng.randint(len(src[0]['vertno']))
    lh_vertno = src[0]['vertno'][[rndi]]

    rndi = rng.randint(len(src[1]['vertno']))
    rh_vertno = src[1]['vertno'][[rndi]]

    vertices = [lh_vertno, rh_vertno]
    tmin, tstep = times.min(), 1 / evoked.info['sfreq']
    stc = mne.SourceEstimate(data, vertices=vertices, tmin=tmin, tstep=tstep)

    sim_evoked = mne.simulation.simulate_evoked(forward, stc, evoked.info,
                                                noise_cov, snr=20,
                                                random_state=rng)

    return sim_evoked, stc


def _check_dipoles(dipoles, fwd, stc, evoked, residual=None):
    src = fwd['src']
    pos1 = fwd['source_rr'][np.where(src[0]['vertno'] ==
                                     stc.vertices[0])]
    pos2 = fwd['source_rr'][np.where(src[1]['vertno'] ==
                                     stc.vertices[1])[0] +
                            len(src[0]['vertno'])]

    # Check the position of the two dipoles
    assert_true(dipoles[0].pos[0] in np.array([pos1, pos2]))
    assert_true(dipoles[1].pos[0] in np.array([pos1, pos2]))

    ori1 = fwd['source_nn'][np.where(src[0]['vertno'] ==
                                     stc.vertices[0])[0]][0]
    ori2 = fwd['source_nn'][np.where(src[1]['vertno'] ==
                                     stc.vertices[1])[0] +
                            len(src[0]['vertno'])][0]

    # Check the orientation of the dipoles
    assert_true(np.max(np.abs(np.dot(dipoles[0].ori[0],
                                     np.array([ori1, ori2]).T))) > 0.99)

    assert_true(np.max(np.abs(np.dot(dipoles[1].ori[0],
                                     np.array([ori1, ori2]).T))) > 0.99)

    if residual is not None:
        picks_grad = mne.pick_types(residual.info, meg='grad')
        picks_mag = mne.pick_types(residual.info, meg='mag')
        rel_tol = 0.02
        for picks in [picks_grad, picks_mag]:
            assert_true(linalg.norm(residual.data[picks], ord='fro') <
                        rel_tol *
                        linalg.norm(evoked.data[picks], ord='fro'))


@testing.requires_testing_data
def test_rap_music_simulated():
    """Test RAP-MUSIC with simulated evoked
    """
    evoked, noise_cov, forward, forward_surf_ori, forward_fixed =\
        _get_data()

    n_dipoles = 2
    sim_evoked, stc = simu_data(evoked, forward_fixed, noise_cov,
                                n_dipoles, evoked.times)
    # Check dipoles for fixed ori
    dipoles = rap_music(sim_evoked, forward_fixed, noise_cov,
                        n_dipoles=n_dipoles)
    _check_dipoles(dipoles, forward_fixed, stc, evoked)

    dipoles, residual = rap_music(sim_evoked, forward_fixed, noise_cov,
                                  n_dipoles=n_dipoles, return_residual=True)
    _check_dipoles(dipoles, forward_fixed, stc, evoked, residual)

    # Check dipoles for free ori
    dipoles, residual = rap_music(sim_evoked, forward, noise_cov,
                                  n_dipoles=n_dipoles, return_residual=True)
    _check_dipoles(dipoles, forward_fixed, stc, evoked, residual)

    # Check dipoles for free surface ori
    dipoles, residual = rap_music(sim_evoked, forward_surf_ori, noise_cov,
                                  n_dipoles=n_dipoles, return_residual=True)
    _check_dipoles(dipoles, forward_fixed, stc, evoked, residual)


@testing.requires_testing_data
def test_rap_music_simulated_sphere():
    """Test RAP-MUSIC with sphere model and MEG only."""
    noise_cov = mne.read_cov(fname_cov)
    evoked = mne.read_evokeds(fname_ave, baseline=(None, 0))[0]

    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.070)
    src = mne.setup_volume_source_space(subject=None, fname=None, pos=10.,
                                        sphere=(0.0, 0.0, 0.0, 65.0),
                                        mindist=5.0, exclude=0.0)
    forward = mne.make_forward_solution(evoked.info, trans=None, src=src,
                                        bem=sphere, eeg=False, meg=True)

    evoked.pick_types(meg=True)
    evoked.crop(0.0, 0.3)

    n_dipoles = 2
    dipoles = rap_music(evoked, forward, noise_cov, n_dipoles=n_dipoles)
    # Test that there is one dipole on each hemisphere
    assert_true(dipoles[0].pos[0, 0] < 0.)
    assert_true(dipoles[1].pos[0, 0] > 0.)

run_tests_if_main()
