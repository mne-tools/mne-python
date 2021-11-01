# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import os.path as op

import pytest
import numpy as np
from scipy import linalg
from numpy.testing import assert_allclose

import mne
from mne.beamformer import rap_music
from mne.cov import regularize
from mne.datasets import testing
from mne.minimum_norm.tests.test_inverse import assert_var_exp_log
from mne.utils import catch_logging


data_path = testing.data_path(download=False)
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')


def _get_data(ch_decim=1):
    """Read in data used in tests."""
    # Read evoked
    evoked = mne.read_evokeds(fname_ave, 0, baseline=(None, 0))
    evoked.info['bads'] = ['MEG 2443']
    with evoked.info._unlock():
        evoked.info['lowpass'] = 16  # fake for decim
    evoked.decimate(12)
    evoked.crop(0.0, 0.3)
    picks = mne.pick_types(evoked.info, meg=True, eeg=False)
    picks = picks[::ch_decim]
    evoked.pick_channels([evoked.ch_names[pick] for pick in picks])
    evoked.info.normalize_proj()

    noise_cov = mne.read_cov(fname_cov)
    noise_cov['projs'] = []
    noise_cov = regularize(noise_cov, evoked.info, rank='full', proj=False)
    return evoked, noise_cov


def simu_data(evoked, forward, noise_cov, n_dipoles, times, nave=1):
    """Simulate an evoked dataset with 2 sources.

    One source is put in each hemisphere.
    """
    # Generate the two dipoles data
    mu, sigma = 0.1, 0.005
    s1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 /
                                                   (2 * sigma ** 2))

    mu, sigma = 0.075, 0.008
    s2 = -1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(times - mu) ** 2 /
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
                                                noise_cov, nave=nave,
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
    assert (dipoles[0].pos[0] in np.array([pos1, pos2]))
    assert (dipoles[1].pos[0] in np.array([pos1, pos2]))

    ori1 = fwd['source_nn'][np.where(src[0]['vertno'] ==
                                     stc.vertices[0])[0]][0]
    ori2 = fwd['source_nn'][np.where(src[1]['vertno'] ==
                                     stc.vertices[1])[0] +
                            len(src[0]['vertno'])][0]

    # Check the orientation of the dipoles
    assert (np.max(np.abs(np.dot(dipoles[0].ori[0],
                                 np.array([ori1, ori2]).T))) > 0.99)

    assert (np.max(np.abs(np.dot(dipoles[1].ori[0],
                                 np.array([ori1, ori2]).T))) > 0.99)

    if residual is not None:
        picks_grad = mne.pick_types(residual.info, meg='grad')
        picks_mag = mne.pick_types(residual.info, meg='mag')
        rel_tol = 0.02
        for picks in [picks_grad, picks_mag]:
            assert (linalg.norm(residual.data[picks], ord='fro') <
                    rel_tol * linalg.norm(evoked.data[picks], ord='fro'))


@testing.requires_testing_data
def test_rap_music_simulated():
    """Test RAP-MUSIC with simulated evoked."""
    evoked, noise_cov = _get_data(ch_decim=16)
    forward = mne.read_forward_solution(fname_fwd)
    forward = mne.pick_channels_forward(forward, evoked.ch_names)
    forward_surf_ori = mne.convert_forward_solution(forward, surf_ori=True)
    forward_fixed = mne.convert_forward_solution(forward, force_fixed=True,
                                                 surf_ori=True, use_cps=True)

    n_dipoles = 2
    sim_evoked, stc = simu_data(evoked, forward_fixed, noise_cov,
                                n_dipoles, evoked.times, nave=evoked.nave)
    # Check dipoles for fixed ori
    with catch_logging() as log:
        dipoles = rap_music(sim_evoked, forward_fixed, noise_cov,
                            n_dipoles=n_dipoles, verbose=True)
    assert_var_exp_log(log.getvalue(), 89, 91)
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked)
    assert 97 < dipoles[0].gof.max() < 100
    assert 91 < dipoles[1].gof.max() < 93
    assert dipoles[0].gof.min() >= 0.

    nave = 100000  # add a tiny amount of noise to the simulated evokeds
    sim_evoked, stc = simu_data(evoked, forward_fixed, noise_cov,
                                n_dipoles, evoked.times, nave=nave)
    dipoles, residual = rap_music(sim_evoked, forward_fixed, noise_cov,
                                  n_dipoles=n_dipoles, return_residual=True)
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked, residual)

    # Check dipoles for free ori
    dipoles, residual = rap_music(sim_evoked, forward, noise_cov,
                                  n_dipoles=n_dipoles, return_residual=True)
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked, residual)

    # Check dipoles for free surface ori
    dipoles, residual = rap_music(sim_evoked, forward_surf_ori, noise_cov,
                                  n_dipoles=n_dipoles, return_residual=True)
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked, residual)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_rap_music_sphere():
    """Test RAP-MUSIC with real data, sphere model, MEG only."""
    evoked, noise_cov = _get_data(ch_decim=8)
    sphere = mne.make_sphere_model(r0=(0., 0., 0.04))
    src = mne.setup_volume_source_space(subject=None, pos=10.,
                                        sphere=(0.0, 0.0, 40, 65.0),
                                        mindist=5.0, exclude=0.0,
                                        sphere_units='mm')
    forward = mne.make_forward_solution(evoked.info, trans=None, src=src,
                                        bem=sphere)

    with catch_logging() as log:
        dipoles = rap_music(evoked, forward, noise_cov, n_dipoles=2,
                            verbose=True)
    assert_var_exp_log(log.getvalue(), 47, 49)
    # Test that there is one dipole on each hemisphere
    pos = np.array([dip.pos[0] for dip in dipoles])
    assert pos.shape == (2, 3)
    assert (pos[:, 0] < 0).sum() == 1
    assert (pos[:, 0] > 0).sum() == 1
    # Check the amplitude scale
    assert (1e-10 < dipoles[0].amplitude[0] < 1e-7)
    # Check the orientation
    dip_fit = mne.fit_dipole(evoked, noise_cov, sphere)[0]
    assert (np.max(np.abs(np.dot(dip_fit.ori, dipoles[0].ori[0]))) > 0.99)
    assert (np.max(np.abs(np.dot(dip_fit.ori, dipoles[1].ori[0]))) > 0.99)
    idx = dip_fit.gof.argmax()
    dist = np.linalg.norm(dipoles[0].pos[idx] - dip_fit.pos[idx])
    assert 0.004 <= dist < 0.007
    assert_allclose(dipoles[0].gof[idx], dip_fit.gof[idx], atol=3)


@testing.requires_testing_data
def test_rap_music_picks():
    """Test RAP-MUSIC with picking."""
    evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',
                              baseline=(None, 0))
    evoked.crop(tmin=0.05, tmax=0.15)  # select N100
    evoked.pick_types(meg=True, eeg=False)
    forward = mne.read_forward_solution(fname_fwd)
    noise_cov = mne.read_cov(fname_cov)
    dipoles = rap_music(evoked, forward, noise_cov, n_dipoles=2)
    assert len(dipoles) == 2
