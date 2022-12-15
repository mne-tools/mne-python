# Authors: Yuval Realpe <yuval.realpe@gmail.com>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np
from scipy import linalg
from random import choice

import mne
from mne.beamformer import alternating_projections
from mne.cov import regularize
from mne.datasets import testing
from mne.beamformer.tests.test_rap_music import simu_data

data_path = testing.data_path(download=False)
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')


def _get_data(ch_decim=1):
    """Read in data used in tests."""
    # Read evoked
    condition = choice(['Left Auditory',
                        'Right Auditory',
                        'Left visual',
                        'Right visual'])
    evoked = mne.read_evokeds(fname_ave, condition=condition,
                              baseline=(None, 0))
    evoked.crop(tmin=0.05, tmax=0.15)
    picks = mne.pick_types(evoked.info, meg=True, eeg=False)
    picks = picks[::ch_decim]
    evoked.pick_channels([evoked.ch_names[pick] for pick in picks])
    evoked.info.normalize_proj()

    # Read noise_cov
    noise_cov = mne.read_cov(fname_cov)
    noise_cov['projs'] = []
    noise_cov = regularize(noise_cov, evoked.info, rank='full', proj=False)

    return evoked, noise_cov


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
                                 np.array([ori1, ori2]).T))) > 0.9)

    assert (np.max(np.abs(np.dot(dipoles[1].ori[0],
                                 np.array([ori1, ori2]).T))) > 0.9)

    if residual is not None:
        picks_grad = mne.pick_types(residual.info, meg='grad')
        picks_mag = mne.pick_types(residual.info, meg='mag')
        rel_tol = 0.15
        for picks in [picks_grad, picks_mag]:
            assert (linalg.norm(residual.data[picks], ord='fro') <
                    rel_tol * linalg.norm(evoked.data[picks], ord='fro'))


@testing.requires_testing_data
def test_ap_simulated():
    """Test AP with simulated evoked."""
    evoked, noise_cov = _get_data(ch_decim=16)
    forward = mne.read_forward_solution(fname_fwd)
    forward = mne.pick_channels_forward(forward, evoked.ch_names)
    forward_surf_ori = mne.convert_forward_solution(forward, surf_ori=True)
    forward_fixed = mne.convert_forward_solution(forward, force_fixed=True,
                                                 surf_ori=True, use_cps=True)

    nsources = 2
    sim_evoked, stc = simu_data(evoked, forward_fixed, noise_cov,
                                evoked.times, nave=evoked.nave)
    # Check dipoles for fixed ori
    dipoles, _, _, var_exp = alternating_projections(sim_evoked,
                                                     forward_fixed,
                                                     nsources,
                                                     noise_cov,
                                                     verbose=True)
    assert 92 < var_exp < 96
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked)
    assert 97 < dipoles[0].gof.max() < 100
    assert 97 < dipoles[1].gof.max() < 100
    assert dipoles[0].gof.min() >= 0.
    assert dipoles[1].gof.min() >= 0.

    nave = 100000  # add a tiny amount of noise to the simulated evokeds
    sim_evoked, stc = simu_data(evoked, forward_fixed, noise_cov,
                                evoked.times, nave=nave)
    dipoles, residual, _, _ = alternating_projections(sim_evoked,
                                                      forward_fixed,
                                                      nsources,
                                                      noise_cov)

    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked, residual)

    # Check dipoles for free ori
    dipoles, residual, _, _ = alternating_projections(sim_evoked,
                                                      forward,
                                                      nsources,
                                                      noise_cov)
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked, residual)

    # Check dipoles for free surface ori
    dipoles, residual, _, _ = alternating_projections(sim_evoked,
                                                      forward_surf_ori,
                                                      nsources,
                                                      noise_cov)
    _check_dipoles(dipoles, forward_fixed, stc, sim_evoked, residual)


@testing.requires_testing_data
def test_ap_picks():
    """Test AP with picking."""
    evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',
                              baseline=(None, 0))
    evoked.crop(tmin=0.05, tmax=0.15)  # select N100
    evoked.pick_types(meg=True, eeg=False)
    forward = mne.read_forward_solution(fname_fwd)
    noise_cov = mne.read_cov(fname_cov)
    nsources = 2
    dipoles = alternating_projections(evoked, forward, nsources,
                                      noise_cov, return_residual=False)
    assert len(dipoles) == 2
