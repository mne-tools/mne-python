# Authors: Yuval Realpe <yuval.realpe@gmail.com>
#
# License: BSD-3-Clause

import os.path as op

import mne
from mne.beamformer import alternating_projections
from mne.datasets import testing
from mne.beamformer.tests.test_rap_music import simu_data, _check_dipoles, _get_data

data_path = testing.data_path(download=False)
fname_ave = op.join(data_path, "MEG", "sample", "sample_audvis-ave.fif")
fname_cov = op.join(data_path, "MEG", "sample", "sample_audvis-cov.fif")
fname_fwd = op.join(
    data_path, "MEG", "sample", "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif"
)


@testing.requires_testing_data
def test_ap_simulated():
    """Test AP with simulated evoked."""
    evoked, noise_cov = _get_data(ch_decim=16)
    forward = mne.read_forward_solution(fname_fwd)
    forward = mne.pick_channels_forward(forward, evoked.ch_names)
    forward_surf_ori = mne.convert_forward_solution(forward, surf_ori=True)
    forward_fixed = mne.convert_forward_solution(
        forward, force_fixed=True, surf_ori=True, use_cps=True
    )

    n_sources = 2
    sim_evoked, stc = simu_data(
        evoked, forward_fixed, noise_cov, evoked.times, nave=evoked.nave
    )
    # Check dipoles for fixed ori
    dipoles, _, _, var_exp = alternating_projections(
        sim_evoked, forward_fixed, n_sources, noise_cov, verbose=True
    )
    assert 92 < var_exp < 96
    _check_dipoles(
        dipoles, forward_fixed, stc, sim_evoked, rel_tol=0.027, ori_check=0.95
    )
    assert 97 < dipoles[0].gof.max() < 100
    assert 97 < dipoles[1].gof.max() < 100
    assert dipoles[0].gof.min() >= 0.0
    assert dipoles[1].gof.min() >= 0.0

    nave = 100000  # add a tiny amount of noise to the simulated evokeds
    sim_evoked, stc = simu_data(
        evoked, forward_fixed, noise_cov, evoked.times, nave=nave
    )
    dipoles, residual, _, _ = alternating_projections(
        sim_evoked, forward_fixed, n_sources, noise_cov
    )

    _check_dipoles(
        dipoles, forward_fixed, stc, sim_evoked, residual, rel_tol=0.08, ori_check=0.7
    )

    # Check dipoles for free ori
    dipoles, residual, _, _ = alternating_projections(
        sim_evoked, forward, n_sources, noise_cov
    )
    _check_dipoles(
        dipoles, forward_fixed, stc, sim_evoked, residual, rel_tol=0.08, ori_check=0.7
    )

    # Check dipoles for free surface ori
    dipoles, residual, _, _ = alternating_projections(
        sim_evoked, forward_surf_ori, n_sources, noise_cov
    )
    _check_dipoles(
        dipoles, forward_fixed, stc, sim_evoked, residual, rel_tol=0.08, ori_check=0.7
    )


@testing.requires_testing_data
def test_ap_picks():
    """Test AP with picking."""
    evoked = mne.read_evokeds(fname_ave, condition="Right Auditory", baseline=(None, 0))
    evoked.crop(tmin=0.05, tmax=0.15)  # select N100
    evoked.pick_types(meg=True, eeg=False)
    forward = mne.read_forward_solution(fname_fwd)
    noise_cov = mne.read_cov(fname_cov)
    n_sources = 2
    dipoles = alternating_projections(
        evoked, forward, n_sources, noise_cov, return_residual=False
    )
    assert len(dipoles) == 2
