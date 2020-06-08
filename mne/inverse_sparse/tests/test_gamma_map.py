# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import mne
from mne.datasets import testing
from mne import (read_cov, read_forward_solution, read_evokeds,
                 convert_forward_solution)
from mne.cov import regularize
from mne.inverse_sparse import gamma_map
from mne.inverse_sparse.mxne_inverse import make_stc_from_dipoles
from mne import pick_types_forward
from mne.utils import assert_stcs_equal, run_tests_if_main
from mne.dipole import Dipole

data_path = testing.data_path(download=False)
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')


def _check_stc(stc, evoked, idx, hemi, fwd, dist_limit=0., ratio=50.):
    """Check correctness."""
    assert_array_almost_equal(stc.times, evoked.times, 5)
    amps = np.sum(stc.data ** 2, axis=1)
    order = np.argsort(amps)[::-1]
    amps = amps[order]
    verts = np.concatenate(stc.vertices)[order]
    hemi_idx = int(order[0] >= len(stc.vertices[1]))
    hemis = ['lh', 'rh']
    assert hemis[hemi_idx] == hemi
    dist = np.linalg.norm(np.diff(fwd['src'][hemi_idx]['rr'][[idx, verts[0]]],
                                  axis=0)[0]) * 1000.
    assert dist <= dist_limit
    assert amps[0] > ratio * amps[1]


@pytest.mark.slowtest
@testing.requires_testing_data
def test_gamma_map():
    """Test Gamma MAP inverse."""
    forward = read_forward_solution(fname_fwd)
    forward = convert_forward_solution(forward, surf_ori=True)

    forward = pick_types_forward(forward, meg=False, eeg=True)
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0),
                          proj=False)
    evoked.resample(50, npad=100)
    evoked.crop(tmin=0.1, tmax=0.14)  # crop to window around peak

    cov = read_cov(fname_cov)
    cov = regularize(cov, evoked.info, rank=None)

    alpha = 0.5
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=True, update_mode=1)
    _check_stc(stc, evoked, 68477, 'lh', fwd=forward)

    vec_stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                        xyz_same_gamma=True, update_mode=1, pick_ori='vector')
    assert_stcs_equal(vec_stc.magnitude(), stc)

    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=1)
    _check_stc(stc, evoked, 82010, 'lh', fwd=forward, dist_limit=4., ratio=20.)

    dips = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                     xyz_same_gamma=False, update_mode=1,
                     return_as_dipoles=True)
    assert (isinstance(dips[0], Dipole))
    stc_dip = make_stc_from_dipoles(dips, forward['src'])
    assert_stcs_equal(stc, stc_dip)

    # force fixed orientation
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=2,
                    loose=0, return_residual=False)
    _check_stc(stc, evoked, 85739, 'lh', fwd=forward, ratio=20.)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_gamma_map_vol_sphere():
    """Gamma MAP with a sphere forward and volumic source space."""
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0),
                          proj=False)
    evoked.resample(50, npad=100)
    evoked.crop(tmin=0.1, tmax=0.16)  # crop to window around peak

    cov = read_cov(fname_cov)
    cov = regularize(cov, evoked.info, rank=None)

    info = evoked.info
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src = mne.setup_volume_source_space(subject=None, pos=30., mri=None,
                                        sphere=(0.0, 0.0, 0.0, 0.08),
                                        bem=None, mindist=5.0,
                                        exclude=2.0, sphere_units='m')
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                    eeg=False, meg=True)

    alpha = 0.5
    pytest.raises(ValueError, gamma_map, evoked, fwd, cov, alpha,
                  loose=0, return_residual=False)

    pytest.raises(ValueError, gamma_map, evoked, fwd, cov, alpha,
                  loose=0.2, return_residual=False)

    stc = gamma_map(evoked, fwd, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=2,
                    return_residual=False)

    assert_array_almost_equal(stc.times, evoked.times, 5)

    # Compare orientation obtained using fit_dipole and gamma_map
    # for a simulated evoked containing a single dipole
    stc = mne.VolSourceEstimate(50e-9 * np.random.RandomState(42).randn(1, 4),
                                vertices=stc.vertices[:1],
                                tmin=stc.tmin,
                                tstep=stc.tstep)
    evoked_dip = mne.simulation.simulate_evoked(fwd, stc, info, cov, nave=1e9,
                                                use_cps=True)

    dip_gmap = gamma_map(evoked_dip, fwd, cov, 0.1, return_as_dipoles=True)

    amp_max = [np.max(d.amplitude) for d in dip_gmap]
    dip_gmap = dip_gmap[np.argmax(amp_max)]
    assert (dip_gmap[0].pos[0] in src[0]['rr'][stc.vertices])

    dip_fit = mne.fit_dipole(evoked_dip, cov, sphere)[0]
    assert (np.abs(np.dot(dip_fit.ori[0], dip_gmap.ori[0])) > 0.99)


run_tests_if_main()
