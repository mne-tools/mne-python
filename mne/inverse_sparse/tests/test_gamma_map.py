# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

import mne
from mne.datasets import testing
from mne import (read_cov, read_forward_solution, read_evokeds,
                 convert_forward_solution, VectorSourceEstimate)
from mne.cov import regularize
from mne.inverse_sparse import gamma_map
from mne.inverse_sparse.mxne_inverse import make_stc_from_dipoles
from mne.minimum_norm.tests.test_inverse import (assert_stc_res,
                                                 assert_var_exp_log)
from mne import pick_types_forward
from mne.utils import assert_stcs_equal, catch_logging
from mne.dipole import Dipole

data_path = testing.data_path(download=False)
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')


def _check_stc(stc, evoked, idx, hemi, fwd, dist_limit=0., ratio=50.,
               res=None, atol=1e-20):
    """Check correctness."""
    assert_array_almost_equal(stc.times, evoked.times, 5)
    stc_orig = stc
    if isinstance(stc, VectorSourceEstimate):
        assert stc.data.any(1).any(1).all()  # all dipoles should have some
        stc = stc.magnitude()
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
    if res is not None:
        assert_stc_res(evoked, stc_orig, fwd, res, atol=atol)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_gamma_map_standard():
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
    with catch_logging() as log:
        stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                        xyz_same_gamma=True, update_mode=1, verbose=True)
    _check_stc(stc, evoked, 68477, 'lh', fwd=forward)
    assert_var_exp_log(log.getvalue(), 20, 22)

    with catch_logging() as log:
        stc_vec, res = gamma_map(
            evoked, forward, cov, alpha, tol=1e-4, xyz_same_gamma=True,
            update_mode=1, pick_ori='vector', return_residual=True,
            verbose=True)
    assert_var_exp_log(log.getvalue(), 20, 22)
    assert_stcs_equal(stc_vec.magnitude(), stc)
    _check_stc(stc_vec, evoked, 68477, 'lh', fwd=forward, res=res)

    stc, res = gamma_map(
        evoked, forward, cov, alpha, tol=1e-4, xyz_same_gamma=False,
        update_mode=1, pick_ori='vector', return_residual=True)
    _check_stc(stc, evoked, 82010, 'lh', fwd=forward, dist_limit=6., ratio=2.,
               res=res)

    with catch_logging() as log:
        dips = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                         xyz_same_gamma=False, update_mode=1,
                         return_as_dipoles=True, verbose=True)
    exp_var = assert_var_exp_log(log.getvalue(), 58, 60)
    dip_exp_var = np.mean(sum(dip.gof for dip in dips))
    assert_allclose(exp_var, dip_exp_var, atol=10)  # not really equiv, close
    assert (isinstance(dips[0], Dipole))
    stc_dip = make_stc_from_dipoles(dips, forward['src'])
    assert_stcs_equal(stc.magnitude(), stc_dip)

    # force fixed orientation
    stc, res = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                         xyz_same_gamma=False, update_mode=2,
                         loose=0, return_residual=True)
    _check_stc(stc, evoked, 85739, 'lh', fwd=forward, ratio=20., res=res)


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
    stc = gamma_map(evoked, fwd, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=2,
                    return_residual=False)
    assert_array_almost_equal(stc.times, evoked.times, 5)

    # Computing inverse with restricted orientations should also work, since
    # we have a discrete source space.
    stc = gamma_map(evoked, fwd, cov, alpha, loose=0.2, return_residual=False)
    assert_array_almost_equal(stc.times, evoked.times, 5)

    # Compare orientation obtained using fit_dipole and gamma_map
    # for a simulated evoked containing a single dipole
    stc = mne.VolSourceEstimate(50e-9 * np.random.RandomState(42).randn(1, 4),
                                vertices=[stc.vertices[0][:1]],
                                tmin=stc.tmin,
                                tstep=stc.tstep)
    evoked_dip = mne.simulation.simulate_evoked(fwd, stc, info, cov, nave=1e9,
                                                use_cps=True)

    dip_gmap = gamma_map(evoked_dip, fwd, cov, 0.1, return_as_dipoles=True)

    amp_max = [np.max(d.amplitude) for d in dip_gmap]
    dip_gmap = dip_gmap[np.argmax(amp_max)]
    assert (dip_gmap[0].pos[0] in src[0]['rr'][stc.vertices[0]])

    dip_fit = mne.fit_dipole(evoked_dip, cov, sphere)[0]
    assert (np.abs(np.dot(dip_fit.ori[0], dip_gmap.ori[0])) > 0.99)
