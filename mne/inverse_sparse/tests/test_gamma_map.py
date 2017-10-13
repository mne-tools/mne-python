# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import os.path as op

from nose.tools import assert_true, assert_raises
import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_equal,
                           assert_allclose)

import mne
from mne.datasets import testing
from mne import (read_cov, read_forward_solution, read_evokeds,
                 convert_forward_solution)
from mne.cov import regularize
from mne.inverse_sparse import gamma_map
from mne.inverse_sparse.mxne_inverse import make_stc_from_dipoles
from mne import pick_types_forward
from mne.utils import run_tests_if_main
from mne.dipole import Dipole

data_path = testing.data_path(download=False)
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
subjects_dir = op.join(data_path, 'subjects')


def _check_stc(stc, evoked, idx, ratio=50.):
    """Helper to check correctness"""
    assert_array_almost_equal(stc.times, evoked.times, 5)
    amps = np.sum(stc.data ** 2, axis=1)
    order = np.argsort(amps)[::-1]
    amps = amps[order]
    verts = np.concatenate(stc.vertices)[order]
    assert_equal(idx, verts[0], err_msg=str(list(verts)))
    assert_true(amps[0] > ratio * amps[1], msg=str(amps[0] / amps[1]))


def _check_stcs(stc1, stc2):
    """Helper to check correctness"""
    assert_allclose(stc1.times, stc2.times)
    assert_allclose(stc1.data, stc2.data)
    assert_allclose(stc1.vertices[0], stc2.vertices[0])
    assert_allclose(stc1.vertices[1], stc2.vertices[1])
    assert_allclose(stc1.tmin, stc2.tmin)
    assert_allclose(stc1.tstep, stc2.tstep)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_gamma_map():
    """Test Gamma MAP inverse"""
    forward = read_forward_solution(fname_fwd)
    forward = convert_forward_solution(forward, surf_ori=True)

    forward = pick_types_forward(forward, meg=False, eeg=True)
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0),
                          proj=False)
    evoked.resample(50, npad=100)
    evoked.crop(tmin=0.1, tmax=0.16)  # crop to window around peak

    cov = read_cov(fname_cov)
    cov = regularize(cov, evoked.info)

    alpha = 0.5
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=True, update_mode=1)
    _check_stc(stc, evoked, 68477)

    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=1)
    _check_stc(stc, evoked, 82010)

    dips = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                     xyz_same_gamma=False, update_mode=1,
                     return_as_dipoles=True)
    assert_true(isinstance(dips[0], Dipole))
    stc_dip = make_stc_from_dipoles(dips, forward['src'])
    _check_stcs(stc, stc_dip)

    # force fixed orientation
    stc = gamma_map(evoked, forward, cov, alpha, tol=1e-4,
                    xyz_same_gamma=False, update_mode=2,
                    loose=0, return_residual=False)
    _check_stc(stc, evoked, 85739, 20)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_gamma_map_vol_sphere():
    """Gamma MAP with a sphere forward and volumic source space"""
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0),
                          proj=False)
    evoked.resample(50, npad=100)
    evoked.crop(tmin=0.1, tmax=0.16)  # crop to window around peak

    cov = read_cov(fname_cov)
    cov = regularize(cov, evoked.info)

    info = evoked.info
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src = mne.setup_volume_source_space(subject=None, pos=15., mri=None,
                                        sphere=(0.0, 0.0, 0.0, 80.0),
                                        bem=None, mindist=5.0,
                                        exclude=2.0)
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                    eeg=False, meg=True)

    alpha = 0.5
    assert_raises(ValueError, gamma_map, evoked, fwd, cov, alpha,
                  loose=0, return_residual=False)

    assert_raises(ValueError, gamma_map, evoked, fwd, cov, alpha,
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
    assert_true(dip_gmap[0].pos[0] in src[0]['rr'][stc.vertices])

    dip_fit = mne.fit_dipole(evoked_dip, cov, sphere)[0]
    assert_true(np.abs(np.dot(dip_fit.ori[0], dip_gmap.ori[0])) > 0.99)

run_tests_if_main()
