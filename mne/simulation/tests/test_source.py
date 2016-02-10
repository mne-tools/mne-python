import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from nose.tools import assert_true, assert_raises

from mne.datasets import testing
from mne import (read_label, read_forward_solution, pick_types_forward,
                 pick_info, read_dipole, read_evokeds, read_cov, fit_dipole,
                 pick_types, make_sphere_model)
from mne.label import Label
from mne.dipole import Dipole
from mne.proj import make_eeg_average_ref_proj
from mne.simulation.evoked import simulate_evoked
from mne.simulation.source import (simulate_stc, simulate_sparse_stc,
                                   make_forward_dipole)
from mne.source_estimate import VolSourceEstimate
from mne.utils import run_tests_if_main, slow_test


data_path = testing.data_path(download=False)
fname_dip = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
fname_evo = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_bem = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
label_names = ['Aud-lh', 'Aud-rh', 'Vis-rh']

label_names_single_hemi = ['Aud-rh', 'Vis-rh']
subjects_dir = op.join(data_path, 'subjects')


def read_forward_solution_meg(*args, **kwargs):
    fwd = read_forward_solution(*args, **kwargs)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)
    return fwd


@testing.requires_testing_data
def test_simulate_stc():
    """ Test generation of source estimate """
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True)
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                         '%s.label' % label)) for label in label_names]
    mylabels = []
    for i, label in enumerate(labels):
        new_label = Label(vertices=label.vertices,
                          pos=label.pos,
                          values=2 * i * np.ones(len(label.values)),
                          hemi=label.hemi,
                          comment=label.comment)
        mylabels.append(new_label)

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels), n_times))
    stc = simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep)

    for label in labels:
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertices[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertices[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertices[0])

        assert_true(np.all(stc.data[idx] == 1.0))
        assert_true(stc.data[idx].shape[1] == n_times)

    # test with function
    def fun(x):
        return x ** 2
    stc = simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep, fun)

    # the first label has value 0, the second value 2, the third value 6

    for i, label in enumerate(labels):
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertices[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertices[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertices[0])

        res = ((2. * i) ** 2.) * np.ones((len(idx), n_times))
        assert_array_almost_equal(stc.data[idx], res)


@testing.requires_testing_data
def test_simulate_sparse_stc():
    """ Test generation of sparse source estimate """
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True)
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                         '%s.label' % label)) for label in label_names]

    n_times = 10
    tmin = 0
    tstep = 1e-3
    times = np.arange(n_times, dtype=np.float) * tstep + tmin

    stc_1 = simulate_sparse_stc(fwd['src'], len(labels), times,
                                labels=labels, random_state=0)

    assert_true(stc_1.data.shape[0] == len(labels))
    assert_true(stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = simulate_sparse_stc(fwd['src'], len(labels), times,
                                labels=labels, random_state=0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)


@testing.requires_testing_data
def test_generate_stc_single_hemi():
    """ Test generation of source estimate, single hemi """
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True)
    labels_single_hemi = [read_label(op.join(data_path, 'MEG', 'sample',
                                             'labels', '%s.label' % label))
                          for label in label_names_single_hemi]
    mylabels = []
    for i, label in enumerate(labels_single_hemi):
        new_label = Label(vertices=label.vertices,
                          pos=label.pos,
                          values=2 * i * np.ones(len(label.values)),
                          hemi=label.hemi,
                          comment=label.comment)
        mylabels.append(new_label)

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels_single_hemi), n_times))
    stc = simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep)

    for label in labels_single_hemi:
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertices[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertices[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertices[0])

        assert_true(np.all(stc.data[idx] == 1.0))
        assert_true(stc.data[idx].shape[1] == n_times)

    # test with function
    def fun(x):
        return x ** 2
    stc = simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep, fun)

    # the first label has value 0, the second value 2, the third value 6

    for i, label in enumerate(labels_single_hemi):
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertices[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertices[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertices[0])

        res = ((2. * i) ** 2.) * np.ones((len(idx), n_times))
        assert_array_almost_equal(stc.data[idx], res)


@testing.requires_testing_data
def test_simulate_sparse_stc_single_hemi():
    """ Test generation of sparse source estimate """
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True)
    n_times = 10
    tmin = 0
    tstep = 1e-3
    times = np.arange(n_times, dtype=np.float) * tstep + tmin

    labels_single_hemi = [read_label(op.join(data_path, 'MEG', 'sample',
                                             'labels', '%s.label' % label))
                          for label in label_names_single_hemi]

    stc_1 = simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                                labels=labels_single_hemi, random_state=0)

    assert_true(stc_1.data.shape[0] == len(labels_single_hemi))
    assert_true(stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                                labels=labels_single_hemi, random_state=0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)


@slow_test
@testing.requires_testing_data
def test_make_forward_dipole():
    """Test forward-projecting dipoles"""
    rng = np.random.RandomState(0)

    evoked = read_evokeds(fname_evo)[0]
    info = evoked.info
    cov = read_cov(fname_cov)
    dip_c = read_dipole(fname_dip)

    # Make new Dipole object with n_test_dipoles, uneven sampling in time
    n_test_dipoles = 5
    dipsel = np.sort(rng.permutation(range(len(dip_c)))[:n_test_dipoles])
    dip_test = Dipole(times=dip_c.times[dipsel],
                      pos=dip_c.pos[dipsel],
                      amplitude=dip_c.amplitude[dipsel],
                      ori=dip_c.ori[dipsel],
                      gof=dip_c.gof[dipsel])

    sphere = make_sphere_model(head_radius=0.1)
    stc, fwd = make_forward_dipole(dip_test, sphere, evoked.info,
                                   trans=fname_trans)
    assert_true(isinstance(stc, list))

    times, pos, amplitude, ori, gof = [], [], [], [], []
    snr = 20
    for s in stc:
        evo_test = simulate_evoked(fwd, s, evoked.info, cov,
                                   snr=snr, random_state=rng)
        evo_test.add_proj(make_eeg_average_ref_proj(evo_test.info))
        dfit, resid = fit_dipole(evo_test, cov, sphere, None)
        times += list(dfit.times)
        pos += list(dfit.pos)
        amplitude += list(dfit.amplitude)
        ori += list(dfit.ori)
        gof += list(dfit.gof)

    dip_fit = Dipole(np.array(times), np.array(pos), np.array(amplitude),
                     np.array(ori), np.array(gof))

    # check that we did at least as well
    diff = dip_test.pos - dip_fit.pos
    corr = np.corrcoef(dip_test.pos.ravel(), dip_fit.pos.ravel())[0, 1]
    dist = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    gc_dist = 180 / np.pi * np.mean(np.arccos(
            np.sum(dip_test.ori * dip_fit.ori, axis=1)))
    amp_err = np.sqrt(np.mean((dip_test.amplitude - dip_fit.amplitude) ** 2))

    # Make sure each coordinate is within 2 mm from reference
    # NB tolerance should be set relative to snr of simulated evoked!
    assert_allclose(dip_fit.pos, dip_test.pos, rtol=0, atol=2e-3,
                    err_msg='position mismatch')
    assert_true(dist < 2e-3, 'dist: %s' % dist)  # within 2 mm
    assert_true(corr > 1 - 2e-3, 'corr: %s' % corr)
    assert_true(gc_dist < 2, 'gc_dist: %s' % gc_dist)  # less than 2 degrees
    assert_true(amp_err < 2e-3, 'amp_err: %s' % amp_err)  # within 2 nAm

    # Make sure rejection works with bem: one dipole at z=1m
    # NB _make_forward.py:_prepare_for_forward will raise a RuntimeError
    # if no points are left after min_dist exclusions
    dip_outside = Dipole(times=np.array([0., 0.001]),
                         pos=np.array([[0., 0., 1.0], [0., 0., 0.040]]),
                         amplitude=np.array([100e-9, 100e-9]),
                         ori=np.array([[1., 0., 0.], [1., 0., 0.]]), gof=1)
    assert_raises(ValueError, make_forward_dipole, dip_outside, fname_bem,
                  evoked.info, fname_trans)

    # Now make an evenly sampled set of dipoles, some simultaneous,
    # should return a VolSourceEstimate regardless
    times = np.array([0., 0., 0., 0.001, 0.001, 0.002])
    pos = np.random.rand(6, 3)*0.020 + np.array([0., 0., 0.040])[np.newaxis, :]
    amplitude = np.random.rand(6,)*100e-9
    ori = np.eye(6, 3) + np.eye(6, 3, -3)
    gof = np.arange(len(times))/len(times)

    dip_even_samp = Dipole(times, pos, amplitude, ori, gof)

    # restrict sensors to EEG using pre-picking
    picks = pick_types(info, meg=False, eeg=True)
    eeg_info = pick_info(evoked.info, picks)
    stc, fwd = make_forward_dipole(dip_even_samp, fname_bem, eeg_info,
                                   trans=fname_trans)

    assert_true(isinstance, VolSourceEstimate)
    assert_allclose(stc.times, np.arange(0., 0.003, 0.001))
    assert_true(fwd['info']['ch_names'][0] == 'EEG 001')
    assert_true(fwd['info']['ch_names'][-1] == 'EEG 060')

run_tests_if_main()
