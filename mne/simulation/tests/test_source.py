import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true, assert_raises, assert_equal

from mne.datasets import testing
from mne import read_label, read_forward_solution, pick_types_forward
from mne.label import Label
from mne.simulation.source import simulate_stc, simulate_sparse_stc
from mne.utils import run_tests_if_main


data_path = testing.data_path(download=False)
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
    assert_equal(stc.subject, 'sample')

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

    # degenerate conditions
    label_subset = mylabels[:2]
    data_subset = stc_data[:2]
    stc = simulate_stc(fwd['src'], label_subset, data_subset, tmin, tstep, fun)
    assert_raises(ValueError, simulate_stc, fwd['src'],
                  label_subset, data_subset[:-1], tmin, tstep, fun)
    assert_raises(RuntimeError, simulate_stc, fwd['src'], label_subset * 2,
                  np.concatenate([data_subset] * 2, axis=0), tmin, tstep, fun)


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

    assert_raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='center', subject='sample',
                  subjects_dir=subjects_dir)  # no non-zero values
    for label in labels:
        label.values.fill(1.)
    for location in ('random', 'center'):
        random_state = 0 if location == 'random' else None
        stc_1 = simulate_sparse_stc(fwd['src'], len(labels), times,
                                    labels=labels, random_state=random_state,
                                    location=location,
                                    subjects_dir=subjects_dir)
        assert_equal(stc_1.subject, 'sample')

        assert_true(stc_1.data.shape[0] == len(labels))
        assert_true(stc_1.data.shape[1] == n_times)

        # make sure we get the same result when using the same seed
        stc_2 = simulate_sparse_stc(fwd['src'], len(labels), times,
                                    labels=labels, random_state=random_state,
                                    location=location,
                                    subjects_dir=subjects_dir)

        assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
        assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)
    # Degenerate cases
    assert_raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='center', subject='foo',
                  subjects_dir=subjects_dir)  # wrong subject
    del fwd['src'][0]['subject_his_id']
    assert_raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='center',
                  subjects_dir=subjects_dir)  # no subject
    assert_raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='foo')  # bad location


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

run_tests_if_main()
