import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
import pytest

from mne.datasets import testing
from mne import (read_label, read_forward_solution, pick_types_forward,
                 convert_forward_solution)
from mne.label import Label
from mne.simulation.source import simulate_stc, simulate_sparse_stc
from mne.simulation.source import SourceSimulator
from mne.utils import run_tests_if_main


data_path = testing.data_path(download=False)
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
label_names = ['Aud-lh', 'Aud-rh', 'Vis-rh']

label_names_single_hemi = ['Aud-rh', 'Vis-rh']
subjects_dir = op.join(data_path, 'subjects')


def read_forward_solution_meg(*args, **kwargs):
    """Read forward MEG."""
    fwd = read_forward_solution(*args)
    fwd = convert_forward_solution(fwd, **kwargs)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)
    return fwd


@testing.requires_testing_data
def test_simulate_stc():
    """Test generation of source estimate."""
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True, use_cps=True)
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

        assert (np.all(stc.data[idx] == 1.0))
        assert (stc.data[idx].shape[1] == n_times)

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
    pytest.raises(ValueError, simulate_stc, fwd['src'],
                  label_subset, data_subset[:-1], tmin, tstep, fun)
    pytest.raises(RuntimeError, simulate_stc, fwd['src'], label_subset * 2,
                  np.concatenate([data_subset] * 2, axis=0), tmin, tstep, fun)


@testing.requires_testing_data
def test_simulate_sparse_stc():
    """Test generation of sparse source estimate."""
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True, use_cps=True)
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                         '%s.label' % label)) for label in label_names]

    n_times = 10
    tmin = 0
    tstep = 1e-3
    times = np.arange(n_times, dtype=np.float) * tstep + tmin

    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
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

        assert (stc_1.data.shape[0] == len(labels))
        assert (stc_1.data.shape[1] == n_times)

        # make sure we get the same result when using the same seed
        stc_2 = simulate_sparse_stc(fwd['src'], len(labels), times,
                                    labels=labels, random_state=random_state,
                                    location=location,
                                    subjects_dir=subjects_dir)

        assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
        assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)
    # Degenerate cases
    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='center', subject='foo',
                  subjects_dir=subjects_dir)  # wrong subject
    del fwd['src'][0]['subject_his_id']
    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='center',
                  subjects_dir=subjects_dir)  # no subject
    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='foo')  # bad location
    err_str = 'Number of labels'
    with pytest.raises(ValueError, match=err_str):
        simulate_sparse_stc(
            fwd['src'], len(labels) + 1, times, labels=labels,
            random_state=random_state, location=location,
            subjects_dir=subjects_dir)


@testing.requires_testing_data
def test_generate_stc_single_hemi():
    """Test generation of source estimate, single hemi."""
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True, use_cps=True)
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

        assert (np.all(stc.data[idx] == 1.0))
        assert (stc.data[idx].shape[1] == n_times)

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
    """Test generation of sparse source estimate."""
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True, use_cps=True)
    n_times = 10
    tmin = 0
    tstep = 1e-3
    times = np.arange(n_times, dtype=np.float) * tstep + tmin

    labels_single_hemi = [read_label(op.join(data_path, 'MEG', 'sample',
                                             'labels', '%s.label' % label))
                          for label in label_names_single_hemi]

    stc_1 = simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                                labels=labels_single_hemi, random_state=0)

    assert (stc_1.data.shape[0] == len(labels_single_hemi))
    assert (stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                                labels=labels_single_hemi, random_state=0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)


@testing.requires_testing_data
def test_simulate_stc_labels_overlap():
    """Test generation of source estimate, overlapping labels."""
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True, use_cps=True)
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
    # Adding the last label twice
    mylabels.append(new_label)

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(mylabels), n_times))

    # Test false
    with pytest.raises(RuntimeError, match='must be non-overlapping'):
        simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep,
                     allow_overlap=False)
    # test True
    stc = simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep,
                       allow_overlap=True)
    assert_equal(stc.subject, 'sample')
    assert (stc.data.shape[1] == n_times)
    # Some of the elements should be equal to 2 since we have duplicate labels
    assert (2 in stc.data)


@testing.requires_testing_data
def test_source_simulator():
    """Test Source Simulator."""
    fwd = read_forward_solution_meg(fname_fwd, force_fixed=True, use_cps=True)
    src = fwd['src']
    hemi_to_ind = {'lh': 0, 'rh': 1}
    tmin = 0
    tstep = 1. / 6.

    label_vertices = [[], [], []]
    label_vertices[0] = np.arange(1000)
    label_vertices[1] = np.arange(500, 1500)
    label_vertices[2] = np.arange(1000)

    hemis = ['lh', 'lh', 'rh']

    mylabels = []
    src_vertices = []
    for i, vert in enumerate(label_vertices):
        new_label = Label(vertices=vert,
                          hemi=hemis[i])
        mylabels.append(new_label)
        src_vertices.append(np.intersect1d(
                            src[hemi_to_ind[hemis[i]]]['vertno'],
                            new_label.vertices))

    wfs = [[], [], []]

    wfs[0] = np.array([0, 1., 0])
    wfs[1] = [np.array([0, 1., 0]),
              np.array([0, 1.5, 0])]
    wfs[2] = np.array([1, 1, 1.])

    events = [[], [], []]
    events[0] = np.array([[0, 0, 1], [3, 0, 1]])
    events[1] = np.array([[0, 0, 1], [3, 0, 1]])
    events[2] = np.array([[0, 0, 1], [2, 0, 1]])

    verts_lh = np.intersect1d(range(1500), src[0]['vertno'])
    verts_rh = np.intersect1d(range(1000), src[1]['vertno'])
    diff_01 = len(np.setdiff1d(src_vertices[0], src_vertices[1]))
    diff_10 = len(np.setdiff1d(src_vertices[1], src_vertices[0]))
    inter_10 = len(np.intersect1d(src_vertices[1], src_vertices[0]))

    output_data_lh = np.zeros([len(verts_lh), 6])
    tmp = np.array([0, 1., 0, 0, 1, 0])
    output_data_lh[:diff_01, :] = np.tile(tmp, (diff_01, 1))

    tmp = np.array([0, 2, 0, 0, 2.5, 0])
    output_data_lh[diff_01:diff_01 + inter_10, :] = np.tile(tmp, (inter_10, 1))
    tmp = np.array([0, 1, 0, 0, 1.5, 0])
    output_data_lh[diff_01 + inter_10:, :] = np.tile(tmp, (diff_10, 1))

    data_rh_wf = np.array([1., 1, 2, 1, 1, 0])
    output_data_rh = np.tile(data_rh_wf, (len(src_vertices[2]), 1))
    output_data = np.vstack([output_data_lh, output_data_rh])

    ss = SourceSimulator(src, tmin, tstep)
    for i in range(3):
        ss.add_data(mylabels[i], wfs[i], events[i])

    stc = ss.get_stc()
    stim_channel = ss.get_stim_channel()

    # Stim channel data must have the same size as stc time samples
    assert(len(stim_channel) == stc.data.shape[1])

    stim_channel = ss.get_stim_channel(0., 0.)
    assert(len(stim_channel) == 0)

    assert (np.all(stc.vertices[0] == verts_lh))
    assert (np.all(stc.vertices[1] == verts_rh))
    assert_array_almost_equal(stc.lh_data, output_data_lh)
    assert_array_almost_equal(stc.rh_data, output_data_rh)
    assert_array_almost_equal(stc.data, output_data)

    counter = 0
    for stc, stim in ss:
        counter += 1
    assert counter == 1

    half_ss = SourceSimulator(src, tmin, tstep, duration=0.5)
    for i in range(3):
        half_ss.add_data(mylabels[i], wfs[i], events[i])
    half_stc = half_ss.get_stc()
    assert_array_almost_equal(stc.data[:, :3], half_stc.data)

    ss = SourceSimulator(src)

    with pytest.raises(ValueError, match='No simulation parameters'):
        ss.get_stc()

    with pytest.raises(ValueError, match='label must be a Label'):
        ss.add_data(1, wfs, events)

    with pytest.raises(ValueError, match='Number of waveforms and events '
                       'should match'):
        ss.add_data(mylabels[0], wfs[:2], events)


run_tests_if_main()
