# Author: Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#         Samuel Deslauriers-Gauthier <sam.deslauriers@gmail.com>
#
# License: BSD (3-clause)

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
from mne.utils import run_tests_if_main, check_version


data_path = testing.data_path(download=False)
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
label_names = ['Aud-lh', 'Aud-rh', 'Vis-rh']

subjects_dir = op.join(data_path, 'subjects')


@pytest.fixture(scope="module", params=[testing._pytest_param()])
def _get_fwd_labels():
    fwd = read_forward_solution(fname_fwd)
    fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=True)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)
    labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                         '%s.label' % label)) for label in label_names]
    return fwd, labels


def _get_idx_label_stc(label, stc):
    hemi_idx_mapping = dict(lh=0, rh=1)

    hemi_idx = hemi_idx_mapping[label.hemi]

    idx = np.intersect1d(stc.vertices[hemi_idx], label.vertices)
    idx = np.searchsorted(stc.vertices[hemi_idx], idx)

    if hemi_idx == 1:
        idx += len(stc.vertices[0])
    return idx


def test_simulate_stc(_get_fwd_labels):
    """Test generation of source estimate."""
    fwd, labels = _get_fwd_labels
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
        idx = _get_idx_label_stc(label, stc)
        assert (np.all(stc.data[idx] == 1.0))
        assert (stc.data[idx].shape[1] == n_times)

    # test with function
    def fun(x):
        return x ** 2

    stc = simulate_stc(fwd['src'], mylabels, stc_data, tmin, tstep, fun)

    # the first label has value 0, the second value 2, the third value 6

    for i, label in enumerate(labels):
        idx = _get_idx_label_stc(label, stc)
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

    i = np.where(fwd['src'][0]['inuse'] == 0)[0][0]
    label_single_vert = Label(vertices=[i],
                              pos=fwd['src'][0]['rr'][i:i + 1, :],
                              hemi='lh')
    stc = simulate_stc(fwd['src'], [label_single_vert], stc_data[:1], tmin,
                       tstep)
    assert_equal(len(stc.lh_vertno), 1)


def test_simulate_sparse_stc(_get_fwd_labels):
    """Test generation of sparse source estimate."""
    fwd, labels = _get_fwd_labels
    n_times = 10
    tmin = 0
    tstep = 1e-3
    times = np.arange(n_times, dtype=np.float64) * tstep + tmin

    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(labels),
                  times, labels=labels, location='center', subject='sample',
                  subjects_dir=subjects_dir)  # no non-zero values

    mylabels = []
    for label in labels:
        this_label = label.copy()
        this_label.values.fill(1.)
        mylabels.append(this_label)

    for location in ('random', 'center'):
        random_state = 0 if location == 'random' else None
        stc_1 = simulate_sparse_stc(fwd['src'], len(mylabels), times,
                                    labels=mylabels, random_state=random_state,
                                    location=location,
                                    subjects_dir=subjects_dir)
        assert_equal(stc_1.subject, 'sample')

        assert (stc_1.data.shape[0] == len(mylabels))
        assert (stc_1.data.shape[1] == n_times)

        # make sure we get the same result when using the same seed
        stc_2 = simulate_sparse_stc(fwd['src'], len(mylabels), times,
                                    labels=mylabels, random_state=random_state,
                                    location=location,
                                    subjects_dir=subjects_dir)

        assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
        assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)

    # Degenerate cases
    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(mylabels),
                  times, labels=mylabels, location='center', subject='foo',
                  subjects_dir=subjects_dir)  # wrong subject
    del fwd['src'][0]['subject_his_id']  # remove subject
    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(mylabels),
                  times, labels=mylabels, location='center',
                  subjects_dir=subjects_dir)  # no subject
    fwd['src'][0]['subject_his_id'] = 'sample'  # put back subject
    pytest.raises(ValueError, simulate_sparse_stc, fwd['src'], len(mylabels),
                  times, labels=mylabels, location='foo')  # bad location
    err_str = 'Number of labels'
    with pytest.raises(ValueError, match=err_str):
        simulate_sparse_stc(
            fwd['src'], len(mylabels) + 1, times, labels=mylabels,
            random_state=random_state, location=location,
            subjects_dir=subjects_dir)


def test_generate_stc_single_hemi(_get_fwd_labels):
    """Test generation of source estimate, single hemi."""
    fwd, labels = _get_fwd_labels
    labels_single_hemi = labels[1:]  # keep only labels in one hemisphere

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
        idx = _get_idx_label_stc(label, stc)
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


def test_simulate_sparse_stc_single_hemi(_get_fwd_labels):
    """Test generation of sparse source estimate."""
    fwd, labels = _get_fwd_labels
    labels_single_hemi = labels[1:]  # keep only labels in one hemisphere

    n_times = 10
    tmin = 0
    tstep = 1e-3
    times = np.arange(n_times, dtype=np.float64) * tstep + tmin

    stc_1 = simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                                labels=labels_single_hemi, random_state=0)

    assert (stc_1.data.shape[0] == len(labels_single_hemi))
    assert (stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                                labels=labels_single_hemi, random_state=0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)

    # smoke test for new API
    if check_version('numpy', '1.17'):
        simulate_sparse_stc(fwd['src'], len(labels_single_hemi), times,
                            labels=labels_single_hemi,
                            random_state=np.random.default_rng(0))


@testing.requires_testing_data
def test_simulate_stc_labels_overlap(_get_fwd_labels):
    """Test generation of source estimate, overlapping labels."""
    fwd, labels = _get_fwd_labels
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


def test_source_simulator(_get_fwd_labels):
    """Test Source Simulator."""
    fwd, _ = _get_fwd_labels

    src = fwd['src']
    hemi_to_ind = {'lh': 0, 'rh': 1}
    tstep = 1. / 6.

    label_vertices = [[], [], []]
    label_vertices[0] = np.arange(1000)
    label_vertices[1] = np.arange(500, 1500)
    label_vertices[2] = np.arange(1000)

    hemis = ['lh', 'lh', 'rh']

    mylabels = []
    src_vertices = []
    for i, vert in enumerate(label_vertices):
        new_label = Label(vertices=vert, hemi=hemis[i])
        mylabels.append(new_label)
        src_vertices.append(np.intersect1d(
                            src[hemi_to_ind[hemis[i]]]['vertno'],
                            new_label.vertices))

    wfs = [[], [], []]

    wfs[0] = np.array([0, 1., 0])  # 1d array
    wfs[1] = [np.array([0, 1., 0]),  # list
              np.array([0, 1.5, 0])]
    wfs[2] = np.array([[1, 1, 1.]])  # 2d array

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

    ss = SourceSimulator(src, tstep)
    for i in range(3):
        ss.add_data(mylabels[i], wfs[i], events[i])

    stc = ss.get_stc()
    stim_channel = ss.get_stim_channel()

    # Stim channel data must have the same size as stc time samples
    assert len(stim_channel) == stc.data.shape[1]

    stim_channel = ss.get_stim_channel(0, 0)
    assert len(stim_channel) == 0

    assert np.all(stc.vertices[0] == verts_lh)
    assert np.all(stc.vertices[1] == verts_rh)
    assert_array_almost_equal(stc.lh_data, output_data_lh)
    assert_array_almost_equal(stc.rh_data, output_data_rh)
    assert_array_almost_equal(stc.data, output_data)

    counter = 0
    for stc, stim in ss:
        assert stc.data.shape[1] == 6
        counter += 1
    assert counter == 1

    half_ss = SourceSimulator(src, tstep, duration=0.5)
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

    # Verify that the chunks have the correct length.
    source_simulator = SourceSimulator(src, tstep=tstep, duration=10 * tstep)
    source_simulator.add_data(mylabels[0], np.array([1, 1, 1]), [[0, 0, 0]])

    source_simulator._chk_duration = 6  # Quick hack to get short chunks.
    stcs = [stc for stc, _ in source_simulator]
    assert len(stcs) == 2
    assert stcs[0].data.shape[1] == 6
    assert stcs[1].data.shape[1] == 4


run_tests_if_main()
