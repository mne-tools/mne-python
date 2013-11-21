import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true

from mne.datasets import sample
from mne import read_label, read_forward_solution
from mne.label import Label
from mne.simulation.source import generate_stc, generate_sparse_stc


data_path = sample.data_path()
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fwd = read_forward_solution(fname_fwd, force_fixed=True)
label_names = ['Aud-lh', 'Aud-rh', 'Vis-rh']
labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                    '%s.label' % label)) for label in label_names]

label_names_single_hemi = ['Aud-rh', 'Vis-rh']
labels_single_hemi = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                    '%s.label' % label)) for label in label_names_single_hemi]


def test_generate_stc():
    """ Test generation of source estimate """
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
    stc = generate_stc(fwd['src'], mylabels, stc_data, tmin, tstep)

    for label in labels:
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertno[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertno[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertno[0])

        assert_true(np.all(stc.data[idx] == 1.0))
        assert_true(stc.data[idx].shape[1] == n_times)

    # test with function
    fun = lambda x: x ** 2
    stc = generate_stc(fwd['src'], mylabels, stc_data, tmin, tstep, fun)

    # the first label has value 0, the second value 2, the third value 6

    for i, label in enumerate(labels):
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertno[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertno[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertno[0])

        assert_array_almost_equal(stc.data[idx],
                        ((2. * i) ** 2.) * np.ones((len(idx), n_times)))


def test_generate_sparse_stc():
    """ Test generation of sparse source estimate """

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels), n_times))\
                     * np.arange(len(labels))[:, None]
    stc_1 = generate_sparse_stc(fwd['src'], labels, stc_data, tmin, tstep, 0)

    for i, label in enumerate(labels):
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc_1.vertno[hemi_idx], label.vertices)
        idx = np.searchsorted(stc_1.vertno[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc_1.vertno[0])

        assert_true(np.all(stc_1.data[idx] == float(i)))

    assert_true(stc_1.data.shape[0] == len(labels))
    assert_true(stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = generate_sparse_stc(fwd['src'], labels, stc_data, tmin, tstep, 0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)


def test_generate_stc_single_hemi():
    """ Test generation of source estimate """
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
    stc = generate_stc(fwd['src'], mylabels, stc_data, tmin, tstep)

    for label in labels_single_hemi:
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertno[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertno[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertno[0])

        assert_true(np.all(stc.data[idx] == 1.0))
        assert_true(stc.data[idx].shape[1] == n_times)

    # test with function
    fun = lambda x: x ** 2
    stc = generate_stc(fwd['src'], mylabels, stc_data, tmin, tstep, fun)

    # the first label has value 0, the second value 2, the third value 6

    for i, label in enumerate(labels_single_hemi):
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc.vertno[hemi_idx], label.vertices)
        idx = np.searchsorted(stc.vertno[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc.vertno[0])

        assert_array_almost_equal(stc.data[idx],
                        ((2. * i) ** 2.) * np.ones((len(idx), n_times)))


def test_generate_sparse_stc_single_hemi():
    """ Test generation of sparse source estimate """

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels_single_hemi), n_times))\
                     * np.arange(len(labels_single_hemi))[:, None]
    stc_1 = generate_sparse_stc(fwd['src'], labels_single_hemi, stc_data,
                                tmin, tstep, 0)

    for i, label in enumerate(labels_single_hemi):
        if label.hemi == 'lh':
            hemi_idx = 0
        else:
            hemi_idx = 1

        idx = np.intersect1d(stc_1.vertno[hemi_idx], label.vertices)
        idx = np.searchsorted(stc_1.vertno[hemi_idx], idx)

        if hemi_idx == 1:
            idx += len(stc_1.vertno[0])

        assert_true(np.all(stc_1.data[idx] == float(i)))

    assert_true(stc_1.data.shape[0] == len(labels_single_hemi))
    assert_true(stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = generate_sparse_stc(fwd['src'], labels_single_hemi, stc_data,
                                tmin, tstep, 0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)
