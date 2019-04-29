# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk@uw.edu>
#          Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from scipy.linalg import norm

from mne import SourceEstimate
from mne import read_source_spaces
from mne.datasets import testing
from mne.simulation import (simulate_sparse_stc,
                            source_estimate_quantification,
                            stc_cosine, stc_dipole_localization_error,
                            stc_precision_score, stc_recall_score,
                            stc_f1_score, stc_roc_auc_score)
from mne.simulation.metrics import (_uniform_stc, _thresholding,
                                    _check_threshold)
from mne.utils import run_tests_if_main

data_path = testing.data_path(download=False)
src_fname = op.join(data_path, 'subjects', 'sample', 'bem',
                    'sample-oct-6-src.fif')


@testing.requires_testing_data
def test_metrics():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    times = np.arange(600) / 1000.
    rng = np.random.RandomState(42)
    stc1 = simulate_sparse_stc(src, n_dipoles=2, times=times, random_state=rng)
    stc2 = simulate_sparse_stc(src, n_dipoles=2, times=times, random_state=rng)
    E1_rms = source_estimate_quantification(stc1, stc1, metric='rms')
    E2_rms = source_estimate_quantification(stc2, stc2, metric='rms')
    E1_cos = source_estimate_quantification(stc1, stc1, metric='cosine')
    E2_cos = source_estimate_quantification(stc2, stc2, metric='cosine')

    # ### Tests to add
    assert (E1_rms == 0.)
    assert (E2_rms == 0.)
    assert_almost_equal(E1_cos, 0.)
    assert_almost_equal(E2_cos, 0.)
    stc_bad = stc2.copy().crop(0, 0.5)
    pytest.raises(ValueError, source_estimate_quantification, stc1, stc_bad)
    stc_bad = stc2.copy()
    stc_bad.tmin -= 0.1
    pytest.raises(ValueError, source_estimate_quantification, stc1, stc_bad)
    pytest.raises(ValueError, source_estimate_quantification, stc1, stc2,
                  metric='foo')


@testing.requires_testing_data
def test_uniform_and_thresholding():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert = [src[0]['vertno'][0:1], []]
    data = np.array([[0.8, -1.]])
    stc_true = SourceEstimate(data, vert, 0, 0.002, subject='sample')
    stc_bad = SourceEstimate(data, vert, 0, 0.002, subject='sample')
    stc_bad.vertices = [stc_bad.vertices[0]]
    pytest.raises(ValueError, _uniform_stc, stc_true, stc_bad)

    threshold = 0.9
    stc1, stc2 = _thresholding(stc_true, stc_true, threshold)
    assert_almost_equal(stc1._data, np.array([[0, -1.]]))
    assert_almost_equal(stc2._data, np.array([[0, -1.]]))
    assert_almost_equal(threshold, _check_threshold(threshold))

    threshold = '90'
    pytest.raises(ValueError, _check_threshold, threshold)


@testing.requires_testing_data
def test_cosine_metric():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:1], []]
    vert2 = [src[0]['vertno'][1:2], []]
    data1 = np.ones([1, 2])
    data2 = data1.copy()
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data2, vert1, 0, 0.002, subject='sample')

    E_per_sample1 = stc_cosine(stc_true, stc_est1)
    E_unique1 = stc_cosine(stc_true, stc_est1, per_sample=False)

    E_per_sample2 = stc_cosine(stc_true, stc_est2)
    E_unique2 = stc_cosine(stc_true, stc_est2, per_sample=False)

    assert_almost_equal(E_per_sample1, np.zeros(2))
    assert_almost_equal(E_unique1, 0.)
    assert_almost_equal(E_per_sample2, np.ones(2))
    assert_almost_equal(E_unique2, 1.)


@testing.requires_testing_data
def test_dle_metric():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:1], []]
    vert2 = [src[0]['vertno'][1:2], []]
    dist = norm(src[0]['rr'][vert1[0]] - src[0]['rr'][vert2[0]])
    data1 = np.ones([1, 2])
    data2 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')

    E_per_sample1 = stc_dipole_localization_error(stc_true, stc_est1, src)
    E_per_sample2 = stc_dipole_localization_error(stc_true, stc_est1, src,
                                                  threshold='70%')
    E_unique = stc_dipole_localization_error(stc_true, stc_est1, src,
                                             per_sample=False)

    # ### Tests to add
    assert_almost_equal(E_per_sample1, [np.inf, dist])
    assert_almost_equal(E_per_sample2, [dist, dist])
    assert_almost_equal(E_unique, dist)


@testing.requires_testing_data
def test_precision_metric():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:2], []]
    vert2 = [src[0]['vertno'][1:3], []]
    vert3 = [src[0]['vertno'][0:1], []]
    data1 = np.ones([2, 2])
    data2 = np.ones([2, 2])
    data3 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data3, vert3, 0, 0.002, subject='sample')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_unique1 = stc_precision_score(stc_true, stc_est1, per_sample=False)
        E_unique2 = stc_precision_score(stc_true, stc_est2, per_sample=False)
        E_per_sample1 = stc_precision_score(stc_true, stc_est2)
        E_per_sample2 = stc_precision_score(stc_true, stc_est2,
                                            threshold='70%')

    # ### Tests to add
    assert_almost_equal(E_unique1, 0.5)
    assert_almost_equal(E_unique2, 1.)
    assert_almost_equal(E_per_sample1, [0., 1.])
    assert_almost_equal(E_per_sample2, [1., 1.])


@testing.requires_testing_data
def test_recall_metric():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:2], []]
    vert2 = [src[0]['vertno'][1:3], []]
    vert3 = [src[0]['vertno'][0:1], []]
    data1 = np.ones([2, 2])
    data2 = np.ones([2, 2])
    data3 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data3, vert3, 0, 0.002, subject='sample')

    E_unique1 = stc_recall_score(stc_true, stc_est1, per_sample=False)
    E_unique2 = stc_recall_score(stc_true, stc_est2, per_sample=False)
    E_per_sample1 = stc_recall_score(stc_true, stc_est2)
    E_per_sample2 = stc_recall_score(stc_true, stc_est2, threshold='70%')

    # ### Tests to add
    assert_almost_equal(E_unique1, 0.5)
    assert_almost_equal(E_unique2, 0.5)
    assert_almost_equal(E_per_sample1, [0., 0.5])
    assert_almost_equal(E_per_sample2, [0.5, 0.5])


@testing.requires_testing_data
def test_f1_metric():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:2], []]
    vert2 = [src[0]['vertno'][1:3], []]
    vert3 = [src[0]['vertno'][0:1], []]
    data1 = np.ones([2, 2])
    data2 = np.ones([2, 2])
    data3 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data3, vert3, 0, 0.002, subject='sample')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_unique1 = stc_f1_score(stc_true, stc_est1, per_sample=False)
        E_unique2 = stc_f1_score(stc_true, stc_est2, per_sample=False)
        E_per_sample1 = stc_f1_score(stc_true, stc_est2)
        E_per_sample2 = stc_f1_score(stc_true, stc_est2, threshold='70%')
    assert_almost_equal(E_unique1, 0.5)
    assert_almost_equal(E_unique2, 1. / 1.5)
    assert_almost_equal(E_per_sample1, [0., 1. / 1.5])
    assert_almost_equal(E_per_sample2, [1. / 1.5, 1. / 1.5])


@testing.requires_testing_data
def test_roc_auc_metric():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:4], []]
    vert2 = [src[0]['vertno'][0:4], []]
    data1 = np.array([[0., 0., 1, 1]]).T
    data2 = np.array([[0.1, -0.4, 0.35, 0.8]]).T
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')

    score = stc_roc_auc_score(stc_true, stc_est, per_sample=False)
    assert_almost_equal(score, 0.75)


run_tests_if_main()
