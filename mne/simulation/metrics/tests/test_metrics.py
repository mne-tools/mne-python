# Authors: Kostiantyn Maksymenko <kostiantyn.maksymenko@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import norm

from mne import SourceEstimate
from mne import read_source_spaces
from mne.datasets import testing
from mne.utils import requires_sklearn
from mne.simulation import metrics
from mne.simulation.metrics import (cosine_score,
                                    region_localization_error,
                                    precision_score, recall_score,
                                    f1_score, roc_auc_score,
                                    peak_position_error,
                                    spatial_deviation_error)

data_path = testing.data_path(download=False)
src_fname = data_path / 'subjects' / 'sample' / 'bem' / 'sample-oct-6-src.fif'


@testing.requires_testing_data
def test_uniform_and_thresholding():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert = [src[0]['vertno'][0:1], []]
    data = np.array([[0.8, -1.]])
    stc_true = SourceEstimate(data, vert, 0, 0.002, subject='sample')
    stc_bad = SourceEstimate(data, vert, 0, 0.002, subject='sample')
    stc_bad.vertices = [stc_bad.vertices[0]]
    with pytest.raises(ValueError, match='same number of vertices'):
        metrics._uniform_stc(stc_true, stc_bad)

    threshold = 0.9
    stc1, stc2 = metrics._thresholding(stc_true, stc_true, threshold)
    assert_allclose(stc1._data, np.array([[0, -1.]]))
    assert_allclose(stc2._data, np.array([[0, -1.]]))
    assert_allclose(threshold, metrics._check_threshold(threshold))

    threshold = '90'
    with pytest.raises(ValueError, match='Threshold if a str.*'):
        metrics._check_threshold(threshold)


@testing.requires_testing_data
def test_cosine_score():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:1], []]
    vert2 = [src[0]['vertno'][1:2], []]
    data1 = np.ones((1, 2))
    data2 = data1.copy()
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data2, vert1, 0, 0.002, subject='sample')

    E_per_sample1 = cosine_score(stc_true, stc_est1)
    E_unique1 = cosine_score(stc_true, stc_est1, per_sample=False)

    E_per_sample2 = cosine_score(stc_true, stc_est2)
    E_unique2 = cosine_score(stc_true, stc_est2, per_sample=False)

    assert_allclose(E_per_sample1, np.zeros(2))
    assert_allclose(E_unique1, 0., atol=1e-08)
    assert_allclose(E_per_sample2, np.ones(2))
    assert_allclose(E_unique2, 1., atol=1e-08)


@testing.requires_testing_data
@requires_sklearn
def test_region_localization_error():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:1], []]
    vert2 = [src[0]['vertno'][1:2], []]
    dist = norm(src[0]['rr'][vert1[0]] - src[0]['rr'][vert2[0]])
    data1 = np.ones((1, 2))
    data2 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')

    E_per_sample1 = region_localization_error(stc_true, stc_est1, src)
    E_per_sample2 = region_localization_error(stc_true, stc_est1, src,
                                              threshold='70%')
    E_unique = region_localization_error(stc_true, stc_est1, src,
                                         per_sample=False)

    assert_allclose(E_per_sample1, [np.inf, dist])
    assert_allclose(E_per_sample2, [dist, dist])
    assert_allclose(E_unique, dist)


@testing.requires_testing_data
@requires_sklearn
def test_precision_score():
    """Test simulation metrics."""
    from sklearn.exceptions import UndefinedMetricWarning
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:2], []]
    vert2 = [src[0]['vertno'][1:3], []]
    vert3 = [src[0]['vertno'][0:1], []]
    data1 = np.ones((2, 2))
    data2 = np.ones((2, 2))
    data3 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data3, vert3, 0, 0.002, subject='sample')

    E_unique1 = precision_score(stc_true, stc_est1, per_sample=False)
    E_unique2 = precision_score(stc_true, stc_est2, per_sample=False)
    with pytest.warns(UndefinedMetricWarning, match='no predicted samples'):
        E_per_sample1 = precision_score(stc_true, stc_est2)
    E_per_sample2 = precision_score(stc_true, stc_est2,
                                    threshold='70%')
    with pytest.raises(ValueError, match='0 and 1'):
        precision_score(stc_true, stc_est2, threshold=2)

    # ### Tests to add
    assert_allclose(E_unique1, 0.5)
    assert_allclose(E_unique2, 1.)
    assert_allclose(E_per_sample1, [0., 1.])
    assert_allclose(E_per_sample2, [1., 1.])


@testing.requires_testing_data
@requires_sklearn
def test_recall_score():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:2], []]
    vert2 = [src[0]['vertno'][1:3], []]
    vert3 = [src[0]['vertno'][0:1], []]
    data1 = np.ones((2, 2))
    data2 = np.ones((2, 2))
    data3 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data3, vert3, 0, 0.002, subject='sample')

    E_unique1 = recall_score(stc_true, stc_est1, per_sample=False)
    E_unique2 = recall_score(stc_true, stc_est2, per_sample=False)
    E_per_sample1 = recall_score(stc_true, stc_est2)
    E_per_sample2 = recall_score(stc_true, stc_est2, threshold='70%')
    with pytest.raises(TypeError, match='numeric'):
        precision_score(stc_true, stc_est2, threshold=None)

    # ### Tests to add
    assert_allclose(E_unique1, 0.5)
    assert_allclose(E_unique2, 0.5)
    assert_allclose(E_per_sample1, [0., 0.5])
    assert_allclose(E_per_sample2, [0.5, 0.5])


@testing.requires_testing_data
@requires_sklearn
def test_f1_score():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:2], []]
    vert2 = [src[0]['vertno'][1:3], []]
    vert3 = [src[0]['vertno'][0:1], []]
    data1 = np.ones((2, 2))
    data2 = np.ones((2, 2))
    data3 = np.array([[0.8, 1]])
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est1 = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    stc_est2 = SourceEstimate(data3, vert3, 0, 0.002, subject='sample')

    E_unique1 = f1_score(stc_true, stc_est1, per_sample=False)
    E_unique2 = f1_score(stc_true, stc_est2, per_sample=False)
    E_per_sample1 = f1_score(stc_true, stc_est2)
    E_per_sample2 = f1_score(stc_true, stc_est2, threshold='70%')
    assert_allclose(E_unique1, 0.5)
    assert_allclose(E_unique2, 1. / 1.5)
    assert_allclose(E_per_sample1, [0., 1. / 1.5])
    assert_allclose(E_per_sample2, [1. / 1.5, 1. / 1.5])


@testing.requires_testing_data
@requires_sklearn
def test_roc_auc_score():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:4], []]
    vert2 = [src[0]['vertno'][0:4], []]
    data1 = np.array([[0., 0., 1, 1]]).T
    data2 = np.array([[0.1, -0.4, 0.35, 0.8]]).T
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')

    score = roc_auc_score(stc_true, stc_est, per_sample=False)
    assert_allclose(score, 0.75)


@testing.requires_testing_data
def test_peak_position_error():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:1], []]
    vert2 = [src[0]['vertno'][0:2], []]
    data1 = np.array([[1]])
    data2 = np.array([[1, 1.]]).T
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    r_mean = 0.5 * (src[0]['rr'][vert2[0][0]] + src[0]['rr'][vert2[0][1]])
    r_true = src[0]['rr'][vert2[0][0]]
    score = peak_position_error(stc_true, stc_est, src, per_sample=False)

    assert_allclose(score, norm(r_true - r_mean))
    with pytest.raises(ValueError, match='must contain only one dipole'):
        peak_position_error(stc_est, stc_est, src)

    data2 = np.array([[0, 0.]]).T
    stc_est = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    score = peak_position_error(stc_true, stc_est, src, per_sample=False)
    assert_allclose(score, np.inf)


@testing.requires_testing_data
def test_spatial_deviation():
    """Test simulation metrics."""
    src = read_source_spaces(src_fname)
    vert1 = [src[0]['vertno'][0:1], []]
    vert2 = [src[0]['vertno'][0:2], []]
    data1 = np.array([[1]])
    data2 = np.array([[1, 1.]]).T
    stc_true = SourceEstimate(data1, vert1, 0, 0.002, subject='sample')
    stc_est = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    std = np.sqrt(0.5 * (0 + norm(src[0]['rr'][vert2[0][1]] -
                                  src[0]['rr'][vert2[0][0]])**2))
    score = spatial_deviation_error(stc_true, stc_est, src,
                                    per_sample=False)
    assert_allclose(score, std)

    data2 = np.array([[0, 0.]]).T
    stc_est = SourceEstimate(data2, vert2, 0, 0.002, subject='sample')
    score = spatial_deviation_error(stc_true, stc_est, src,
                                    per_sample=False)
    assert_allclose(score, np.inf)
