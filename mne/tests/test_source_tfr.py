from copy import deepcopy
import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest
from scipy.fftpack import fft
from scipy import sparse

from mne.datasets import testing
from mne import (stats, SourceEstimate, VectorSourceEstimate,
                 VolSourceEstimate, Label, read_source_spaces,
                 read_evokeds, MixedSourceEstimate, find_events, Epochs,
                 read_source_estimate, extract_label_time_course,
                 spatio_temporal_tris_connectivity,
                 spatio_temporal_src_connectivity,
                 spatial_inter_hemi_connectivity,
                 spatial_src_connectivity, spatial_tris_connectivity,
                 SourceSpaces, VolVectorSourceEstimate)
from mne.source_estimate import grade_to_tris, _get_vol_mask

from mne.minimum_norm import (read_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
from mne.label import read_labels_from_annot, label_sign_flip
from mne.utils import (_TempDir, requires_pandas, requires_sklearn,
                       requires_h5py, run_tests_if_main, requires_nibabel)
from mne.io import read_raw_fif
from mne.source_tfr import SourceTFR


def test_source_tfr():
    # compare kernelized and normal data shapes
    kernel_stfr = SourceTFR((np.ones([1800, 300]), np.ones([300, 40, 30])),
                            vertices=np.ones([1800, 1]), tmin=0, tstep=1)

    full_stfr = SourceTFR(np.ones([1800, 40, 30]), vertices=np.ones([1800, 1]), tmin=0, tstep=1)

    assert_equal(_fake_stfr().shape, _fake_kernel_stfr().shape)

    # check dot product
    kernel = np.random.rand(100, 40)
    sens_data = np.random.rand(40, 10, 30)
    verts = [np.arange(10), np.arange(90)]

    assert_allclose(SourceTFR((kernel, sens_data), verts, tmin=0, tstep=1).data,
                    np.tensordot(kernel, sens_data, axes=([-1], [0])))

    # check if data is in correct shape
    assert_equal(kernel_stfr.shape, full_stfr.shape)
    assert_array_equal(kernel_stfr.data.shape, full_stfr.data.shape)

    # alternatively with the fake data
    assert_equal(_fake_stfr().shape, _fake_kernel_stfr().shape)
    assert_array_equal(_fake_stfr().data.shape, _fake_kernel_stfr().data.shape)


def _fake_stfr():
    verts = [np.arange(10), np.arange(90)]
    return SourceTFR(np.random.rand(100, 20, 10), verts, 0, 1e-1, 'foo')


def _fake_kernel_stfr():
    kernel = np.random.rand(100, 40)
    sens_data = np.random.rand(40, 20, 10)
    verts = [np.arange(10), np.arange(90)]
    return SourceTFR((kernel, sens_data), verts, 0, 1e-1, 'foo')


def test_stfr_attributes():
    """Test STC attributes."""
    stfr = _fake_stfr()
    stfr_kernel = _fake_kernel_stfr()

    assert_array_almost_equal(
        stfr.times, [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def attempt_times_mutation(stfr):
        stfr.times -= 1

    def attempt_assignment(stfr, attr, val):
        setattr(stfr, attr, val)

    # .times is read-only
    pytest.raises(ValueError, attempt_times_mutation, stfr)
    pytest.raises(ValueError, attempt_assignment, stfr, 'times', [1])

    # Changing .tmin or .tstep re-computes .times
    stfr.tmin = 1
    assert (type(stfr.tmin) == float)
    assert_array_almost_equal(
        stfr.times, [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

    stfr.tstep = 1
    assert (type(stfr.tstep) == float)
    assert_array_almost_equal(
        stfr.times, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # tstep <= 0 is not allowed
    pytest.raises(ValueError, attempt_assignment, stfr, 'tstep', 0)
    pytest.raises(ValueError, attempt_assignment, stfr, 'tstep', -1)

    # Changing .data re-computes .times
    stfr.data = np.random.rand(100, 20, 5)
    assert_array_almost_equal(
        stfr.times, [1., 2., 3., 4., 5.])

    # .data must match the number of vertices
    pytest.raises(ValueError, attempt_assignment, stfr, 'data', [[1]])
    pytest.raises(ValueError, attempt_assignment, stfr, 'data', None)

    # .data much match number of dimensions
    pytest.raises(ValueError, attempt_assignment, stfr, 'data', np.arange(100))
    pytest.raises(ValueError, attempt_assignment, stfr_kernel, 'data',
                  [np.arange(100)])
    pytest.raises(ValueError, attempt_assignment, stfr_kernel, 'data',
                  [[[np.arange(100)]]])

    # .shape attribute must also work when ._data is None
    stfr._kernel = np.zeros((2, 2))
    stfr._sens_data = np.zeros((2, 5, 3))
    stfr._data = None
    assert_equal(stfr.shape, (2, 5, 3))

    # bad size of data
    stfr = _fake_stfr()
    data = stfr.data[:, np.newaxis, :, :]
    with pytest.raises(ValueError, match='3 dimensions for SourceTFR'):
        SourceTFR(data, stfr.vertices)
    # TODO: check this
    # stfr = SourceTFR(data[:, 0, 0], stfr.vertices, 0, 1)
    # assert stfr.data.shape == (len(data), 1)
