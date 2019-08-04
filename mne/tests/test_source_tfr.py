from copy import deepcopy
import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest
from mne.utils import _TempDir, requires_h5py, run_tests_if_main
from mne.source_tfr import SourceTFR


def _fake_stfr():
    """Create a fake SourceTFR object for testing."""
    verts = [np.arange(10), np.arange(90)]
    return SourceTFR(np.random.rand(100, 20, 10), verts, 0, 1e-1, 'foo')


def _fake_kernel_stfr():
    """Create a fake kernel SourceTFR object for testing."""
    kernel = np.random.rand(100, 40)
    sens_data = np.random.rand(40, 20, 10)
    verts = [np.arange(10), np.arange(90)]
    return SourceTFR((kernel, sens_data), verts, 0, 1e-1, 'foo')


def test_stfr_kernel_equality():
    """Test if kernelized SourceTFR produce correct data."""
    # compare kernelized and normal data
    kernel = np.random.rand(100, 40)
    sens_data = np.random.rand(40, 10, 30)
    verts = [np.arange(10), np.arange(90)]
    data = np.tensordot(kernel, sens_data, axes=([-1], [0]))
    tmin = 0
    tstep = 1e-3

    kernel_stfr = SourceTFR((kernel, sens_data), verts, tmin, tstep)
    full_stfr = SourceTFR(data, verts, tmin, tstep)

    # check if data is in correct shape
    expected = [100, 10, 30]
    assert_allclose([kernel_stfr.shape, full_stfr.shape,
                     kernel_stfr.data.shape, full_stfr.data.shape],
                    [expected] * 4)
    assert_allclose(kernel_stfr.data, full_stfr.data)

    # alternatively with the fake data
    assert_equal(_fake_stfr().shape, _fake_kernel_stfr().shape)
    assert_array_equal(_fake_stfr().data.shape, _fake_kernel_stfr().data.shape)


def test_stfr_attributes():
    """Test stfr attributes."""
    stfr = _fake_stfr()
    stfr_kernel = _fake_kernel_stfr()

    n_times = len(stfr.times)
    assert_equal(stfr._data.shape[-1], n_times)
    assert_array_equal(stfr.times, stfr.tmin + np.arange(n_times) * stfr.tstep)

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
    data = stfr.data[:, :, np.newaxis, :]
    with pytest.raises(ValueError, match='3 dimensions for SourceTFR'):
        SourceTFR(data, stfr.vertices)
    stfr = SourceTFR(data[:, :, :, 0], stfr.vertices, 0, 1)
    assert stfr.data.shape == (data.shape[0], data.shape[1], 1)


@requires_h5py
def test_io_stfr_h5():
    """Test IO for stfr files using HDF5."""
    for stfr in [_fake_stfr(), _fake_kernel_stfr()]:
        tempdir = _TempDir()
        pytest.raises(ValueError, stfr.save, op.join(tempdir, 'tmp'),
                      ftype='foo')
        out_name = op.join(tempdir, 'tempfile')
        stfr.save(out_name, ftype='h5')
        stfr.save(out_name, ftype='h5')  # test overwrite
        # TODO: no read_source_tfr yet


def test_stfr_resample():
    """Test sftr.resample()."""
    stfr_ = _fake_stfr()
    kernel_stfr_ = _fake_kernel_stfr()

    for stfr in [stfr_, kernel_stfr_]:
        stfr_new = deepcopy(stfr)
        o_sfreq = 1.0 / stfr.tstep
        # note that using no padding for this stfr reduces edge ringing...
        stfr_new.resample(2 * o_sfreq, npad=0)
        assert (stfr_new.data.shape[-1] == 2 * stfr.data.shape[-1])
        assert (stfr_new.tstep == stfr.tstep / 2)
        stfr_new.resample(o_sfreq, npad=0)
        assert (stfr_new.data.shape[-1] == stfr.data.shape[-1])
        assert (stfr_new.tstep == stfr.tstep)
        assert_array_almost_equal(stfr_new.data, stfr.data, 5)


def test_stfr_crop():
    """Test cropping of SourceTFR data."""
    stfr = _fake_stfr()
    kernel_stfr = _fake_kernel_stfr()

    for inst in [stfr, kernel_stfr]:
        copy_1 = inst.copy()
        assert_allclose(copy_1.crop(tmax=0.8).data, inst.data[:, :, :9])
        # FIXME: cropping like this does not work for kernelized  stfr/stc
        # assert_allclose(copy_1.times, inst.times[:9])

        copy_2 = inst.copy()
        assert_allclose(copy_2.crop(tmin=0.2).data, inst.data[:, :, 2:])
        assert_allclose(copy_2.times, inst.times[2:])


def test_invalid_params():
    """Test invalid SourceTFR parameters."""
    data = np.random.rand(40, 10, 20)
    verts = [np.arange(10), np.arange(30)]
    tmin = 0
    tstep = 1e-3

    with pytest.raises(ValueError, match='Vertices must be a numpy array '
                                         'or a list of arrays'):
        SourceTFR(data, {"1": 1, "2": 2}, tmin, tstep)

    with pytest.raises(ValueError,
                       match='tuple it has to be length 2'):
        SourceTFR((data, (42, 42), (42, 42)), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='kernel and sens_data have invalid dimension'):
        SourceTFR((np.zeros((42, 42)), data), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='sensor data must have .*? dimensions'):
        SourceTFR((np.zeros((2, 20)), np.zeros((20, 3))), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='Vertices must be ordered in increasing order.'):
        SourceTFR(data, [np.zeros(10), np.zeros(90)], tmin, tstep)

    with pytest.raises(ValueError,
                       match='vertices .*? and stfr.shape.*? must match'):
        SourceTFR(np.random.rand(42, 10, 20), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='(shape .*?) must have .*? dimensions'):
        SourceTFR(np.random.rand(40, 10, 20, 10), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='multiple orientations.*? must be 3'):
        SourceTFR(np.random.rand(40, 10, 20, 10), verts, tmin, tstep,
                  dims=("dipoles", "orientations", "freqs", "times"))

    with pytest.raises(ValueError,
                       match="Invalid value for the 'dims' parameter"):
        SourceTFR(data, verts, tmin, tstep, dims=("dipoles", "nonsense"))

    with pytest.raises(ValueError,
                       match="Invalid value for the 'method' parameter"):
        SourceTFR(data, verts, tmin, tstep, method="multitape")


run_tests_if_main()
