# -*- coding: utf-8 -*-
#
# Authors: Dirk Gütlin <dirk.guetlin@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import os.path as op

import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_allclose, assert_equal)
import pytest
from mne.utils import _TempDir, requires_h5py, run_tests_if_main
from mne.source_tfr import SourceTFR

rnd = np.random.RandomState(23)

@pytest.fixture(scope="module")
def fake_stfr():
    """Create a fake SourceTFR object for testing."""
    verts = [np.arange(10), np.arange(90)]
    return SourceTFR(rnd.rand(100, 20, 10), verts, 0, 1e-1, 'foo')


@pytest.fixture(scope="module")
def fake_kernel_stfr():
    """Create a fake kernel SourceTFR object for testing."""
    kernel = rnd.rand(100, 40)
    sens_data = rnd.rand(40, 20, 10)
    verts = [np.arange(10), np.arange(90)]
    return SourceTFR((kernel, sens_data), verts, 0, 1e-1, 'foo')


def test_stfr_kernel_equality(fake_stfr, fake_kernel_stfr):
    """Test if kernelized SourceTFR produce correct data."""
    # compare kernelized and normal data
    kernel = rnd.rand(100, 40)
    sens_data = rnd.rand(40, 10, 30)
    verts = [np.arange(10), np.arange(90)]
    data = np.tensordot(kernel, sens_data, axes=([-1], [0]))
    tmin = 0
    tstep = 1e-3

    kernel_stfr = SourceTFR((kernel, sens_data), verts, tmin, tstep)
    full_stfr = SourceTFR(data, verts, tmin, tstep)

    # check if data is in correct shape
    assert kernel_stfr.shape == (100, 10, 30)
    assert full_stfr.shape == (100, 10, 30)
    assert kernel_stfr.data.shape == (100, 10, 30)
    assert full_stfr.data.shape == (100, 10, 30)
    assert_allclose(kernel_stfr.data, full_stfr.data)

    stfr = fake_stfr
    kernel_stfr = fake_kernel_stfr

    # alternatively with the fake data
    assert_equal(stfr.shape, kernel_stfr.shape)
    assert_array_equal(stfr.data.shape, kernel_stfr.data.shape)


def test_stfr_attributes(fake_stfr):
    """Test stfr attributes."""
    stfr = fake_stfr.copy()

    n_times = len(stfr.times)
    assert_equal(stfr._data.shape[-1], n_times)
    assert_array_equal(stfr.times, stfr.tmin + np.arange(n_times) * stfr.tstep)

    assert_allclose(stfr.times,
                    [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def attempt_times_mutation(stfr):
        stfr.times -= 1

    def attempt_assignment(stfr, attr, val):
        setattr(stfr, attr, val)

    # .times is read-only
    with pytest.raises(RuntimeError,
                       match="cannot write to the .times attribute directly"):
        attempt_times_mutation(stfr)
    with pytest.raises(RuntimeError,
                       match="cannot write to the .times attribute directly"):
        attempt_assignment(stfr, "times", [1])

    # Changing .tmin or .tstep re-computes .times
    stfr.tmin = 1
    assert type(stfr.tmin) == float
    assert_allclose(stfr.times,
                    [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

    stfr.tstep = 1
    assert (type(stfr.tstep) == float)
    assert_allclose(stfr.times,
                    [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # tstep <= 0 is not allowed
    with pytest.raises(ValueError, match="must be greater than 0"):
        attempt_assignment(stfr, 'tstep', 0)
    with pytest.raises(ValueError, match="must be greater than 0"):
        attempt_assignment(stfr, 'tstep', -1)

    # Changing .data re-computes .times
    stfr.data = rnd.rand(100, 20, 5)
    assert_allclose(stfr.times, [1., 2., 3., 4., 5.])

    # .data must match the number of vertices
    with pytest.raises(ValueError, match="must match the number of vertices"):
        attempt_assignment(stfr, "data", [[[1]]])

    # .data much match number of dimensions
    assign = [None, np.arange(100), [np.arange(100)], [[[np.arange(100)]]]]
    for val in assign:
        with pytest.raises(ValueError,
                           match="Data.*? should have.*? dimensions"):
            attempt_assignment(stfr, "data", val)

    # .shape attribute must also work when ._data is None
    stfr._kernel = np.zeros((2, 2))
    stfr._sens_data = np.zeros((2, 5, 3))
    stfr._data = None
    assert_equal(stfr.shape, (2, 5, 3))

    # bad size of data
    stfr = fake_stfr
    data = stfr.data[:, :, np.newaxis, :]
    with pytest.raises(ValueError, match='3 dimensions for SourceTFR'):
        SourceTFR(data, stfr.vertices)
    stfr = SourceTFR(data[:, :, :, 0], stfr.vertices, 0, 1)
    assert stfr.data.shape == (data.shape[0], data.shape[1], 1)


@requires_h5py
def test_io_stfr_h5(fake_stfr, fake_kernel_stfr):
    """Test IO for stfr files using HDF5."""
    for stfr in [fake_stfr, fake_kernel_stfr]:
        tempdir = _TempDir()
        with pytest.raises(ValueError, match="can only be written as HDF5"):
            stfr.save(op.join(tempdir, 'tmp'), ftype='foo')
        out_name = op.join(tempdir, 'tempfile')
        stfr.save(out_name, ftype='h5')
        stfr.save(out_name, ftype='h5')  # test overwrite
        # TODO: no read_source_tfr yet


def test_stfr_resample(fake_stfr, fake_kernel_stfr):
    """Test sftr.resample()."""
    stfr_ = fake_stfr
    kernel_stfr_ = fake_kernel_stfr

    for stfr in [stfr_, kernel_stfr_]:
        stfr_new = deepcopy(stfr)
        o_sfreq = 1.0 / stfr.tstep
        # note that using no padding for this stfr reduces edge ringing...
        stfr_new.resample(2 * o_sfreq, npad=0)
        assert stfr_new.data.shape[-1] == 2 * stfr.data.shape[-1]
        assert stfr_new.tstep == stfr.tstep / 2
        stfr_new.resample(o_sfreq, npad=0)
        assert stfr_new.data.shape[-1] == stfr.data.shape[-1]
        assert stfr_new.tstep == stfr.tstep
        assert_allclose(stfr_new.data, stfr.data, 5)


def test_stfr_crop(fake_stfr, fake_kernel_stfr):
    """Test cropping of SourceTFR data."""
    stfr = fake_stfr
    kernel_stfr = fake_kernel_stfr

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
    data = rnd.rand(40, 10, 20)
    verts = [np.arange(10), np.arange(30)]
    tmin = 0
    tstep = 1e-3

    with pytest.raises(TypeError, match="vertices must be an instance of "
                                        "ndarray or list"):
        SourceTFR(data, {"1": 1, "2": 2}, tmin, tstep)

    with pytest.raises(ValueError,
                       match='data.*? tuple .*? has to be length 2'):
        SourceTFR((data, (42, 42), (42, 42)), verts, tmin, tstep)

    with pytest.raises(ValueError, match='last kernel.*? first data dimension'
                                         ' must be of equal size'):
        SourceTFR((np.zeros((42, 42)), data), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='sensor data must have .*? dimensions'):
        SourceTFR((np.zeros((2, 20)), np.zeros((20, 3))), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='Vertices must be ordered in increasing order.'):
        SourceTFR(data, [np.zeros(10), np.zeros(90)], tmin, tstep)

    with pytest.raises(ValueError,
                       match='vertices .*? and stfr.shape.*? must match'):
        SourceTFR(np.ones([42, 10, 20]), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='(shape .*?) must have .*? dimensions'):
        SourceTFR(np.ones([40, 10, 20, 10]), verts, tmin, tstep)

    with pytest.raises(ValueError,
                       match='multiple orientations.*? must be 3'):
        SourceTFR(np.ones([40, 10, 20, 10]), verts, tmin, tstep,
                  dims=("dipoles", "orientations", "freqs", "times"))

    with pytest.raises(ValueError,
                       match="Invalid value for the 'dims' parameter"):
        SourceTFR(data, verts, tmin, tstep, dims=("dipoles", "nonsense"))

    with pytest.raises(ValueError,
                       match="Invalid value for the 'method' parameter"):
        SourceTFR(data, verts, tmin, tstep, method="invalid")

    with pytest.raises(ValueError,
                       match="Invalid value for the 'src_type' parameter"):
        SourceTFR(data, verts, tmin, tstep, src_type="invalid")


run_tests_if_main()
