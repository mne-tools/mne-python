# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_raises, assert_equal, assert_true
import warnings

from mne.io import read_info, Raw
from mne.chpi import _rot_to_quat, _quat_to_rot, get_chpi_positions
from mne.utils import run_tests_if_main, _TempDir
from mne.datasets import testing

base_dir = op.join(op.dirname(__file__), '..', 'tests', 'data')
test_fif_fname = op.join(base_dir, 'test_raw.fif')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
hp_fname = op.join(base_dir, 'test_chpi_raw_hp.txt')

data_path = testing.data_path(download=False)
raw_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
sss_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_sss.fif')

warnings.simplefilter('always')


def test_quaternions():
    """Test quaternion calculations
    """
    rots = [np.eye(3)]
    for fname in [test_fif_fname, ctf_fname, hp_fif_fname]:
        rots += [read_info(fname)['dev_head_t']['trans'][:3, :3]]
    for rot in rots:
        assert_allclose(rot, _quat_to_rot(_rot_to_quat(rot)),
                        rtol=1e-5, atol=1e-5)
        rot = rot[np.newaxis, np.newaxis, :, :]
        assert_allclose(rot, _quat_to_rot(_rot_to_quat(rot)),
                        rtol=1e-5, atol=1e-5)


def test_get_chpi():
    """Test CHPI position computation
    """
    trans0, rot0 = get_chpi_positions(hp_fname)[:2]
    trans0, rot0 = trans0[:-1], rot0[:-1]
    raw = Raw(hp_fif_fname)
    out = get_chpi_positions(raw)
    trans1, rot1, t1 = out
    trans1, rot1 = trans1[2:], rot1[2:]
    # these will not be exact because they don't use equiv. time points
    assert_allclose(trans0, trans1, atol=1e-5, rtol=1e-1)
    assert_allclose(rot0, rot1, atol=1e-6, rtol=1e-1)
    # run through input checking
    assert_raises(TypeError, get_chpi_positions, 1)
    assert_raises(ValueError, get_chpi_positions, hp_fname, [1])


@testing.requires_testing_data
def test_hpi_info():
    """Test getting HPI info
    """
    tempdir = _TempDir()
    temp_name = op.join(tempdir, 'temp_raw.fif')
    for fname in (raw_fif_fname, sss_fif_fname):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            raw = Raw(fname, allow_maxshield=True)
        assert_true(len(raw.info['hpi_subsystem']) > 0)
        raw.save(temp_name, overwrite=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            raw_2 = Raw(temp_name, allow_maxshield=True)
        assert_equal(len(raw_2.info['hpi_subsystem']),
                     len(raw.info['hpi_subsystem']))

run_tests_if_main(False)
