
# Author: Luke Bloy <bloyl@chop.edu>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

from mne.io import read_raw_artemis123
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets import testing
from mne.io.artemis123.utils import _generate_mne_locs_file, _load_mne_locs
from mne import pick_types
from mne.transforms import rot_to_quat, _angle_between_quats
from mne.io.constants import FIFF

artemis123_dir = op.join(testing.data_path(download=False), 'ARTEMIS123')

short_HPI_dip_fname = op.join(artemis123_dir,
                              'Artemis_Data_2017-04-04-15h-44m-' +
                              '22s_Motion_Translation-z.bin')

dig_fname = op.join(artemis123_dir, 'Phantom_040417_dig.pos')

short_hpi_1kz_fname = op.join(artemis123_dir, 'Artemis_Data_2017-04-14-10h' +
                              '-38m-59s_Phantom_1k_HPI_1s.bin')


# XXX this tol is way too high, but it's not clear which is correct
# (old or new)
def _assert_trans(actual, desired, dist_tol=0.017, angle_tol=5.):
    __tracebackhide__ = True
    trans_est = actual[0:3, 3]
    quat_est = rot_to_quat(actual[0:3, 0:3])
    trans = desired[0:3, 3]
    quat = rot_to_quat(desired[0:3, 0:3])

    angle = np.rad2deg(_angle_between_quats(quat_est, quat))
    dist = np.linalg.norm(trans - trans_est)
    assert dist <= dist_tol, \
        '%0.3f > %0.3f mm translation' % (1000 * dist, 1000 * dist_tol)
    assert angle <= angle_tol, \
        '%0.3f > %0.3f° rotation' % (angle, angle_tol)


@pytest.mark.timeout(60)  # ~25 sec on Travis Linux OpenBLAS
@testing.requires_testing_data
def test_artemis_reader():
    """Test reading raw Artemis123 files."""
    _test_raw_reader(read_raw_artemis123, input_fname=short_hpi_1kz_fname,
                     pos_fname=dig_fname, verbose='error')


@pytest.mark.timeout(60)
@testing.requires_testing_data
def test_dev_head_t():
    """Test dev_head_t computation for Artemis123."""
    # test a random selected point
    raw = read_raw_artemis123(short_hpi_1kz_fname, preload=True,
                              add_head_trans=False)
    meg_picks = pick_types(raw.info, meg=True, eeg=False)

    # checked against matlab reader.
    assert_allclose(raw[meg_picks[12]][0][0][123], 1.08239606023e-11)

    dev_head_t_1 = np.array([[9.713e-01, 2.340e-01, -4.164e-02, 1.302e-04],
                             [-2.371e-01, 9.664e-01, -9.890e-02, 1.977e-03],
                             [1.710e-02, 1.059e-01, 9.942e-01, -8.159e-03],
                             [0.0, 0.0, 0.0, 1.0]])

    dev_head_t_2 = np.array([[9.890e-01, 1.475e-01, -8.090e-03, 4.997e-04],
                             [-1.476e-01, 9.846e-01, -9.389e-02, 1.962e-03],
                             [-5.888e-03, 9.406e-02, 9.955e-01, -1.610e-02],
                             [0.0, 0.0, 0.0, 1.0]])

    expected_dev_hpi_rr = np.array([[-0.01579644, 0.06527367, 0.00152648],
                                    [0.06666813, 0.0148956, 0.00545488],
                                    [-0.06699212, -0.01732376, 0.0112027]])
    # test with head loc no digitization
    raw = read_raw_artemis123(short_HPI_dip_fname, add_head_trans=True)
    _assert_trans(raw.info['dev_head_t']['trans'], dev_head_t_1)
    assert_equal(raw.info['sfreq'], 5000.0)

    # test with head loc and digitization
    with pytest.warns(RuntimeWarning, match='Large difference'):
        raw = read_raw_artemis123(short_HPI_dip_fname, add_head_trans=True,
                                  pos_fname=dig_fname)
    _assert_trans(raw.info['dev_head_t']['trans'], dev_head_t_1)

    # test cHPI localization..
    dev_hpi_rr = np.array([p['r'] for p in raw.info['dig']
                           if p['coord_frame'] == FIFF.FIFFV_COORD_DEVICE])
    # points should be within 0.1 mm (1e-4m) and within 1%
    assert_allclose(dev_hpi_rr, expected_dev_hpi_rr, atol=1e-4, rtol=0.01)

    # test 1kz hpi head loc (different freq)
    raw = read_raw_artemis123(short_hpi_1kz_fname, add_head_trans=True)
    _assert_trans(raw.info['dev_head_t']['trans'], dev_head_t_2)
    assert_equal(raw.info['sfreq'], 1000.0)


def test_utils(tmp_path):
    """Test artemis123 utils."""
    # make a tempfile
    tmp_dir = str(tmp_path)
    tmp_fname = op.join(tmp_dir, 'test_gen_mne_locs.csv')
    _generate_mne_locs_file(tmp_fname)
    installed_locs = _load_mne_locs()
    generated_locs = _load_mne_locs(tmp_fname)
    assert_equal(set(installed_locs.keys()), set(generated_locs.keys()))
    for key in installed_locs.keys():
        assert_allclose(installed_locs[key], generated_locs[key], atol=1e-7)
