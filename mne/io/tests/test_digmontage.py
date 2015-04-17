import os.path as op
import inspect
from nose.tools import assert_equal

import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_allclose)

from mne.channels import apply_montage
from mne.io.meas_info import read_digmontage, _read_dig_points
from mne.utils import _TempDir
from mne import create_info
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
from mne.coreg import fit_matched_points
from mne.io.kit import read_mrk


FILE = inspect.getfile(inspect.currentframe())
p_dir = op.dirname(op.abspath(FILE))
elp = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_elp.txt')
hsp = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_hsp.txt')
hpi = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_mrk.sqd')
elp_names = []

def test_read_digmontage():
    """Test read_digmontage"""
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    montage = read_digmontage(hsp, hpi, elp, names, unit='m', transform=False)
    elp_points = _read_dig_points(elp)
    hsp_points = _read_dig_points(hsp)
    hpi_points = read_mrk(hpi)
    assert_equal(montage.point_names, names)
    assert_array_equal(montage.elp, elp_points)
    assert_array_equal(montage.hsp, hsp_points)
    assert_array_equal(montage.hpi, hpi_points)
    assert_array_equal(montage.dev_head_t, np.identity(4))
    montage = read_digmontage(hsp, hpi, elp, names,
                              transform=True, dev_head_t=True)
    # check coordinate transformation
    # nasion
    assert_almost_equal(montage.elp[0, 0], 0)
    assert_almost_equal(montage.nasion[0], 0)
    assert_almost_equal(montage.elp[0, 2], 0)
    assert_almost_equal(montage.nasion[0], 0)
    # lpa and rpa
    assert_allclose(montage.elp[1:3, 1:], 0, atol=1e-16)
    assert_allclose(montage.lpa[1:], 0, atol=1e-16)
    assert_allclose(montage.rpa[1:], 0, atol=1e-16)
    # device head transform
    dev_head_t = fit_matched_points(tgt_pts=montage.elp[3:],
                                    src_pts=montage.hpi, out='trans')
    assert_array_equal(montage.dev_head_t, dev_head_t)
