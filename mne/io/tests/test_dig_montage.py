# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import inspect
from nose.tools import assert_equal

import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_allclose)

from mne.channels.layout import _set_montage
from mne.io.meas_info import read_dig_montage, _read_dig_points
from mne.io.constants import FIFF
from mne import create_info
from mne.coreg import fit_matched_points
from mne.io.kit import read_mrk
from mne.transforms import apply_trans, get_ras_to_neuromag_trans


FILE = inspect.getfile(inspect.currentframe())
p_dir = op.dirname(op.abspath(FILE))
elp = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_elp.txt')
hsp = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_hsp.txt')
hpi = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_mrk.sqd')
elp_names = []


def test_read_dig_montage():
    """Test read_dig_montage"""
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    montage = read_dig_montage(hsp, hpi, elp, names, unit='m', transform=False)
    elp_points = _read_dig_points(elp)
    hsp_points = _read_dig_points(hsp)
    hpi_points = read_mrk(hpi)
    assert_equal(montage.point_names, names)
    assert_array_equal(montage.elp, elp_points)
    assert_array_equal(montage.hsp, hsp_points)
    assert_array_equal(montage.hpi, hpi_points)
    assert_array_equal(montage.dev_head_t, np.identity(4))
    montage = read_dig_montage(hsp, hpi, elp, names,
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


def test_set_dig_montage():
    """Test applying DigMontage to inst

    Extensive testing of applying `dig` to info is done in test_meas_info
    with `test_make_dig_points`.
    """
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    hsp_points = _read_dig_points(hsp)
    elp_points = _read_dig_points(elp)
    hpi_points = read_mrk(hpi)
    p0, p1, p2 = elp_points[:3]
    nm_trans = get_ras_to_neuromag_trans(p0, p1, p2)
    elp_points = apply_trans(nm_trans, elp_points)
    nasion_point, lpa_point, rpa_point = elp_points[:3]
    hsp_points = apply_trans(nm_trans, hsp_points)

    montage = read_dig_montage(hsp, hpi, elp, names, unit='m', transform=True)
    info = create_info(['Test Ch'], 1e3, ['eeg'])
    _set_montage(info, montage)
    hs = np.array([p['r'] for i, p in enumerate(info['dig'])
                   if p['kind'] == FIFF.FIFFV_POINT_EXTRA])
    nasion_dig = np.array([p['r'] for p in info['dig']
                           if all([p['ident'] == FIFF.FIFFV_POINT_NASION,
                                   p['kind'] == FIFF.FIFFV_POINT_CARDINAL])])
    lpa_dig = np.array([p['r'] for p in info['dig']
                        if all([p['ident'] == FIFF.FIFFV_POINT_LPA,
                                p['kind'] == FIFF.FIFFV_POINT_CARDINAL])])
    rpa_dig = np.array([p['r'] for p in info['dig']
                        if all([p['ident'] == FIFF.FIFFV_POINT_RPA,
                                p['kind'] == FIFF.FIFFV_POINT_CARDINAL])])
    hpi_dig = np.array([p['r'] for p in info['dig']
                        if p['kind'] == FIFF.FIFFV_POINT_HPI])
    assert_array_equal(hs, hsp_points)
    assert_array_equal(nasion_dig.ravel(), nasion_point)
    assert_array_equal(lpa_dig.ravel(), lpa_point)
    assert_array_equal(rpa_dig.ravel(), rpa_point)
    assert_array_equal(hpi_dig, hpi_points)
    assert_array_equal(montage.dev_head_t, info['dev_head_t']['trans'])
