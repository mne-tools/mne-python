# Author: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_equal

import numpy as np
from numpy.testing import (assert_array_equal, assert_almost_equal,
                           assert_allclose, assert_array_almost_equal)

from mne.channels.montage import read_montage, _set_montage, read_dig_montage
from mne.utils import _TempDir
from mne import create_info, EvokedArray
from mne.coreg import fit_matched_points
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
from mne.io.constants import FIFF
from mne.io.meas_info import _read_dig_points
from mne.io.kit import read_mrk


p_dir = op.dirname(__file__)
elp = op.join(p_dir, '..', '..', 'io', 'kit', 'tests', 'data', 'test_elp.txt')
hsp = op.join(p_dir, '..', '..', 'io', 'kit', 'tests', 'data', 'test_hsp.txt')
hpi = op.join(p_dir, '..', '..', 'io', 'kit', 'tests', 'data', 'test_mrk.sqd')


def test_montage():
    """Test making montages"""
    tempdir = _TempDir()
    # no pep8
    input_str = ["""FidNz 0.00000 10.56381 -2.05108
    FidT9 -7.82694 0.45386 -3.76056
    FidT10 7.82694 0.45386 -3.76056""",
    """// MatLab   Sphere coordinates [degrees]         Cartesian coordinates
    // Label       Theta       Phi    Radius         X         Y         Z       off sphere surface
      E1      37.700     -14.000       1.000    0.7677    0.5934   -0.2419  -0.00000000000000011
      E2      44.600      -0.880       1.000    0.7119    0.7021   -0.0154   0.00000000000000000
      E3      51.700      11.000       1.000    0.6084    0.7704    0.1908   0.00000000000000000""",  # noqa
    """# ASA electrode file
    ReferenceLabel  avg
    UnitPosition    mm
    NumberPositions=    68
    Positions
    -86.0761 -19.9897 -47.9860
    85.7939 -20.0093 -48.0310
    0.0083 86.8110 -39.9830
    Labels
    LPA
    RPA
    Nz
    """,
    """Site  Theta  Phi
    Fp1  -92    -72
    Fp2   92     72
    F3   -60    -51
    """,
    """346
     EEG	      F3	 -62.027	 -50.053	      85
     EEG	      Fz	  45.608	      90	      85
     EEG	      F4	   62.01	  50.103	      85
    """,
    """
    eeg Fp1 -95.0 -31.0 -3.0
    eeg AF7 -81 -59 -3
    eeg AF3 -87 -41 28
    """]
    kinds = ['test.sfp', 'test.csd', 'test.elc', 'test.txt', 'test.elp',
             'test.hpts']
    for kind, text in zip(kinds, input_str):
        fname = op.join(tempdir, kind)
        with open(fname, 'w') as fid:
            fid.write(text)
        montage = read_montage(fname)
        assert_equal(len(montage.ch_names), 3)
        assert_equal(len(montage.ch_names), len(montage.pos))
        assert_equal(montage.pos.shape, (3, 3))
        assert_equal(montage.kind, op.splitext(kind)[0])
        if kind.endswith('csd'):
            dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                     ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                     ('off_sph', 'f8')]
            try:
                table = np.loadtxt(fname, skip_header=2, dtype=dtype)
            except TypeError:
                table = np.loadtxt(fname, skiprows=2, dtype=dtype)
            pos2 = np.c_[table['x'], table['y'], table['z']]
            assert_array_almost_equal(pos2, montage.pos, 4)
    # test transform
    input_str = """
    eeg Fp1 -95.0 -31.0 -3.0
    eeg AF7 -81 -59 -3
    eeg AF3 -87 -41 28
    cardinal 2 -91 0 -42
    cardinal 1 0 -91 -42
    cardinal 3 0 91 -42
    """
    kind = 'test_fid.hpts'
    fname = op.join(tempdir, kind)
    with open(fname, 'w') as fid:
        fid.write(input_str)
    montage = read_montage(op.join(tempdir, 'test_fid.hpts'), transform=True)
    # check coordinate transformation
    pos = np.array([-95.0, -31.0, -3.0])
    nasion = np.array([-91, 0, -42])
    lpa = np.array([0, -91, -42])
    rpa = np.array([0, 91, -42])
    fids = np.vstack((nasion, lpa, rpa))
    trans = get_ras_to_neuromag_trans(fids[0], fids[1], fids[2])
    pos = apply_trans(trans, pos)
    assert_array_equal(montage.pos[0], pos)
    idx = montage.ch_names.index('2')
    assert_array_equal(montage.pos[idx, [0, 2]], [0, 0])
    idx = montage.ch_names.index('1')
    assert_array_equal(montage.pos[idx, [1, 2]], [0, 0])
    idx = montage.ch_names.index('3')
    assert_array_equal(montage.pos[idx, [1, 2]], [0, 0])
    pos = np.array([-95.0, -31.0, -3.0])
    montage_fname = op.join(tempdir, 'test_fid.hpts')
    montage = read_montage(montage_fname, unit='mm')
    assert_array_equal(montage.pos[0], pos * 1e-3)

    # test with last
    info = create_info(montage.ch_names, 1e3, ['eeg'] * len(montage.ch_names))
    _set_montage(info, montage)
    pos2 = np.array([c['loc'][:3] for c in info['chs']])
    pos3 = np.array([c['eeg_loc'][:, 0] for c in info['chs']])
    assert_array_equal(pos2, montage.pos)
    assert_array_equal(pos3, montage.pos)
    assert_equal(montage.ch_names, info['ch_names'])

    info = create_info(
        montage.ch_names, 1e3, ['eeg'] * len(montage.ch_names))

    evoked = EvokedArray(
        data=np.zeros((len(montage.ch_names), 1)), info=info, tmin=0)
    evoked.set_montage(montage)
    pos4 = np.array([c['loc'][:3] for c in evoked.info['chs']])
    pos5 = np.array([c['eeg_loc'][:, 0] for c in evoked.info['chs']])
    assert_array_equal(pos4, montage.pos)
    assert_array_equal(pos5, montage.pos)
    assert_equal(montage.ch_names, evoked.info['ch_names'])


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
