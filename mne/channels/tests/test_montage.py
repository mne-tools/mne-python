import os.path as op

from nose.tools import assert_equal, assert_true, assert_false

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne.channels import read_montage, apply_montage
from mne.utils import _TempDir
from mne import create_info


tempdir = _TempDir()


def test_montage():
    """Test making montages"""
    # no pep8
    input_str = ["""FidNz 0.00000 10.56381 -2.05108
    FidT9 -7.82694 0.45386 -3.76056
    FidT10 7.82694 0.45386 -3.76056""",
    """// MatLab   Sphere coordinates [degrees]         Cartesian coordinates
    // Label       Theta       Phi    Radius         X         Y         Z       off sphere surface
      E1      37.700     -14.000       1.000    0.7677    0.5934   -0.2419  -0.00000000000000011
      E2      44.600      -0.880       1.000    0.7119    0.7021   -0.0154   0.00000000000000000
      E3      51.700      11.000       1.000    0.6084    0.7704    0.1908   0.00000000000000000""",
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
    # 3 cardinal landmarks
    # --------------------
    # Nasion
    cardinal 2 -91 0 -42
    # Left auricular
    cardinal 1 0 -91 -42
    # Right auricular
    cardinal 3 0 91 -42
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
            table = np.loadtxt(fname, skiprows=2, dtype=dtype)
            pos2 = np.c_[table['x'], table['y'], table['z']]
            assert_array_almost_equal(pos2, montage.pos, 4)
        if kind.endswith('hpts'):
            assert_true(montage.fids)

    # test with last
    info = create_info(montage.ch_names, 1e3, ['eeg'] * len(montage.ch_names))
    assert_false(info['dig'])
    apply_montage(info, montage)
    pos2 = np.array([c['loc'][:3] for c in info['chs']])
    pos3 = np.array([c['eeg_loc'][:, 0] for c in info['chs']])
    assert_array_equal(pos2, montage.pos)
    assert_array_equal(pos3, montage.pos)
    assert_equal(montage.ch_names, info['ch_names'])
    assert_true(info['dig'])

