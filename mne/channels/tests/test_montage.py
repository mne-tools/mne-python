import os.path as op

from nose.tools import assert_equal

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne.channels import read_montage, apply_montage
from mne.utils import _TempDir
from mne import create_info


path = op.dirname(op.abspath(__file__))
hpts_fname = op.join(path, '../../io/edf/tests/data/biosemi.hpts')
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
    """]
    kinds = ['test.sfp', 'test.csd', 'test.elc', 'test.txt', 'test.elp']
    for kind, text in zip(kinds, input_str):
        fname = op.join(tempdir, kind)
        with open(fname, 'w') as fid:
            fid.write(text)
        montage = read_montage(fname)
        assert_equal(len(montage.ch_names), 3)
        assert_equal(len(montage.ch_names), len(montage.pos))
        assert_equal(montage.pos.shape, (3, 3))
        assert_equal(montage.kind, kind[:-4])
        if kind.endswith('csd'):
            dtype = [('label', 'S4'), ('theta', 'f8'), ('phi', 'f8'),
                     ('radius', 'f8'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                     ('off_sph', 'f8')]
            table = np.loadtxt(fname, skiprows=2, dtype=dtype)
            pos2 = np.c_[table['x'], table['y'], table['z']]
            assert_array_almost_equal(pos2, montage.pos, 4)

    # test with last
    info = create_info(montage.ch_names, 1e3, ['eeg'] * len(montage.ch_names))
    apply_montage(info, montage)
    pos2 = np.array([c['loc'][:3] for c in info['chs']])
    pos3 = np.array([c['eeg_loc'][:, 0] for c in info['chs']])
    assert_array_equal(pos2, montage.pos)
    assert_array_equal(pos3, montage.pos)
    assert_equal(montage.ch_names, info['ch_names'])

def test_fiducials():
    """"Test reading and applying fiducials"""
    ch_names = ['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1',
                'C1', 'C3','C5', 'T7','TP7','CP5','CP3','CP1','P1','P3','P5',
                'P7','P9', 'PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz',
                'Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6',
                'FC4','FC2','FCz','Cz', 'C2','C4','C6','T8','TP8','CP6','CP4',
                'CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
    ch_types = ['eeg']*len(ch_names)
    info = create_info(ch_names, 1000., ch_types)
    hpts_dir = op.dirname(hpts_fname)
    montage = read_montage(hpts_fname, path=hpts_dir)
    assert montage.fids is not None
    assert info['dig'] is None
    apply_montage(info, montage)

