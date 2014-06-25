import os.path as op

from nose.tools import assert_equal

from mne.montages import read_montage
from mne.utils import _TempDir


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
    """]
    kinds = ['test.sfp', 'test.csd', 'test.elc', 'test.txt']
    for kind, text in zip(kinds, input_str):
        fname = op.join(tempdir, kind)
        with open(fname, 'w') as fid:
            fid.write(text)
        montage = read_montage(fname)
        assert_equal(len(montage.names), 3)
        assert_equal(len(montage.names), len(montage.pos))
        assert_equal(montage.pos.shape, (3, 3))
        assert_equal(montage.kind, kind)
