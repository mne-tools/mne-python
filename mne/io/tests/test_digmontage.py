import os.path as op

from nose.tools import assert_equal

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mne.channels import apply_montage
from mne.io.meas_info import read_digmontage
from mne.utils import _TempDir
from mne import create_info
from mne.transforms import (apply_trans, get_ras_to_neuromag_trans)


FILE = inspect.getfile(inspect.currentframe())
p_dir = op.dirname(op.abspath(FILE))
elp = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_elp.txt')
hsp = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_hsp.txt')
mrk = op.join(p_dir, '..', 'kit', 'tests', 'data', 'test_mrk.sqd')
elp_names = []

def test_read_digmontage():
    """Test read_digmontage"""
    names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
    montage = read_digmontage(hsp, hpi, elp, elp)
    assert_equal(m.ch_names, names)
    # check coordinate transformation
    assert_almost_equal(montage.pos[0, 0], 0)
    assert_almost_equal(montage.pos[0, 2], 0)
    assert_allclose(montage.pos[1:3, 1:], 0, atol=1e-16)

