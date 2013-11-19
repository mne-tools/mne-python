# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true
from numpy.testing import assert_allclose

from mne.fiff.compensator import make_compensator, get_current_comp
from mne.fiff import Raw
from mne.utils import _TempDir

base_dir = op.join(op.dirname(__file__), 'data')
ctf_comp_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')

tempdir = _TempDir()

def test_compensation():
    """Test compensation
    """
    raw = Raw(ctf_comp_fname, compensation=None)
    comp1 = make_compensator(raw.info, 3, 1, exclude_comp_chs=False)
    assert_true(comp1.shape == (340, 340))
    comp2 = make_compensator(raw.info, 3, 1, exclude_comp_chs=True)
    assert_true(comp2.shape == (311, 340))

    # make sure that chaning the comp doesn't modify the original data
    raw2 = Raw(ctf_comp_fname, compensation=2)
    assert_true(get_current_comp(raw2.info) == 2)
    fname = op.join(tempdir, 'ctf-raw.fif')
    raw2.save(fname)
    raw2 = Raw(fname, compensation=None)
    data, _ = raw[:, :]
    data2, _ = raw2[:, :]
    assert_allclose(data, data2, rtol=1e-9, atol=1e-20)
    for ch1, ch2 in zip(raw.info['chs'], raw2.info['chs']):
        assert_true(ch1['coil_type'] == ch2['coil_type'])
