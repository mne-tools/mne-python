# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true

from mne.fiff.compensator import make_compensator
from mne.fiff import Raw

base_dir = op.join(op.dirname(__file__), 'data')
ctf_comp_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')


def test_compensation():
    raw = Raw(ctf_comp_fname, compensation=None)
    comp1 = make_compensator(raw.info, 3, 1, exclude_comp_chs=False)
    assert_true(comp1.shape == (340, 340))
    comp2 = make_compensator(raw.info, 3, 1, exclude_comp_chs=True)
    assert_true(comp2.shape == (311, 340))
