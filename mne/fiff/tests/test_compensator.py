# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_allclose

from mne import Epochs
from mne.fiff.compensator import make_compensator, get_current_comp
from mne.fiff import Raw, read_evoked, pick_types
from mne.utils import _TempDir, requires_mne, run_subprocess

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


@requires_mne
def test_compensation_mne():
    """Test comensation by comparing with MNE
    """
    def make_evoked(fname, comp):
        raw = Raw(fname, compensation=comp, proj=False)
        picks = pick_types(raw.info, meg=True, ref_meg=True)
        print len(picks)
        events = np.array([[0, 0, 1]], dtype=np.int)

        evoked = Epochs(raw, events, 1, 0, 20e-3, picks=picks).average()
        return evoked

    def compensate_mne(fname, comp):
        tmp_fname = '%s-%d.fif' % (fname[:-4], comp)
        cmd = ['mne_compensate_data', '--in', fname,
               '--out', tmp_fname, '--grad', str(comp)]
        run_subprocess(cmd)
        return read_evoked(tmp_fname)

    # save evoked response with default compensation
    tempdir = '/tmp/'
    fname_default = op.join(tempdir, 'ctf_default-ave.fif')
    make_evoked(ctf_comp_fname ,None).save(fname_default)

    for comp in [1]:
        evoked_py = make_evoked(ctf_comp_fname, comp)
        evoked_py.save('%s-py-%d.fif' % (fname_default[:-4], comp))
        evoked_c = compensate_mne(fname_default, comp)
        #assert_allclose(evoked_py.data, evoked_c.data, rtol=1e-6, atol=0)
        e = np.sum((evoked_c.data - evoked_py.data) ** 2) / np.sum(evoked_c.data ** 2)
        print '%d: %0.6e' % (comp, e)
    #return evoked_py - evoked_c
