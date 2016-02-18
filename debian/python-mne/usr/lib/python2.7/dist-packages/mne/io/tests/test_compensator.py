# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_allclose

from mne import Epochs, read_evokeds, pick_types
from mne.io.compensator import make_compensator, get_current_comp
from mne.io import Raw
from mne.utils import _TempDir, requires_mne, run_subprocess

base_dir = op.join(op.dirname(__file__), 'data')
ctf_comp_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')


def test_compensation():
    """Test compensation
    """
    tempdir = _TempDir()
    raw = Raw(ctf_comp_fname, compensation=None)
    comp1 = make_compensator(raw.info, 3, 1, exclude_comp_chs=False)
    assert_true(comp1.shape == (340, 340))
    comp2 = make_compensator(raw.info, 3, 1, exclude_comp_chs=True)
    assert_true(comp2.shape == (311, 340))

    # make sure that changing the comp doesn't modify the original data
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
    tempdir = _TempDir()

    def make_evoked(fname, comp):
        raw = Raw(fname, compensation=comp)
        picks = pick_types(raw.info, meg=True, ref_meg=True)
        events = np.array([[0, 0, 1]], dtype=np.int)
        evoked = Epochs(raw, events, 1, 0, 20e-3, picks=picks).average()
        return evoked

    def compensate_mne(fname, comp):
        tmp_fname = '%s-%d-ave.fif' % (fname[:-4], comp)
        cmd = ['mne_compensate_data', '--in', fname,
               '--out', tmp_fname, '--grad', str(comp)]
        run_subprocess(cmd)
        return read_evokeds(tmp_fname)[0]

    # save evoked response with default compensation
    fname_default = op.join(tempdir, 'ctf_default-ave.fif')
    make_evoked(ctf_comp_fname, None).save(fname_default)

    for comp in [0, 1, 2, 3]:
        evoked_py = make_evoked(ctf_comp_fname, comp)
        evoked_c = compensate_mne(fname_default, comp)
        picks_py = pick_types(evoked_py.info, meg=True, ref_meg=True)
        picks_c = pick_types(evoked_c.info, meg=True, ref_meg=True)
        assert_allclose(evoked_py.data[picks_py], evoked_c.data[picks_c],
                        rtol=1e-3, atol=1e-17)
