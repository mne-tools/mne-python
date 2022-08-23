# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import os.path as op
import numpy as np
from numpy.testing import assert_allclose
import pytest

from mne import Epochs, read_evokeds, pick_types
from mne.io.compensator import make_compensator, get_current_comp
from mne.io import read_raw_fif
from mne.utils import requires_mne, run_subprocess

base_dir = op.join(op.dirname(__file__), 'data')
ctf_comp_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')


def test_compensation_identity():
    """Test compensation identity."""
    raw = read_raw_fif(ctf_comp_fname)
    assert get_current_comp(raw.info) == 3
    comp1 = make_compensator(raw.info, 3, 1, exclude_comp_chs=False)
    assert comp1.shape == (340, 340)
    comp2 = make_compensator(raw.info, 3, 1, exclude_comp_chs=True)
    assert comp2.shape == (311, 340)

    # round-trip
    desired = np.eye(340)
    for from_ in range(3):
        for to in range(3):
            if from_ == to:
                continue
            comp1 = make_compensator(raw.info, from_, to)
            comp2 = make_compensator(raw.info, to, from_)
            # To get 1e-12 here (instead of 1e-6) we must use the linalg.inv
            # method mentioned in compensator.py
            assert_allclose(np.dot(comp1, comp2), desired, atol=1e-12)
            assert_allclose(np.dot(comp2, comp1), desired, atol=1e-12)


@pytest.mark.parametrize('preload', (True, False))
@pytest.mark.parametrize('pick', (False, True))
def test_compensation_apply(tmp_path, preload, pick):
    """Test applying compensation."""
    # make sure that changing the comp doesn't modify the original data
    raw = read_raw_fif(ctf_comp_fname, preload=preload)
    assert raw._comp is None
    raw2 = raw.copy()
    raw2.apply_gradient_compensation(2)
    if pick:
        raw2.pick([0] + list(range(2, len(raw.ch_names))))
        raw.pick([0] + list(range(2, len(raw.ch_names))))
    assert get_current_comp(raw2.info) == 2
    if preload:
        assert raw2._comp is None
    else:
        assert raw2._comp.shape == (len(raw2.ch_names),) * 2
    fname = op.join(tmp_path, 'ctf-raw.fif')
    raw2.save(fname)
    raw2 = read_raw_fif(fname)
    assert raw2.compensation_grade == 2
    raw2.apply_gradient_compensation(3)
    assert raw2.compensation_grade == 3
    data, _ = raw[:, :]
    data2, _ = raw2[:, :]
    # channels have norm ~1e-12
    assert_allclose(data, data2, rtol=1e-9, atol=1e-18)
    for ch1, ch2 in zip(raw.info['chs'], raw2.info['chs']):
        assert ch1['coil_type'] == ch2['coil_type']


@requires_mne
def test_compensation_mne(tmp_path):
    """Test comensation by comparing with MNE."""
    def make_evoked(fname, comp):
        """Make evoked data."""
        raw = read_raw_fif(fname)
        if comp is not None:
            raw.apply_gradient_compensation(comp)
        picks = pick_types(raw.info, meg=True, ref_meg=True)
        events = np.array([[0, 0, 1]], dtype=np.int64)
        evoked = Epochs(raw, events, 1, 0, 20e-3, picks=picks,
                        baseline=None).average()
        return evoked

    def compensate_mne(fname, comp):
        """Compensate using MNE-C."""
        tmp_fname = '%s-%d-ave.fif' % (fname[:-4], comp)
        cmd = ['mne_compensate_data', '--in', fname,
               '--out', tmp_fname, '--grad', str(comp)]
        run_subprocess(cmd)
        return read_evokeds(tmp_fname)[0]

    # save evoked response with default compensation
    fname_default = op.join(tmp_path, 'ctf_default-ave.fif')
    make_evoked(ctf_comp_fname, None).save(fname_default)

    for comp in [0, 1, 2, 3]:
        evoked_py = make_evoked(ctf_comp_fname, comp)
        evoked_c = compensate_mne(fname_default, comp)
        picks_py = pick_types(evoked_py.info, meg=True, ref_meg=True)
        picks_c = pick_types(evoked_c.info, meg=True, ref_meg=True)
        assert_allclose(evoked_py.data[picks_py], evoked_c.data[picks_c],
                        rtol=1e-3, atol=1e-17)
        chs_py = [evoked_py.info['chs'][ii] for ii in picks_py]
        chs_c = [evoked_c.info['chs'][ii] for ii in picks_c]
        for ch_py, ch_c in zip(chs_py, chs_c):
            assert ch_py['coil_type'] == ch_c['coil_type']
