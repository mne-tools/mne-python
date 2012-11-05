import os.path as op
import warnings

from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_equal

from ...fiff import Raw
from ...fiff.proj import make_projector, activate_proj
from ..ssp import compute_proj_ecg, compute_proj_eog

data_path = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')


def test_compute_proj_ecg():
    """Test computation of ECG SSP projectors"""
    for average in [False, True]:
        raw = Raw(raw_fname, preload=True)
        with warnings.catch_warnings(record=True) as w:
            # this will throw a warning only for short filter_length (only
            # once, so it's for average == True), so set filter_length short
            projs, events = compute_proj_ecg(raw, n_mag=2, n_grad=2, n_eeg=2,
                                        ch_name='MEG 1531', bads=['MEG 2443'],
                                        average=average, avg_ref=True,
                                        no_proj=True, filter_length=2048)
            assert_equal(len(w), 0 if average else 1)
        raw.close()
        assert_true(len(projs) == 7)
        #XXX: better tests

        # without setting a bad channel, this should throw a warning (only
        # thrown once, so it's for average == True)
        with warnings.catch_warnings(record=True) as w:
            projs, events = compute_proj_ecg(raw, n_mag=2, n_grad=2, n_eeg=2,
                                            ch_name='MEG 1531', bads=[],
                                            average=average, avg_ref=True,
                                            no_proj=True)
            assert_equal(len(w), 0 if average else 1)
        assert_equal(projs, None)


def test_compute_proj_eog():
    """Test computation of EOG SSP projectors"""
    for average in [False, True]:
        raw = Raw(raw_fname, preload=True)
        n_projs_init = len(raw.info['projs'])
        projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                     bads=['MEG 2443'], average=average,
                                     avg_ref=True, no_proj=False)
        raw.close()
        assert_true(len(projs) == (7 + n_projs_init))
        #XXX: better tests

        # This will not throw a warning (?)
        with warnings.catch_warnings(record=True) as w:
            projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                         average=average, bads=[],
                                         avg_ref=True, no_proj=False)
            assert_equal(len(w), 0)
        assert_equal(projs, None)


def test_compute_proj_parallel():
    """Test computation of ExG projectors using parallelization"""
    raw = Raw(raw_fname, preload=True)
    projs, _ = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                bads=['MEG 2443'], average=False,
                                avg_ref=True, no_proj=False, n_jobs=1)
    raw.close()

    raw = Raw(raw_fname, preload=True)
    projs_2, _ = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                  bads=['MEG 2443'], average=False,
                                  avg_ref=True, no_proj=False, n_jobs=2)
    projs = activate_proj(projs)
    projs_2 = activate_proj(projs_2)
    projs, _, _ = make_projector(projs, raw.info['ch_names'],
                                 bads=['MEG 2443'])
    projs_2, _, _ = make_projector(projs_2, raw.info['ch_names'],
                                   bads=['MEG 2443'])
    raw.close()
    assert_array_equal(projs, projs_2)
