import os.path as op

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from ...fiff import Raw, proj
from ..ssp import compute_proj_ecg, compute_proj_eog

data_path = op.join(op.dirname(__file__), '..', '..', 'fiff', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')


def test_compute_proj_ecg():
    """Test computation of ECG SSP projectors"""
    for average in [False, True]:
        raw = Raw(raw_fname, preload=True)
        projs, events = compute_proj_ecg(raw, n_mag=2, n_grad=2, n_eeg=2,
                                         ch_name='MEG 1531', bads=['MEG 2443'],
                                         average=average, avg_ref=True,
                                         no_proj=True)
        raw.close()
        assert_true(len(projs) == 7)
        #XXX: better tests


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
        p, _, _ = proj.make_projector(projs, raw.info['ch_names'],
                                 raw.info['bads'])

        raw = [Raw(raw_fname, preload=True), Raw(raw_fname, preload=True)]
        projs_2, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                     bads=['MEG 2443'], average=average,
                                     avg_ref=True, no_proj=False)
        raw[0].close()
        raw[1].close()
        assert_true(len(projs_2) == (7 + n_projs_init))
        p_2, _, _ = proj.make_projector(projs_2, raw[0].info['ch_names'],
                                 raw[0].info['bads'])
        assert_array_almost_equal(p, p_2)


def test_compute_proj_eog_multiple():
    """Test computation of EOG SSP projectors with multiple Raw"""
    average = False
    raw = Raw(raw_fname, preload=True)
    projs, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                 bads=['MEG 2443'], average=average,
                                 avg_ref=True, no_proj=False)
    raw.close()
    p, _, _ = proj.make_projector(projs, raw.info['ch_names'],
                             raw.info['bads'])

    raw = [Raw(raw_fname, preload=True), Raw(raw_fname, preload=True)]
    projs_2, events = compute_proj_eog(raw, n_mag=2, n_grad=2, n_eeg=2,
                                 bads=['MEG 2443'], average=average,
                                 avg_ref=True, no_proj=False)
    raw[0].close()
    raw[1].close()
    p_2, _, _ = proj.make_projector(projs_2, raw[0].info['ch_names'],
                             raw[0].info['bads'])
    assert_array_almost_equal(p, p_2)
