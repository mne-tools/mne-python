import os.path as op
import itertools as itt

from numpy.testing import assert_equal, assert_array_equal
import numpy as np

import pytest

from mne.datasets import testing
from mne import read_evokeds, read_cov
from mne.cov import prepare_noise_cov
from mne import compute_raw_covariance, pick_types, pick_info
from mne.io.proj import _has_eeg_average_ref_proj
from mne.io.pick import channel_type, _picks_by_type
from mne.io import read_raw_fif
from mne.proj import compute_proj_raw
from mne.rank import estimate_rank, _estimate_rank_meeg_cov
from mne.rank import _get_rank_sss, _get_sss_rank


base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
cov_fname = op.join(base_dir, 'test-cov.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
ave_fname = op.join(base_dir, 'test-ave.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')

testing_path = testing.data_path(download=False)
data_dir = op.join(testing_path, 'MEG', 'sample')
fif_fname = op.join(data_dir, 'sample_audvis_trunc_raw.fif')


def test_estimate_rank():
    """Test rank estimation."""
    data = np.eye(10)
    assert_array_equal(estimate_rank(data, return_singular=True)[1],
                       np.ones(10))
    data[0, 0] = 0
    assert_equal(estimate_rank(data), 9)
    pytest.raises(ValueError, estimate_rank, data, 'foo')


@pytest.mark.slowtest
@testing.requires_testing_data
def test_rank_estimation():
    """Test raw rank estimation."""
    iter_tests = itt.product(
        [fif_fname, hp_fif_fname],  # sss
        ['norm', dict(mag=1e11, grad=1e9, eeg=1e5)]
    )
    for fname, scalings in iter_tests:
        raw = read_raw_fif(fname).crop(0, 4.).load_data()
        (_, picks_meg), (_, picks_eeg) = _picks_by_type(raw.info,
                                                        meg_combined=True)
        n_meg = len(picks_meg)
        n_eeg = len(picks_eeg)

        if len(raw.info['proc_history']) == 0:
            expected_rank = n_meg + n_eeg
        else:
            expected_rank = _get_rank_sss(raw.info) + n_eeg
        assert_array_equal(raw.estimate_rank(scalings=scalings), expected_rank)
        assert_array_equal(raw.estimate_rank(picks=picks_eeg,
                                             scalings=scalings), n_eeg)
        if 'sss' in fname:
            raw.add_proj(compute_proj_raw(raw))
        raw.apply_proj()
        n_proj = len(raw.info['projs'])
        assert_array_equal(raw.estimate_rank(tstart=0, tstop=3.,
                                             scalings=scalings),
                           expected_rank - (0 if 'sss' in fname else n_proj))


@pytest.mark.slowtest
def test_rank():
    """Test cov rank estimation."""
    # Test that our rank estimation works properly on a simple case
    evoked = read_evokeds(ave_fname, condition=0, baseline=(None, 0),
                          proj=False)
    cov = read_cov(cov_fname)
    ch_names = [ch for ch in evoked.info['ch_names'] if '053' not in ch and
                ch.startswith('EEG')]
    cov = prepare_noise_cov(cov, evoked.info, ch_names, None)
    assert_equal(cov['eig'][0], 0.)  # avg projector should set this to zero
    assert (cov['eig'][1:] > 0).all()  # all else should be > 0

    # Now do some more comprehensive tests
    raw_sample = read_raw_fif(raw_fname)
    assert not _has_eeg_average_ref_proj(raw_sample.info['projs'])

    raw_sss = read_raw_fif(hp_fif_fname)
    assert not _has_eeg_average_ref_proj(raw_sss.info['projs'])
    raw_sss.add_proj(compute_proj_raw(raw_sss))

    cov_sample = compute_raw_covariance(raw_sample)
    cov_sample_proj = compute_raw_covariance(
        raw_sample.copy().apply_proj())

    cov_sss = compute_raw_covariance(raw_sss)
    cov_sss_proj = compute_raw_covariance(
        raw_sss.copy().apply_proj())

    picks_all_sample = pick_types(raw_sample.info, meg=True, eeg=True)
    picks_all_sss = pick_types(raw_sss.info, meg=True, eeg=True)

    info_sample = pick_info(raw_sample.info, picks_all_sample)
    picks_stack_sample = [('eeg', pick_types(info_sample, meg=False,
                                             eeg=True))]
    picks_stack_sample += [('meg', pick_types(info_sample, meg=True))]
    picks_stack_sample += [('all',
                            pick_types(info_sample, meg=True, eeg=True))]

    info_sss = pick_info(raw_sss.info, picks_all_sss)
    picks_stack_somato = [('eeg', pick_types(info_sss, meg=False, eeg=True))]
    picks_stack_somato += [('meg', pick_types(info_sss, meg=True))]
    picks_stack_somato += [('all',
                            pick_types(info_sss, meg=True, eeg=True))]

    iter_tests = list(itt.product(
        [(cov_sample, picks_stack_sample, info_sample),
         (cov_sample_proj, picks_stack_sample, info_sample),
         (cov_sss, picks_stack_somato, info_sss),
         (cov_sss_proj, picks_stack_somato, info_sss)],  # sss
        [dict(mag=1e15, grad=1e13, eeg=1e6)]
    ))

    for (cov, picks_list, this_info), scalings in iter_tests:
        for ch_type, picks in picks_list:

            this_very_info = pick_info(this_info, picks)

            # compute subset of projs
            this_projs = [c['active'] and
                          len(set(c['data']['col_names'])
                              .intersection(set(this_very_info['ch_names']))) >
                          0 for c in cov['projs']]
            n_projs = sum(this_projs)

            # count channel types
            ch_types = [channel_type(this_very_info, idx)
                        for idx in range(len(picks))]
            n_eeg, n_mag, n_grad = [ch_types.count(k) for k in
                                    ['eeg', 'mag', 'grad']]
            n_meg = n_mag + n_grad

            # check sss
            if len(this_very_info['proc_history']) > 0:
                mf = this_very_info['proc_history'][0]['max_info']
                n_free = _get_sss_rank(mf)
                if 'mag' not in ch_types and 'grad' not in ch_types:
                    n_free = 0
                # - n_projs XXX clarify
                expected_rank = n_free + n_eeg
            else:
                expected_rank = n_meg + n_eeg - n_projs

            C = cov['data'][np.ix_(picks, picks)]
            est_rank = _estimate_rank_meeg_cov(C, this_very_info,
                                               scalings=scalings)

            assert_equal(expected_rank, est_rank)


def test_maxfilter_get_rank():
    """Test maxfilter rank lookup."""
    raw = read_raw_fif(raw_fname)
    mf = raw.info['proc_history'][0]['max_info']
    rank1 = mf['sss_info']['nfree']
    rank2 = _get_rank_sss(raw)
    assert rank1 == rank2
