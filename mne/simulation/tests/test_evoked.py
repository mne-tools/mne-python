# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
import pytest

from mne import (read_cov, read_forward_solution, convert_forward_solution,
                 pick_types_forward, read_evokeds, pick_types, EpochsArray,
                 compute_covariance, compute_raw_covariance)
from mne.datasets import testing
from mne.simulation import simulate_sparse_stc, simulate_evoked, add_noise
from mne.io import read_raw_fif
from mne.io.pick import pick_channels_cov
from mne.cov import regularize, whiten_evoked
from mne.utils import catch_logging, check_version

data_path = testing.data_path(download=False)
fwd_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
raw_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test_raw.fif')
ave_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-ave.fif')
cov_fname = op.join(op.dirname(__file__), '..', '..', 'io', 'tests',
                    'data', 'test-cov.fif')


@testing.requires_testing_data
def test_simulate_evoked():
    """Test simulation of evoked data."""
    raw = read_raw_fif(raw_fname)
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=False)
    fwd = pick_types_forward(fwd, meg=True, eeg=True, exclude=raw.info['bads'])
    cov = read_cov(cov_fname)

    evoked_template = read_evokeds(ave_fname, condition=0, baseline=None)
    evoked_template.pick_types(meg=True, eeg=True, exclude=raw.info['bads'])

    cov = regularize(cov, evoked_template.info)
    nave = evoked_template.nave

    tmin = -0.1
    sfreq = 1000.  # Hz
    tstep = 1. / sfreq
    n_samples = 600
    times = np.linspace(tmin, tmin + n_samples * tstep, n_samples)

    # Generate times series for 2 dipoles
    stc = simulate_sparse_stc(fwd['src'], n_dipoles=2, times=times,
                              random_state=42)

    # Generate noisy evoked data
    iir_filter = [1, -0.9]
    evoked = simulate_evoked(fwd, stc, evoked_template.info, cov,
                             iir_filter=iir_filter, nave=nave,
                             random_state=0)
    assert_array_almost_equal(evoked.times, stc.times)
    assert len(evoked.data) == len(fwd['sol']['data'])
    assert_equal(evoked.nave, nave)
    assert len(evoked.info['projs']) == len(cov['projs'])
    evoked_white = whiten_evoked(evoked, cov)
    assert abs(evoked_white.data[:, 0].std() - 1.) < 0.1

    # make a vertex that doesn't exist in fwd, should throw error
    stc_bad = stc.copy()
    mv = np.max(fwd['src'][0]['vertno'][fwd['src'][0]['inuse']])
    stc_bad.vertices[0][0] = mv + 1

    pytest.raises(ValueError, simulate_evoked, fwd, stc_bad,
                  evoked_template.info, cov)
    evoked_1 = simulate_evoked(fwd, stc, evoked_template.info, cov,
                               nave=np.inf)
    evoked_2 = simulate_evoked(fwd, stc, evoked_template.info, cov,
                               nave=np.inf)
    assert_array_equal(evoked_1.data, evoked_2.data)

    cov['names'] = cov.ch_names[:-2]  # Error channels are different.
    with pytest.raises(RuntimeError, match='Not all channels present'):
        simulate_evoked(fwd, stc, evoked_template.info, cov)


# We don't use an avg ref here, but let's ignore it. Also we know we have
# few samples, and that our epochs are not baseline corrected.
@pytest.mark.filterwarnings('ignore:No average EEG reference present')
@pytest.mark.filterwarnings('ignore:Too few samples')
@pytest.mark.filterwarnings('ignore:Epochs are not baseline corrected')
def test_add_noise():
    """Test noise addition."""
    if check_version('numpy', '1.17'):
        rng = np.random.default_rng(0)
    else:
        rng = np.random.RandomState(0)
    raw = read_raw_fif(raw_fname)
    raw.del_proj()
    picks = pick_types(raw.info, meg=True, eeg=True, exclude=())
    cov = compute_raw_covariance(raw, picks=picks)
    with pytest.raises(RuntimeError, match='to be loaded'):
        add_noise(raw, cov)
    raw.crop(0, 1).load_data()
    with pytest.raises(TypeError, match='Raw, Epochs, or Evoked'):
        add_noise(0., cov)
    with pytest.raises(TypeError, match='Covariance'):
        add_noise(raw, 0.)
    # test a no-op (data preserved)
    orig_data = raw[:][0]
    zero_cov = cov.copy()
    zero_cov['data'].fill(0)
    add_noise(raw, zero_cov)
    new_data = raw[:][0]
    assert_allclose(orig_data, new_data, atol=1e-30)
    # set to zero to make comparisons easier
    raw._data[:] = 0.
    epochs = EpochsArray(np.zeros((1, len(raw.ch_names), 100)),
                         raw.info.copy())
    epochs.info['bads'] = []
    evoked = epochs.average(picks=np.arange(len(raw.ch_names)))
    for inst in (raw, epochs, evoked):
        with catch_logging() as log:
            add_noise(inst, cov, random_state=rng, verbose=True)
        log = log.getvalue()
        want = ('to {0}/{1} channels ({0}'
                .format(len(cov['names']), len(raw.ch_names)))
        assert want in log
        if inst is evoked:
            inst = EpochsArray(inst.data[np.newaxis], inst.info)
        if inst is raw:
            cov_new = compute_raw_covariance(inst, picks=picks)
        else:
            cov_new = compute_covariance(inst)
        assert cov['names'] == cov_new['names']
        r = np.corrcoef(cov['data'].ravel(), cov_new['data'].ravel())[0, 1]
        assert r > 0.99


def test_rank_deficiency():
    """Test adding noise from M/EEG float32 (I/O) cov with projectors."""
    # See gh-5940
    evoked = read_evokeds(ave_fname, 0, baseline=(None, 0))
    evoked.info['bads'] = ['MEG 2443']
    with evoked.info._unlock():
        evoked.info['lowpass'] = 20  # fake for decim
    picks = pick_types(evoked.info, meg=True, eeg=False)
    picks = picks[::16]
    evoked.pick_channels([evoked.ch_names[pick] for pick in picks])
    evoked.info.normalize_proj()
    cov = read_cov(cov_fname)
    cov['projs'] = []
    cov = regularize(cov, evoked.info, rank=None)
    cov = pick_channels_cov(cov, evoked.ch_names)
    evoked.data[:] = 0
    add_noise(evoked, cov, random_state=0)
    cov_new = compute_covariance(
        EpochsArray(evoked.data[np.newaxis], evoked.info), verbose='error')
    assert cov['names'] == cov_new['names']
    r = np.corrcoef(cov['data'].ravel(), cov_new['data'].ravel())[0, 1]
    assert r > 0.98


@testing.requires_testing_data
def test_order():
    """Test that order does not matter."""
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=False)
    evoked = read_evokeds(ave_fname)[0].pick_types(meg=True, eeg=True)
    assert 'meg' in evoked
    assert 'eeg' in evoked
    meg_picks = pick_types(evoked.info, meg=True)
    eeg_picks = pick_types(evoked.info, eeg=True)
    # MEG then EEG
    assert (eeg_picks > meg_picks.max()).all()
    times = np.arange(10) / 1000.
    stc = simulate_sparse_stc(fwd['src'], 1, times=times, random_state=0)
    evoked_sim = simulate_evoked(fwd, stc, evoked.info, nave=np.inf)
    reorder = np.concatenate([eeg_picks, meg_picks])
    evoked.reorder_channels([evoked.ch_names[pick] for pick in reorder])
    evoked_sim_2 = simulate_evoked(fwd, stc, evoked.info, nave=np.inf)
    want_data = evoked_sim.data[reorder]
    assert_allclose(evoked_sim_2.data, want_data)
