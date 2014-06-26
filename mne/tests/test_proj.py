import os.path as op
from nose.tools import assert_true
import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_equal)

import copy as cp

import mne
from mne.datasets import sample
from mne import pick_types
from mne.io import Raw
from mne import compute_proj_epochs, compute_proj_evoked, compute_proj_raw
from mne.io.proj import make_projector, activate_proj
from mne.proj import read_proj, write_proj, make_eeg_average_ref_proj
from mne import read_events, Epochs, sensitivity_map, read_source_estimate
from mne.utils import _TempDir

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')
proj_fname = op.join(base_dir, 'test-proj.fif')
proj_gz_fname = op.join(base_dir, 'test-proj.fif.gz')
bads_fname = op.join(base_dir, 'test_bads.txt')

data_path = sample.data_path(download=False)
sample_path = op.join(data_path, 'MEG', 'sample')
fwd_fname = op.join(sample_path, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
sensmap_fname = op.join(sample_path, 'sample_audvis-%s-oct-6-fwd-sensmap-%s.w')

# sample dataset should be updated to reflect mne conventions
eog_fname = op.join(sample_path, 'sample_audvis_eog_proj.fif')

tempdir = _TempDir()


@sample.requires_sample_data
def test_sensitivity_maps():
    """Test sensitivity map computation"""
    fwd = mne.read_forward_solution(fwd_fname, surf_ori=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        proj_eog = read_proj(eog_fname)
    decim = 6
    for ch_type in ['eeg', 'grad', 'mag']:
        w = read_source_estimate(sensmap_fname % (ch_type, 'lh')).data
        stc = sensitivity_map(fwd, projs=None, ch_type=ch_type,
                              mode='free', exclude='bads')
        assert_array_almost_equal(stc.data, w, decim)
        assert_true(stc.subject == 'sample')
        # let's just make sure the others run
        if ch_type == 'grad':
            # fixed (2)
            w = read_source_estimate(sensmap_fname % (ch_type, '2-lh')).data
            stc = sensitivity_map(fwd, projs=None, mode='fixed',
                                  ch_type=ch_type, exclude='bads')
            assert_array_almost_equal(stc.data, w, decim)
        if ch_type == 'mag':
            # ratio (3)
            w = read_source_estimate(sensmap_fname % (ch_type, '3-lh')).data
            stc = sensitivity_map(fwd, projs=None, mode='ratio',
                                  ch_type=ch_type, exclude='bads')
            assert_array_almost_equal(stc.data, w, decim)
        if ch_type == 'eeg':
            # radiality (4), angle (5), remaining (6), and  dampening (7)
            modes = ['radiality', 'angle', 'remaining', 'dampening']
            ends = ['4-lh', '5-lh', '6-lh', '7-lh']
            for mode, end in zip(modes, ends):
                w = read_source_estimate(sensmap_fname % (ch_type, end)).data
                stc = sensitivity_map(fwd, projs=proj_eog, mode=mode,
                                      ch_type=ch_type, exclude='bads')
                assert_array_almost_equal(stc.data, w, decim)

    # test corner case for EEG
    stc = sensitivity_map(fwd, projs=[make_eeg_average_ref_proj(fwd['info'])],
                          ch_type='eeg', exclude='bads')


def test_compute_proj_epochs():
    """Test SSP computation on epochs"""
    event_id, tmin, tmax = 1, -0.2, 0.3

    raw = Raw(raw_fname, preload=True)
    events = read_events(event_fname)
    bad_ch = 'MEG 2443'
    picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude=[])
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, proj=False)

    evoked = epochs.average()
    projs = compute_proj_epochs(epochs, n_grad=1, n_mag=1, n_eeg=0, n_jobs=1)
    write_proj(op.join(tempdir, 'test-proj.fif.gz'), projs)
    for p_fname in [proj_fname, proj_gz_fname,
                    op.join(tempdir, 'test-proj.fif.gz')]:
        projs2 = read_proj(p_fname)

        assert_true(len(projs) == len(projs2))

        for p1, p2 in zip(projs, projs2):
            assert_true(p1['desc'] == p2['desc'])
            assert_true(p1['data']['col_names'] == p2['data']['col_names'])
            assert_true(p1['active'] == p2['active'])
            # compare with sign invariance
            p1_data = p1['data']['data'] * np.sign(p1['data']['data'][0, 0])
            p2_data = p2['data']['data'] * np.sign(p2['data']['data'][0, 0])
            if bad_ch in p1['data']['col_names']:
                bad = p1['data']['col_names'].index('MEG 2443')
                mask = np.ones(p1_data.size, dtype=np.bool)
                mask[bad] = False
                p1_data = p1_data[:, mask]
                p2_data = p2_data[:, mask]
            corr = np.corrcoef(p1_data, p2_data)[0, 1]
            assert_array_almost_equal(corr, 1.0, 5)

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, epochs.ch_names, bads=[])

    assert_true(nproj == 2)
    assert_true(U.shape[1] == 2)

    # test that you can save them
    epochs.info['projs'] += projs
    evoked = epochs.average()
    evoked.save(op.join(tempdir, 'foo-ave.fif'))

    projs = read_proj(proj_fname)

    projs_evoked = compute_proj_evoked(evoked, n_grad=1, n_mag=1, n_eeg=0)
    assert_true(len(projs_evoked) == 2)
    # XXX : test something

    # test parallelization
    projs = compute_proj_epochs(epochs, n_grad=1, n_mag=1, n_eeg=0, n_jobs=2)
    projs = activate_proj(projs)
    proj_par, _, _ = make_projector(projs, epochs.ch_names, bads=[])
    assert_allclose(proj, proj_par, rtol=1e-8, atol=1e-16)

    # test warnings on bad filenames
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        proj_badname = op.join(tempdir, 'test-bad-name.fif.gz')
        write_proj(proj_badname, projs)
        read_proj(proj_badname)
        print([ww.message for ww in w])
    assert_equal(len(w), 2)


def test_compute_proj_raw():
    """Test SSP computation on raw"""
    # Test that the raw projectors work
    raw_time = 2.5  # Do shorter amount for speed
    raw = Raw(raw_fname, preload=True).crop(0, raw_time, False)
    for ii in (0.25, 0.5, 1, 2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            projs = compute_proj_raw(raw, duration=ii - 0.1, stop=raw_time,
                                     n_grad=1, n_mag=1, n_eeg=0)
            assert_true(len(w) == 1)

        # test that you can compute the projection matrix
        projs = activate_proj(projs)
        proj, nproj, U = make_projector(projs, raw.ch_names, bads=[])

        assert_true(nproj == 2)
        assert_true(U.shape[1] == 2)

        # test that you can save them
        raw.info['projs'] += projs
        raw.save(op.join(tempdir, 'foo_%d_raw.fif' % ii), overwrite=True)

    # Test that purely continuous (no duration) raw projection works
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        projs = compute_proj_raw(raw, duration=None, stop=raw_time,
                                 n_grad=1, n_mag=1, n_eeg=0)
        assert_equal(len(w), 1)

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, raw.ch_names, bads=[])

    assert_true(nproj == 2)
    assert_true(U.shape[1] == 2)

    # test that you can save them
    raw.info['projs'] += projs
    raw.save(op.join(tempdir, 'foo_rawproj_continuous_raw.fif'))

    # test resampled-data projector, upsampling instead of downsampling
    # here to save an extra filtering (raw would have to be LP'ed to be equiv)
    raw_resamp = cp.deepcopy(raw)
    raw_resamp.resample(raw.info['sfreq'] * 2, n_jobs=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        projs = compute_proj_raw(raw_resamp, duration=None, stop=raw_time,
                                 n_grad=1, n_mag=1, n_eeg=0)
    projs = activate_proj(projs)
    proj_new, _, _ = make_projector(projs, raw.ch_names, bads=[])
    assert_array_almost_equal(proj_new, proj, 4)

    # test with bads
    raw.load_bad_channels(bads_fname)  # adds 2 bad mag channels
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        projs = compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=1)

    # test that bad channels can be excluded
    proj, nproj, U = make_projector(projs, raw.ch_names,
                                    bads=raw.ch_names)
    assert_array_almost_equal(proj, np.eye(len(raw.ch_names)))
