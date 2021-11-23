import copy as cp
import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_equal)
import pytest
from scipy import linalg

from mne import (compute_proj_epochs, compute_proj_evoked, compute_proj_raw,
                 pick_types, read_events, Epochs, sensitivity_map,
                 read_source_estimate, compute_raw_covariance, create_info,
                 read_forward_solution, convert_forward_solution)
from mne.cov import regularize, compute_whitener
from mne.datasets import testing
from mne.io import read_raw_fif, RawArray
from mne.io.proj import (make_projector, activate_proj,
                         _needs_eeg_average_ref_proj)
from mne.preprocessing import maxwell_filter
from mne.proj import (read_proj, write_proj, make_eeg_average_ref_proj,
                      _has_eeg_average_ref_proj)
from mne.rank import _compute_rank_int

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_fname = op.join(base_dir, 'test-eve.fif')
proj_fname = op.join(base_dir, 'test-proj.fif')
proj_gz_fname = op.join(base_dir, 'test-proj.fif.gz')
bads_fname = op.join(base_dir, 'test_bads.txt')

sample_path = op.join(testing.data_path(download=False), 'MEG', 'sample')
fwd_fname = op.join(sample_path, 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
sensmap_fname = op.join(sample_path,
                        'sample_audvis_trunc-%s-oct-4-fwd-sensmap-%s.w')

eog_fname = op.join(sample_path, 'sample_audvis_eog-proj.fif')
ecg_fname = op.join(sample_path, 'sample_audvis_ecg-proj.fif')


def test_bad_proj():
    """Test dealing with bad projection application."""
    raw = read_raw_fif(raw_fname, preload=True)
    events = read_events(event_fname)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[2:18:3]
    _check_warnings(raw, events, picks)
    # still bad
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    _check_warnings(raw, events)
    # "fixed"
    raw.info.normalize_proj()  # avoid projection warnings
    _check_warnings(raw, events, count=0)
    # eeg avg ref is okay
    raw = read_raw_fif(raw_fname, preload=True).pick_types(meg=False, eeg=True)
    raw.set_eeg_reference(projection=True)
    _check_warnings(raw, events, count=0)
    raw.info['bads'] = raw.ch_names[:10]
    _check_warnings(raw, events, count=0)

    raw = read_raw_fif(raw_fname)
    pytest.raises(ValueError, raw.del_proj, 'foo')
    n_proj = len(raw.info['projs'])
    raw.del_proj(0)
    assert_equal(len(raw.info['projs']), n_proj - 1)
    raw.del_proj()
    assert_equal(len(raw.info['projs']), 0)

    # Ensure we deal with newer-style Neuromag projs properly, were getting:
    #
    #     Projection vector "PCA-v2" has magnitude 1.00 (should be unity),
    #     applying projector with 101/306 of the original channels available
    #     may be dangerous.
    raw = read_raw_fif(raw_fname).crop(0, 1)
    raw.set_eeg_reference(projection=True)
    raw.info['bads'] = ['MEG 0111']
    meg_picks = pick_types(raw.info, meg=True, exclude=())
    ch_names = [raw.ch_names[pick] for pick in meg_picks]
    for p in raw.info['projs'][:-1]:
        data = np.zeros((1, len(ch_names)))
        idx = [ch_names.index(ch_name) for ch_name in p['data']['col_names']]
        data[:, idx] = p['data']['data']
        p['data'].update(ncol=len(meg_picks), col_names=ch_names, data=data)
    # smoke test for no warnings during reg
    regularize(compute_raw_covariance(raw, verbose='error'), raw.info)


def _check_warnings(raw, events, picks=None, count=3):
    """Count warnings."""
    with pytest.warns(None) as w:
        Epochs(raw, events, dict(aud_l=1, vis_l=3),
               -0.2, 0.5, picks=picks, preload=True, proj=True)
    assert len(w) == count
    assert all('dangerous' in str(ww.message) for ww in w)


@testing.requires_testing_data
def test_sensitivity_maps():
    """Test sensitivity map computation."""
    fwd = read_forward_solution(fwd_fname)
    fwd = convert_forward_solution(fwd, surf_ori=True)
    projs = read_proj(eog_fname)
    projs.extend(read_proj(ecg_fname))
    decim = 6
    for ch_type in ['eeg', 'grad', 'mag']:
        w = read_source_estimate(sensmap_fname % (ch_type, 'lh')).data
        stc = sensitivity_map(fwd, projs=None, ch_type=ch_type,
                              mode='free', exclude='bads')
        assert_array_almost_equal(stc.data, w, decim)
        assert stc.subject == 'sample'
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
                stc = sensitivity_map(fwd, projs=projs, mode=mode,
                                      ch_type=ch_type, exclude='bads')
                assert_array_almost_equal(stc.data, w, decim)

    # test corner case for EEG
    stc = sensitivity_map(fwd, projs=[make_eeg_average_ref_proj(fwd['info'])],
                          ch_type='eeg', exclude='bads')
    # test corner case for projs being passed but no valid ones (#3135)
    pytest.raises(ValueError, sensitivity_map, fwd, projs=None, mode='angle')
    pytest.raises(RuntimeError, sensitivity_map, fwd, projs=[], mode='angle')
    # test volume source space
    fname = op.join(sample_path, 'sample_audvis_trunc-meg-vol-7-fwd.fif')
    fwd = read_forward_solution(fname)
    sensitivity_map(fwd)


def test_compute_proj_epochs(tmp_path):
    """Test SSP computation on epochs."""
    tempdir = str(tmp_path)
    event_id, tmin, tmax = 1, -0.2, 0.3

    raw = read_raw_fif(raw_fname, preload=True)
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

        assert len(projs) == len(projs2)

        for p1, p2 in zip(projs, projs2):
            assert p1['desc'] == p2['desc']
            assert p1['data']['col_names'] == p2['data']['col_names']
            assert p1['active'] == p2['active']
            # compare with sign invariance
            p1_data = p1['data']['data'] * np.sign(p1['data']['data'][0, 0])
            p2_data = p2['data']['data'] * np.sign(p2['data']['data'][0, 0])
            if bad_ch in p1['data']['col_names']:
                bad = p1['data']['col_names'].index('MEG 2443')
                mask = np.ones(p1_data.size, dtype=bool)
                mask[bad] = False
                p1_data = p1_data[:, mask]
                p2_data = p2_data[:, mask]
            corr = np.corrcoef(p1_data, p2_data)[0, 1]
            assert_array_almost_equal(corr, 1.0, 5)
            if p2['explained_var']:
                assert_array_almost_equal(p1['explained_var'],
                                          p2['explained_var'])

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, epochs.ch_names, bads=[])

    assert nproj == 2
    assert U.shape[1] == 2

    # test that you can save them
    with epochs.info._unlock():
        epochs.info['projs'] += projs
    evoked = epochs.average()
    evoked.save(op.join(tempdir, 'foo-ave.fif'))

    projs = read_proj(proj_fname)

    projs_evoked = compute_proj_evoked(evoked, n_grad=1, n_mag=1, n_eeg=0)
    assert len(projs_evoked) == 2
    # XXX : test something

    # test parallelization
    projs = compute_proj_epochs(epochs, n_grad=1, n_mag=1, n_eeg=0, n_jobs=1,
                                desc_prefix='foobar')
    assert all('foobar' in x['desc'] for x in projs)
    projs = activate_proj(projs)
    proj_par, _, _ = make_projector(projs, epochs.ch_names, bads=[])
    assert_allclose(proj, proj_par, rtol=1e-8, atol=1e-16)

    # test warnings on bad filenames
    proj_badname = op.join(tempdir, 'test-bad-name.fif.gz')
    with pytest.warns(RuntimeWarning, match='-proj.fif'):
        write_proj(proj_badname, projs)
    with pytest.warns(RuntimeWarning, match='-proj.fif'):
        read_proj(proj_badname)

    # bad inputs
    fname = op.join(tempdir, 'out-proj.fif')
    with pytest.raises(TypeError, match='projs'):
        write_proj(fname, 'foo')
    with pytest.raises(TypeError, match=r'projs\[0\] must be .*'):
        write_proj(fname, ['foo'])


@pytest.mark.slowtest
def test_compute_proj_raw(tmp_path):
    """Test SSP computation on raw."""
    tempdir = str(tmp_path)
    # Test that the raw projectors work
    raw_time = 2.5  # Do shorter amount for speed
    raw = read_raw_fif(raw_fname).crop(0, raw_time)
    raw.load_data()
    for ii in (0.25, 0.5, 1, 2):
        with pytest.warns(RuntimeWarning, match='Too few samples'):
            projs = compute_proj_raw(raw, duration=ii - 0.1, stop=raw_time,
                                     n_grad=1, n_mag=1, n_eeg=0)

        # test that you can compute the projection matrix
        projs = activate_proj(projs)
        proj, nproj, U = make_projector(projs, raw.ch_names, bads=[])

        assert nproj == 2
        assert U.shape[1] == 2

        # test that you can save them
        with raw.info._unlock():
            raw.info['projs'] += projs
        raw.save(op.join(tempdir, 'foo_%d_raw.fif' % ii), overwrite=True)

    # Test that purely continuous (no duration) raw projection works
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        projs = compute_proj_raw(raw, duration=None, stop=raw_time,
                                 n_grad=1, n_mag=1, n_eeg=0)

    # test that you can compute the projection matrix
    projs = activate_proj(projs)
    proj, nproj, U = make_projector(projs, raw.ch_names, bads=[])

    assert nproj == 2
    assert U.shape[1] == 2

    # test that you can save them
    with raw.info._unlock():
        raw.info['projs'] += projs
    raw.save(op.join(tempdir, 'foo_rawproj_continuous_raw.fif'))

    # test resampled-data projector, upsampling instead of downsampling
    # here to save an extra filtering (raw would have to be LP'ed to be equiv)
    raw_resamp = cp.deepcopy(raw)
    raw_resamp.resample(raw.info['sfreq'] * 2, n_jobs=2, npad='auto')
    projs = compute_proj_raw(raw_resamp, duration=None, stop=raw_time,
                             n_grad=1, n_mag=1, n_eeg=0)
    projs = activate_proj(projs)
    proj_new, _, _ = make_projector(projs, raw.ch_names, bads=[])
    assert_array_almost_equal(proj_new, proj, 4)

    # test with bads
    raw.load_bad_channels(bads_fname)  # adds 2 bad mag channels
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        projs = compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=1)
    assert len(projs) == 1

    # test that bad channels can be excluded, and empty support
    for projs_ in (projs, []):
        proj, nproj, U = make_projector(projs_, raw.ch_names,
                                        bads=raw.ch_names)
        assert_array_almost_equal(proj, np.eye(len(raw.ch_names)))
        assert nproj == 0  # all channels excluded
        assert U.shape == (len(raw.ch_names), nproj)


@pytest.mark.parametrize('duration', [1, np.pi / 2.])
@pytest.mark.parametrize('sfreq', [600.614990234375, 1000.])
def test_proj_raw_duration(duration, sfreq):
    """Test equivalence of `duration` options."""
    n_ch, n_dim = 30, 3
    rng = np.random.RandomState(0)
    signals = rng.randn(n_dim, 10000)
    mixing = rng.randn(n_ch, n_dim) + [0, 1, 2]
    data = np.dot(mixing, signals)
    raw = RawArray(data, create_info(n_ch, sfreq, 'eeg'))
    raw.set_eeg_reference(projection=True)
    n_eff = int(round(raw.info['sfreq'] * duration))
    # crop to an even "duration" number of epochs
    stop = ((len(raw.times) // n_eff) * n_eff - 1) / raw.info['sfreq']
    raw.crop(0, stop)
    proj_def = compute_proj_raw(raw, n_eeg=n_dim)
    proj_dur = compute_proj_raw(raw, duration=duration, n_eeg=n_dim)
    proj_none = compute_proj_raw(raw, duration=None, n_eeg=n_dim)
    assert len(proj_dur) == len(proj_none) == len(proj_def) == n_dim
    # proj_def is not in here because it does not necessarily evenly divide
    # the signal length:
    for pu, pn in zip(proj_dur, proj_none):
        assert_allclose(pu['data']['data'], pn['data']['data'])
    # but we can test it here since it should still be a small subspace angle:
    for proj in (proj_dur, proj_none, proj_def):
        computed = np.concatenate([p['data']['data'] for p in proj], 0)
        angle = np.rad2deg(linalg.subspace_angles(computed.T, mixing)[0])
        assert angle < 1e-5


def test_make_eeg_average_ref_proj():
    """Test EEG average reference projection."""
    raw = read_raw_fif(raw_fname, preload=True)
    eeg = pick_types(raw.info, meg=False, eeg=True)

    # No average EEG reference
    assert not np.all(raw._data[eeg].mean(axis=0) < 1e-19)

    # Apply average EEG reference
    car = make_eeg_average_ref_proj(raw.info)
    reref = raw.copy()
    reref.add_proj(car)
    reref.apply_proj()
    assert_array_almost_equal(reref._data[eeg].mean(axis=0), 0, decimal=19)

    # Error when custom reference has already been applied
    with raw.info._unlock():
        raw.info['custom_ref_applied'] = True
    pytest.raises(RuntimeError, make_eeg_average_ref_proj, raw.info)

    # test that an average EEG ref is not added when doing proj
    raw.set_eeg_reference(projection=True)
    assert _has_eeg_average_ref_proj(raw.info['projs'])
    raw.del_proj(idx=-1)
    assert not _has_eeg_average_ref_proj(raw.info['projs'])
    raw.apply_proj()
    assert not _has_eeg_average_ref_proj(raw.info['projs'])


def test_has_eeg_average_ref_proj():
    """Test checking whether an EEG average reference exists."""
    assert not _has_eeg_average_ref_proj([])

    raw = read_raw_fif(raw_fname)
    raw.set_eeg_reference(projection=True)
    assert _has_eeg_average_ref_proj(raw.info['projs'])


def test_needs_eeg_average_ref_proj():
    """Test checking whether a recording needs an EEG average reference."""
    raw = read_raw_fif(raw_fname)
    assert _needs_eeg_average_ref_proj(raw.info)

    raw.set_eeg_reference(projection=True)
    assert not _needs_eeg_average_ref_proj(raw.info)

    # No EEG channels
    raw = read_raw_fif(raw_fname, preload=True)
    eeg = [raw.ch_names[c] for c in pick_types(raw.info, meg=False, eeg=True)]
    raw.drop_channels(eeg)
    assert not _needs_eeg_average_ref_proj(raw.info)

    # Custom ref flag set
    raw = read_raw_fif(raw_fname)
    with raw.info._unlock():
        raw.info['custom_ref_applied'] = True
    assert not _needs_eeg_average_ref_proj(raw.info)


def test_sss_proj():
    """Test `meg` proj option."""
    raw = read_raw_fif(raw_fname)
    raw.crop(0, 1.0).load_data().pick_types(meg=True, exclude=())
    raw.pick_channels(raw.ch_names[:51]).del_proj()
    raw_sss = maxwell_filter(raw, int_order=5, ext_order=2)
    sss_rank = 21  # really low due to channel picking
    assert len(raw_sss.info['projs']) == 0
    for meg, n_proj, want_rank in (('separate', 6, sss_rank),
                                   ('combined', 3, sss_rank - 3)):
        proj = compute_proj_raw(raw_sss, n_grad=3, n_mag=3, meg=meg,
                                verbose='error')
        this_raw = raw_sss.copy().add_proj(proj).apply_proj()
        assert len(this_raw.info['projs']) == n_proj
        sss_proj_rank = _compute_rank_int(this_raw)
        cov = compute_raw_covariance(this_raw, verbose='error')
        W, ch_names, rank = compute_whitener(cov, this_raw.info,
                                             return_rank=True)
        assert ch_names == this_raw.ch_names
        assert want_rank == sss_proj_rank == rank  # proper reduction
        if meg == 'combined':
            assert this_raw.info['projs'][0]['data']['col_names'] == ch_names
        else:
            mag_names = ch_names[2::3]
            assert this_raw.info['projs'][3]['data']['col_names'] == mag_names
