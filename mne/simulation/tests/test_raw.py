# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Yousra Bekhti <yousra.bekhti@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from mne import (read_source_spaces, pick_types, read_trans, read_cov,
                 make_sphere_model, create_info, setup_volume_source_space,
                 find_events, Epochs, fit_dipole, transform_surface_to,
                 make_ad_hoc_cov, SourceEstimate, setup_source_space,
                 read_bem_solution, make_forward_solution,
                 convert_forward_solution, VolSourceEstimate,
                 make_bem_solution)
from mne.bem import _surfaces_to_bem
from mne.chpi import (read_head_pos, compute_chpi_amplitudes,
                      compute_chpi_locs, compute_head_pos, get_chpi_info)
from mne.tests.test_chpi import _assert_quats
from mne.datasets import testing
from mne.simulation import (simulate_sparse_stc, simulate_raw, add_eog,
                            add_ecg, add_chpi, add_noise)
from mne.source_space import _compare_source_spaces
from mne.simulation.source import SourceSimulator
from mne.label import Label
from mne.surface import _get_ico_surface
from mne.io import read_raw_fif, RawArray
from mne.io.constants import FIFF
from mne.utils import catch_logging, check_version

base_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname_short = op.join(base_path, 'test_raw.fif')

data_path = testing.data_path(download=False)
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
cov_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-cov.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc-trans.fif')
subjects_dir = op.join(data_path, 'subjects')
bem_path = op.join(subjects_dir, 'sample', 'bem')
src_fname = op.join(bem_path, 'sample-oct-2-src.fif')
bem_fname = op.join(bem_path, 'sample-320-320-320-bem-sol.fif')
bem_1_fname = op.join(bem_path, 'sample-320-bem-sol.fif')

raw_chpi_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
pos_fname = op.join(data_path, 'SSS', 'test_move_anon_raw_subsampled.pos')


def _assert_iter_sim(raw_sim, raw_new, new_event_id):
    events = find_events(raw_sim, initial_event=True)
    events_tuple = find_events(raw_new, initial_event=True)
    assert_array_equal(events_tuple[:, :2], events[:, :2])
    assert_array_equal(events_tuple[:, 2], new_event_id)
    data_sim = raw_sim[:-1][0]
    data_new = raw_new[:-1][0]
    assert_array_equal(data_new, data_sim)


@pytest.mark.slowtest
def test_iterable():
    """Test iterable support for simulate_raw."""
    raw = read_raw_fif(raw_fname_short).load_data()
    raw.pick_channels(raw.ch_names[:10] + ['STI 014'])
    src = setup_volume_source_space(
        pos=dict(rr=[[-0.05, 0, 0], [0.1, 0, 0]],
                 nn=[[0, 1., 0], [0, 1., 0]]))
    assert src.kind == 'discrete'
    trans = None
    sphere = make_sphere_model(head_radius=None, info=raw.info)
    tstep = 1. / raw.info['sfreq']
    rng = np.random.RandomState(0)
    vertices = [np.array([1])]
    data = rng.randn(1, 2)
    stc = VolSourceEstimate(data, vertices, 0, tstep)
    assert isinstance(stc.vertices[0], np.ndarray)
    with pytest.raises(ValueError, match='at least three time points'):
        simulate_raw(raw.info, stc, trans, src, sphere, None)
    data = rng.randn(1, 1000)
    n_events = (len(raw.times) - 1) // 1000 + 1
    stc = VolSourceEstimate(data, vertices, 0, tstep)
    assert isinstance(stc.vertices[0], np.ndarray)
    raw_sim = simulate_raw(raw.info, [stc] * 15, trans, src, sphere, None,
                           first_samp=raw.first_samp)
    raw_sim.crop(0, raw.times[-1])
    assert_allclose(raw.times, raw_sim.times)
    events = find_events(raw_sim, initial_event=True)
    assert len(events) == n_events
    assert_array_equal(events[:, 2], 1)

    # Degenerate STCs
    with pytest.raises(RuntimeError,
                       match=r'Iterable did not provide stc\[0\]'):
        simulate_raw(raw.info, [], trans, src, sphere, None)
    # tuple with ndarray
    event_data = np.zeros(len(stc.times), int)
    event_data[0] = 3
    raw_new = simulate_raw(raw.info, [(stc, event_data)] * 15,
                           trans, src, sphere, None, first_samp=raw.first_samp)
    assert raw_new.n_times == 15000
    raw_new.crop(0, raw.times[-1])
    _assert_iter_sim(raw_sim, raw_new, 3)
    with pytest.raises(ValueError, match='event data had shape .* but need'):
        simulate_raw(raw.info, [(stc, event_data[:-1])], trans, src, sphere,
                     None)
    with pytest.raises(ValueError, match='stim_data in a stc tuple .* int'):
        simulate_raw(raw.info, [(stc, event_data * 1.)], trans, src, sphere,
                     None)

    # iterable
    def stc_iter():
        stim_data = np.zeros(len(stc.times), int)
        stim_data[0] = 4
        ii = 0
        while ii < 15:
            ii += 1
            yield (stc, stim_data)
    raw_new = simulate_raw(raw.info, stc_iter(), trans, src, sphere, None,
                           first_samp=raw.first_samp)
    raw_new.crop(0, raw.times[-1])
    _assert_iter_sim(raw_sim, raw_new, 4)

    def stc_iter_bad():
        ii = 0
        while ii < 100:
            ii += 1
            yield (stc, 4, 3)
    with pytest.raises(ValueError, match='stc, if tuple, must be length'):
        simulate_raw(raw.info, stc_iter_bad(), trans, src, sphere, None)
    _assert_iter_sim(raw_sim, raw_new, 4)

    def stc_iter_bad():
        ii = 0
        while ii < 100:
            ii += 1
            stc_new = stc.copy()
            stc_new.vertices[0] = np.array([ii % 2])
            yield stc_new
    with pytest.raises(RuntimeError, match=r'Vertex mismatch for stc\[1\]'):
        simulate_raw(raw.info, stc_iter_bad(), trans, src, sphere, None)

    # Forward omission
    vertices = [np.array([0, 1])]
    data = rng.randn(2, 1000)
    stc = VolSourceEstimate(data, vertices, 0, tstep)
    assert isinstance(stc.vertices[0], np.ndarray)
    # XXX eventually we should support filtering based on sphere radius, too,
    # by refactoring the code in source_space.py that does it!
    surf = _get_ico_surface(3)
    surf['rr'] *= 60  # mm
    model = _surfaces_to_bem([surf], [FIFF.FIFFV_BEM_SURF_ID_BRAIN], [0.3])
    bem = make_bem_solution(model)
    with pytest.warns(RuntimeWarning,
                      match='1 of 2 SourceEstimate vertices'):
        simulate_raw(raw.info, stc, trans, src, bem, None)


def _make_stc(raw, src):
    """Make a STC."""
    seed = 42
    sfreq = raw.info['sfreq']  # Hz
    tstep = 1. / sfreq
    n_samples = len(raw.times) // 10
    times = np.arange(0, n_samples) * tstep
    stc = simulate_sparse_stc(src, 10, times, random_state=seed)
    return stc


@pytest.fixture(scope='function', params=[testing._pytest_param()])
def raw_data():
    """Get some starting data."""
    # raw with ECG channel
    raw = read_raw_fif(raw_fname).crop(0., 5.0).load_data()
    data_picks = pick_types(raw.info, meg=True, eeg=True)
    other_picks = pick_types(raw.info, meg=False, stim=True, eog=True)
    picks = np.sort(np.concatenate((data_picks[::16], other_picks)))
    raw = raw.pick_channels([raw.ch_names[p] for p in picks])
    raw.info.normalize_proj()
    ecg = RawArray(np.zeros((1, len(raw.times))),
                   create_info(['ECG 063'], raw.info['sfreq'], 'ecg'))
    with ecg.info._unlock():
        for key in ('dev_head_t', 'highpass', 'lowpass', 'dig'):
            ecg.info[key] = raw.info[key]
    raw.add_channels([ecg])

    src = read_source_spaces(src_fname)
    trans = read_trans(trans_fname)
    sphere = make_sphere_model('auto', 'auto', raw.info)
    stc = _make_stc(raw, src)
    return raw, src, stc, trans, sphere


def _get_head_pos_sim(raw):
    head_pos_sim = dict()
    # these will be at 1., 2., ... sec
    shifts = [[0.001, 0., -0.001], [-0.001, 0.001, 0.]]

    for time_key, shift in enumerate(shifts):
        # Create 4x4 matrix transform and normalize
        temp_trans = deepcopy(raw.info['dev_head_t'])
        temp_trans['trans'][:3, 3] += shift
        head_pos_sim[time_key + 1.] = temp_trans['trans']
    return head_pos_sim


def test_simulate_raw_sphere(raw_data, tmp_path):
    """Test simulation of raw data with sphere model."""
    seed = 42
    raw, src, stc, trans, sphere = raw_data
    assert len(pick_types(raw.info, meg=False, ecg=True)) == 1
    tempdir = str(tmp_path)

    # head pos
    head_pos_sim = _get_head_pos_sim(raw)

    #
    # Test raw simulation with basic parameters
    #
    raw.info.normalize_proj()
    cov = read_cov(cov_fname)
    cov['projs'] = raw.info['projs']
    raw.info['bads'] = raw.ch_names[:1]
    sphere_norad = make_sphere_model('auto', None, raw.info)
    raw_meg = raw.copy().pick_types(meg=True)
    raw_sim = simulate_raw(raw_meg.info, stc, trans, src, sphere_norad,
                           head_pos=head_pos_sim)
    # Test IO on processed data
    test_outname = op.join(tempdir, 'sim_test_raw.fif')
    raw_sim.save(test_outname)

    raw_sim_loaded = read_raw_fif(test_outname, preload=True)
    assert_allclose(raw_sim_loaded[:][0], raw_sim[:][0], rtol=1e-6, atol=1e-20)
    del raw_sim

    # make sure it works with EEG-only and MEG-only
    raw_sim_meg = simulate_raw(
        raw.copy().pick_types(meg=True, eeg=False).info,
        stc, trans, src, sphere)
    raw_sim_eeg = simulate_raw(
        raw.copy().pick_types(meg=False, eeg=True).info,
        stc, trans, src, sphere)
    raw_sim_meeg = simulate_raw(
        raw.copy().pick_types(meg=True, eeg=True).info,
        stc, trans, src, sphere)
    for this_raw in (raw_sim_meg, raw_sim_eeg, raw_sim_meeg):
        add_eog(this_raw, random_state=seed)
    for this_raw in (raw_sim_meg, raw_sim_meeg):
        add_ecg(this_raw, random_state=seed)
    with pytest.raises(RuntimeError, match='only add ECG artifacts if MEG'):
        add_ecg(raw_sim_eeg)
    assert_allclose(np.concatenate((raw_sim_meg[:][0], raw_sim_eeg[:][0])),
                    raw_sim_meeg[:][0], rtol=1e-7, atol=1e-20)
    del raw_sim_meg, raw_sim_eeg, raw_sim_meeg

    # check that raw-as-info is supported
    n_samp = len(stc.times)
    raw_crop = raw.copy().crop(0., (n_samp - 1.) / raw.info['sfreq'])
    assert len(raw_crop.times) == len(stc.times)
    raw_sim = simulate_raw(raw_crop.info, stc, trans, src, sphere)
    with catch_logging() as log:
        raw_sim_2 = simulate_raw(raw_crop.info, stc, trans, src, sphere,
                                 verbose=True)
    log = log.getvalue()
    assert '1 STC iteration provided' in log
    assert len(raw_sim_2.times) == n_samp
    assert_allclose(raw_sim[:, :n_samp][0],
                    raw_sim_2[:, :n_samp][0], rtol=1e-5, atol=1e-30)
    del raw_sim, raw_sim_2

    # check that different interpolations are similar given small movements
    raw_sim = simulate_raw(raw.info, stc, trans, src, sphere,
                           head_pos=head_pos_sim, interp='linear')
    raw_sim_hann = simulate_raw(raw.info, stc, trans, src, sphere,
                                head_pos=head_pos_sim, interp='hann')
    assert_allclose(raw_sim[:][0], raw_sim_hann[:][0], rtol=1e-1, atol=1e-14)
    del raw_sim_hann

    # check that new Generator objects can be used
    if check_version('numpy', '1.17'):
        random_state = np.random.default_rng(seed)
        add_ecg(raw_sim, random_state=random_state)
        add_eog(raw_sim, random_state=random_state)


def test_degenerate(raw_data):
    """Test degenerate conditions."""
    raw, src, stc, trans, sphere = raw_data
    info = raw.info
    # Make impossible transform (translate up into helmet) and ensure failure
    hp_err = _get_head_pos_sim(raw)
    hp_err[1.][2, 3] -= 0.1  # z trans upward 10cm
    with pytest.raises(RuntimeError, match='collided with inner skull'):
        simulate_raw(info, stc, trans, src, sphere, head_pos=hp_err)
    # other degenerate conditions
    with pytest.raises(TypeError, match='info must be an instance of'):
        simulate_raw('foo', stc, trans, src, sphere)
    with pytest.raises(TypeError, match='stc must be an instance of'):
        simulate_raw(info, 'foo', trans, src, sphere)
    with pytest.raises(ValueError, match='stc must have at least three time'):
        simulate_raw(info, stc.copy().crop(0, 0), trans, src, sphere)
    with pytest.raises(TypeError, match='must be an instance of Info'):
        simulate_raw(0, stc, trans, src, sphere)
    stc_bad = stc.copy()
    stc_bad.tstep += 0.1
    with pytest.raises(ValueError, match='same sample rate'):
        simulate_raw(info, stc_bad, trans, src, sphere)
    with pytest.raises(ValueError, match='interp must be one of'):
        simulate_raw(info, stc, trans, src, sphere, interp='foo')
    with pytest.raises(TypeError, match='unknown head_pos type'):
        simulate_raw(info, stc, trans, src, sphere, head_pos=1.)
    head_pos_sim_err = _get_head_pos_sim(raw)
    head_pos_sim_err[-1.] = head_pos_sim_err[1.]  # negative time
    with pytest.raises(RuntimeError, match='All position times'):
        simulate_raw(info, stc, trans, src, sphere,
                     head_pos=head_pos_sim_err)
    raw_bad = raw.copy()
    with raw_bad.info._unlock():
        raw_bad.info['dig'] = None
    with pytest.raises(RuntimeError, match='Cannot fit headshape'):
        add_eog(raw_bad)


@pytest.mark.slowtest
def test_simulate_raw_bem(raw_data):
    """Test simulation of raw data with BEM."""
    raw, src_ss, stc, trans, sphere = raw_data
    src = setup_source_space('sample', 'oct1', subjects_dir=subjects_dir)
    for s in src:
        s['nuse'] = 3
        s['vertno'] = src[1]['vertno'][:3]
        s['inuse'].fill(0)
        s['inuse'][s['vertno']] = 1
    # use different / more complete STC here
    vertices = [s['vertno'] for s in src]
    stc = SourceEstimate(np.eye(sum(len(v) for v in vertices)), vertices,
                         0, 1. / raw.info['sfreq'])
    stcs = [stc] * 15
    raw_sim_sph = simulate_raw(raw.info, stcs, trans, src, sphere)
    raw_sim_bem = simulate_raw(raw.info, stcs, trans, src, bem_fname)
    # some components (especially radial) might not match that well,
    # so just make sure that most components have high correlation
    assert_array_equal(raw_sim_sph.ch_names, raw_sim_bem.ch_names)
    picks = pick_types(raw.info, meg=True, eeg=True)
    n_ch = len(picks)
    corr = np.corrcoef(raw_sim_sph[picks][0], raw_sim_bem[picks][0])
    assert_array_equal(corr.shape, (2 * n_ch, 2 * n_ch))
    med_corr = np.median(np.diag(corr[:n_ch, -n_ch:]))
    assert med_corr > 0.65
    # do some round-trip localization
    for s in src:
        transform_surface_to(s, 'head', trans)
    locs = np.concatenate([s['rr'][s['vertno']] for s in src])
    tmax = (len(locs) - 1) / raw.info['sfreq']
    cov = make_ad_hoc_cov(raw.info)
    # The tolerance for the BEM is surprisingly high (28) but I get the same
    # result when using MNE-C and Xfit, even when using a proper 5120 BEM :(
    for use_raw, bem, tol in ((raw_sim_sph, sphere, 2),
                              (raw_sim_bem, bem_fname, 31)):
        events = find_events(use_raw, 'STI 014')
        assert len(locs) == 6
        evoked = Epochs(use_raw, events, 1, 0, tmax, baseline=None).average()
        assert len(evoked.times) == len(locs)
        fits = fit_dipole(evoked, cov, bem, trans, min_dist=1.)[0].pos
        diffs = np.sqrt(np.sum((locs - fits) ** 2, axis=-1)) * 1000
        med_diff = np.median(diffs)
        assert med_diff < tol, '%s: %s' % (bem, med_diff)
    # also test event timings with SourceSimulator
    first_samp = raw.first_samp
    events = find_events(raw, initial_event=True, verbose=False)
    evt_times = events[:, 0]
    assert len(events) == 3
    labels_sim = [[], [], []]  # random l+r hemisphere points
    labels_sim[0] = Label([src_ss[0]['vertno'][1]], hemi='lh')
    labels_sim[1] = Label([src_ss[0]['vertno'][4]], hemi='lh')
    labels_sim[2] = Label([src_ss[1]['vertno'][2]], hemi='rh')
    wf_sim = np.array([2, 1, 0])
    for this_fs in (0, first_samp):
        ss = SourceSimulator(src_ss, 1. / raw.info['sfreq'],
                             first_samp=this_fs)
        for i in range(3):
            ss.add_data(labels_sim[i], wf_sim, events[np.newaxis, i])
        assert ss.n_times == evt_times[-1] + len(wf_sim) - this_fs
    raw_sim = simulate_raw(raw.info, ss, src=src_ss, bem=bem_fname,
                           first_samp=first_samp)
    data = raw_sim.get_data()
    amp0 = data[:, evt_times - first_samp].max()
    amp1 = data[:, evt_times + 1 - first_samp].max()
    amp2 = data[:, evt_times + 2 - first_samp].max()
    assert_allclose(amp0 / amp1, wf_sim[0] / wf_sim[1], rtol=1e-5)
    assert amp2 == 0
    assert raw_sim.n_times == ss.n_times


@pytest.mark.slowtest  # slow on Windows Azure
def test_simulate_round_trip(raw_data):
    """Test simulate_raw round trip calculations."""
    # Check a diagonal round-trip
    raw, src, stc, trans, sphere = raw_data
    raw.pick_types(meg=True, stim=True)
    bem = read_bem_solution(bem_1_fname)
    old_bem = bem.copy()
    old_src = src.copy()
    old_trans = trans.copy()
    fwd = make_forward_solution(raw.info, trans, src, bem)
    # no omissions
    assert (sum(len(s['vertno']) for s in src) ==
            sum(len(s['vertno']) for s in fwd['src']) ==
            36)
    # make sure things were not modified
    assert (old_bem['surfs'][0]['coord_frame'] ==
            bem['surfs'][0]['coord_frame'])
    assert trans == old_trans
    _compare_source_spaces(src, old_src)
    data = np.eye(fwd['nsource'])
    raw.crop(0, len(data) / raw.info['sfreq'], include_tmax=False)
    stc = SourceEstimate(data, [s['vertno'] for s in fwd['src']],
                         0, 1. / raw.info['sfreq'])
    for use_fwd in (None, fwd):
        if use_fwd is None:
            use_trans, use_src, use_bem = trans, src, bem
        else:
            use_trans = use_src = use_bem = None
        this_raw = simulate_raw(raw.info, stc, use_trans, use_src, use_bem,
                                forward=use_fwd)
        this_raw.pick_types(meg=True, eeg=True)
        assert (old_bem['surfs'][0]['coord_frame'] ==
                bem['surfs'][0]['coord_frame'])
        assert trans == old_trans
        _compare_source_spaces(src, old_src)
        this_fwd = convert_forward_solution(fwd, force_fixed=True)
        assert_allclose(this_raw[:][0], this_fwd['sol']['data'],
                        atol=1e-12, rtol=1e-6)
    with pytest.raises(ValueError, match='If forward is not None then'):
        simulate_raw(raw.info, stc, trans, src, bem, forward=fwd)
    # Not iterable
    with pytest.raises(TypeError, match='SourceEstimate, tuple, or iterable'):
        simulate_raw(raw.info, 0., trans, src, bem, None)
    # STC with a source that `src` does not have
    assert 0 not in src[0]['vertno']
    vertices = [[0, fwd['src'][0]['vertno'][0]], []]
    stc_bad = SourceEstimate(data[:2], vertices, 0, 1. / raw.info['sfreq'])
    with pytest.warns(RuntimeWarning,
                      match='1 of 2 SourceEstimate vertices'):
        simulate_raw(raw.info, stc_bad, trans, src, bem)
    assert 0 not in fwd['src'][0]['vertno']
    with pytest.warns(RuntimeWarning,
                      match='1 of 2 SourceEstimate vertices'):
        simulate_raw(raw.info, stc_bad, None, None, None, forward=fwd)
    # dev_head_t mismatch
    fwd['info']['dev_head_t']['trans'][0, 0] = 1.
    with pytest.raises(ValueError, match='dev_head_t.*does not match'):
        simulate_raw(raw.info, stc, None, None, None, forward=fwd)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_simulate_raw_chpi():
    """Test simulation of raw data with cHPI."""
    raw = read_raw_fif(raw_chpi_fname, allow_maxshield='yes')
    picks = np.arange(len(raw.ch_names))
    picks = np.setdiff1d(picks, pick_types(raw.info, meg=True, eeg=True)[::4])
    raw.load_data().pick_channels([raw.ch_names[pick] for pick in picks])
    raw.info.normalize_proj()
    sphere = make_sphere_model('auto', 'auto', raw.info)
    # make sparse spherical source space
    sphere_vol = tuple(sphere['r0']) + (sphere.radius,)
    src = setup_volume_source_space(sphere=sphere_vol, pos=70.,
                                    sphere_units='m')
    stcs = [_make_stc(raw, src)] * 15
    # simulate data with cHPI on
    raw_sim = simulate_raw(raw.info, stcs, None, src, sphere,
                           head_pos=pos_fname, interp='zero',
                           first_samp=raw.first_samp)
    # need to trim extra samples off this one
    raw_chpi = add_chpi(raw_sim.copy(), head_pos=pos_fname, interp='zero')
    # test cHPI indication
    hpi_freqs, hpi_pick, hpi_ons = get_chpi_info(raw.info, on_missing='raise')
    assert_allclose(raw_sim[hpi_pick][0], 0.)
    assert_allclose(raw_chpi[hpi_pick][0], hpi_ons.sum())
    # test that the cHPI signals make some reasonable values
    picks_meg = pick_types(raw.info, meg=True, eeg=False)
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)

    for picks in [picks_meg[:3], picks_eeg[:3]]:
        psd_sim, freqs_sim = (
            raw_sim.compute_psd(picks=picks).get_data(return_freqs=True))
        psd_chpi, freqs_chpi = (
            raw_chpi.compute_psd(picks=picks).get_data(return_freqs=True))

        assert_array_equal(freqs_sim, freqs_chpi)
        freq_idx = np.sort([np.argmin(np.abs(freqs_sim - f))
                            for f in hpi_freqs])
        if picks is picks_meg:
            assert (psd_chpi[:, freq_idx] >
                    100 * psd_sim[:, freq_idx]).all()
        else:
            assert_allclose(psd_sim, psd_chpi, atol=1e-20)

    # test localization based on cHPI information
    chpi_amplitudes = compute_chpi_amplitudes(raw, t_step_min=10.)
    coil_locs = compute_chpi_locs(raw.info, chpi_amplitudes)
    quats_sim = compute_head_pos(raw_chpi.info, coil_locs)
    quats = read_head_pos(pos_fname)
    _assert_quats(quats, quats_sim, dist_tol=5e-3, angle_tol=3.5,
                  vel_atol=0.03)  # velicity huge because of t_step_min above


@testing.requires_testing_data
def test_simulation_cascade():
    """Test that cascading operations do not overwrite data."""
    # Create 10 second raw dataset with zeros in the data matrix
    raw_null = read_raw_fif(raw_chpi_fname, allow_maxshield='yes')
    raw_null.crop(0, 1).pick_types(meg=True).load_data()
    raw_null.apply_function(lambda x: np.zeros_like(x))
    assert_array_equal(raw_null.get_data(), 0.)

    # Calculate independent signal additions
    raw_eog = raw_null.copy()
    add_eog(raw_eog, random_state=0)

    raw_ecg = raw_null.copy()
    add_ecg(raw_ecg, random_state=0)

    raw_noise = raw_null.copy()
    cov = make_ad_hoc_cov(raw_null.info)
    add_noise(raw_noise, cov, random_state=0)

    raw_chpi = raw_null.copy()
    add_chpi(raw_chpi)

    # Calculate Cascading signal additions
    raw_cascade = raw_null.copy()
    add_eog(raw_cascade, random_state=0)
    add_ecg(raw_cascade, random_state=0)
    add_chpi(raw_cascade)
    add_noise(raw_cascade, cov, random_state=0)

    cascade_data = raw_cascade.get_data()
    serial_data = 0.
    for raw_other in (raw_eog, raw_ecg, raw_noise, raw_chpi):
        serial_data += raw_other.get_data()

    assert_allclose(cascade_data, serial_data, atol=1e-20)
