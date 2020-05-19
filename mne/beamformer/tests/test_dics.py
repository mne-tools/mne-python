# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD 3 clause

import copy as cp
import os.path as op

import pytest
from numpy.testing import assert_array_equal, assert_allclose
import numpy as np

import mne
from mne.datasets import testing
from mne.beamformer import (make_dics, apply_dics, apply_dics_epochs,
                            apply_dics_csd, tf_dics, read_beamformer,
                            Beamformer)
from mne.time_frequency import csd_morlet
from mne.utils import run_tests_if_main, object_diff, requires_h5py
from mne.proj import compute_proj_evoked, make_projector
from mne.surface import _compute_nearest

data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')

subjects_dir = op.join(data_path, 'subjects')


@pytest.fixture(scope='module', params=[testing._pytest_param()])
def _load_forward():
    """Load forward models."""
    fwd_free = mne.read_forward_solution(fname_fwd)
    fwd_free = mne.pick_types_forward(fwd_free, meg=True, eeg=False)
    fwd_free = mne.convert_forward_solution(fwd_free, surf_ori=False)
    fwd_surf = mne.convert_forward_solution(fwd_free, surf_ori=True,
                                            use_cps=False)
    fwd_fixed = mne.convert_forward_solution(fwd_free, force_fixed=True,
                                             use_cps=False)
    fwd_vol = mne.read_forward_solution(fname_fwd_vol)
    return fwd_free, fwd_surf, fwd_fixed, fwd_vol


def _simulate_data(fwd, idx):  # Somewhere on the frontal lobe by default
    """Simulate an oscillator on the cortex."""
    source_vertno = fwd['src'][0]['vertno'][idx]

    sfreq = 50.  # Hz.
    times = np.arange(10 * sfreq) / sfreq  # 10 seconds of data
    signal = np.sin(20 * 2 * np.pi * times)  # 20 Hz oscillator
    signal[:len(times) // 2] *= 2  # Make signal louder at the beginning
    signal *= 1e-9  # Scale to be in the ballpark of MEG data

    # Construct a SourceEstimate object that describes the signal at the
    # cortical level.
    stc = mne.SourceEstimate(
        signal[np.newaxis, :],
        vertices=[[source_vertno], []],
        tmin=0,
        tstep=1 / sfreq,
        subject='sample',
    )

    # Create an info object that holds information about the sensors
    info = mne.create_info(fwd['info']['ch_names'], sfreq, ch_types='grad')
    info.update(fwd['info'])  # Merge in sensor position information
    # heavily decimate sensors to make it much faster
    info = mne.pick_info(info, np.arange(info['nchan'])[::5])
    fwd = mne.pick_channels_forward(fwd, info['ch_names'])

    # Run the simulated signal through the forward model, obtaining
    # simulated sensor data.
    raw = mne.apply_forward_raw(fwd, stc, info)

    # Add a little noise
    random = np.random.RandomState(42)
    noise = random.randn(*raw._data.shape) * 1e-14
    raw._data += noise

    # Define a single epoch (weird baseline but shouldn't matter)
    epochs = mne.Epochs(raw, [[0, 0, 1]], event_id=1, tmin=0,
                        tmax=raw.times[-1], baseline=(0., 0.), preload=True)
    evoked = epochs.average()

    # Compute the cross-spectral density matrix
    csd = csd_morlet(epochs, frequencies=[10, 20], n_cycles=[5, 10], decim=10)

    labels = mne.read_labels_from_annot(
        'sample', hemi='lh', subjects_dir=subjects_dir)
    label = [
        label for label in labels if np.in1d(source_vertno, label.vertices)[0]]
    assert len(label) == 1
    label = label[0]
    vertices = np.intersect1d(label.vertices, fwd['src'][0]['vertno'])
    source_ind = vertices.tolist().index(source_vertno)
    assert vertices[source_ind] == source_vertno
    return epochs, evoked, csd, source_vertno, label, vertices, source_ind


def _test_weight_norm(filters, norm=1):
    """Test weight normalization."""
    for ws in filters['weights']:
        ws = ws.reshape(-1, filters['n_orient'], ws.shape[1])
        for w in ws:
            assert_allclose(np.trace(w.dot(w.conjugate().T)), norm)


idx_param = pytest.mark.parametrize('idx, mat_tol, vol_tol', [
    (0, 0.055, 0.),
    (100, 0.020, 0.007),
    (200, 0., 0.015),
    (233, 0.035, 0.),
])


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_h5py
@idx_param
def test_make_dics(tmpdir, _load_forward, idx, mat_tol, vol_tol):
    """Test making DICS beamformer filters."""
    # We only test proper handling of parameters here. Testing the results is
    # done in test_apply_dics_timeseries and test_apply_dics_csd.

    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, _, csd, _, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    with pytest.raises(RuntimeError, match='several sensor types'):
        make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None)
    epochs.pick_types(meg='grad')

    with pytest.raises(ValueError, match="Invalid value for the 'pick_ori'"):
        make_dics(epochs.info, fwd_fixed, csd, pick_ori="notexistent")
    with pytest.raises(ValueError, match='rank, if str'):
        make_dics(epochs.info, fwd_fixed, csd, rank='foo')
    with pytest.raises(TypeError, match='rank must be'):
        make_dics(epochs.info, fwd_fixed, csd, rank=1.)

    # Test if fixed forward operator is detected when picking normal
    # orientation
    with pytest.raises(ValueError, match='forward operator with free ori'):
        make_dics(epochs.info, fwd_fixed, csd, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    with pytest.raises(ValueError, match='oriented in surface coordinates'):
        make_dics(epochs.info, fwd_free, csd, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    with pytest.raises(ValueError, match='oriented in surface coordinates'):
        make_dics(epochs.info, fwd_vol, csd, pick_ori="normal")

    # Test invalid combinations of parameters
    with pytest.raises(ValueError, match='reduce_rank cannot be used with'):
        make_dics(epochs.info, fwd_free, csd, inversion='single',
                  reduce_rank=True)
    with pytest.raises(ValueError, match='not stable with depth'):
        make_dics(epochs.info, fwd_free, csd, weight_norm='unit-noise-gain',
                  inversion='single', normalize_fwd=True)

    # Sanity checks on the returned filters
    n_freq = len(csd.frequencies)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    n_verts = len(vertices)
    n_orient = 3

    n_channels = len(epochs.ch_names)
    # Test return values
    filters = make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None,
                        weight_norm='unit-noise-gain', normalize_fwd=False)
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert np.iscomplexobj(filters['weights'])
    assert filters['csd'] == csd
    assert filters['ch_names'] == epochs.ch_names
    assert_array_equal(filters['proj'], np.eye(n_channels))
    assert_array_equal(filters['vertices'][0], vertices)
    assert_array_equal(filters['vertices'][1], [])  # Label was on the LH
    assert filters['subject'] == fwd_free['src']._subject
    assert filters['pick_ori'] is None
    assert filters['n_orient'] == n_orient
    assert filters['inversion'] == 'single'
    assert not filters['normalize_fwd']
    assert filters['weight_norm'] == 'unit-noise-gain'
    assert 'DICS' in repr(filters)
    assert 'subject "sample"' in repr(filters)
    assert str(len(vertices)) in repr(filters)
    assert str(n_channels) in repr(filters)
    assert 'rank' not in repr(filters)
    _test_weight_norm(filters)

    # Test picking orientations. Also test weight norming under these different
    # conditions.
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', weight_norm='unit-noise-gain',
                        normalize_fwd=False)
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert filters['n_orient'] == n_orient
    _test_weight_norm(filters)

    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='max-power', weight_norm='unit-noise-gain',
                        normalize_fwd=False)
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert filters['n_orient'] == n_orient
    _test_weight_norm(filters)

    # From here on, only work on a single frequency
    csd = csd[0]

    # Test using a real-valued filter
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', real_filter=True)
    assert not np.iscomplexobj(filters['weights'])

    # Test forward normalization. When inversion='single', the power of a
    # unit-noise CSD should be 1, even without weight normalization.
    csd_noise = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    # Using [:, :] syntax for in-place broadcasting
    csd_noise._data[:, :] = np.eye(csd.n_channels)[inds][:, np.newaxis]
    filters = make_dics(epochs.info, fwd_surf, csd_noise, label=label,
                        weight_norm=None, normalize_fwd=True)
    w = filters['weights'][0][:3]
    assert_allclose(np.diag(w.dot(w.conjugate().T)), 1.0, rtol=1e-6, atol=0)

    # Test turning off both forward and weight normalization
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        weight_norm=None, normalize_fwd=False)
    w = filters['weights'][0][:3]
    assert not np.allclose(np.diag(w.dot(w.conjugate().T)), 1.0,
                           rtol=1e-2, atol=0)

    # Test neural-activity-index weight normalization. It should be a scaled
    # version of the unit-noise-gain beamformer.
    filters_nai = make_dics(
        epochs.info, fwd_surf, csd, label=label, pick_ori='max-power',
        weight_norm='nai', normalize_fwd=False)
    w_nai = filters_nai['weights'][0]
    filters_ung = make_dics(
        epochs.info, fwd_surf, csd, label=label, pick_ori='max-power',
        weight_norm='unit-noise-gain', normalize_fwd=False)
    w_ung = filters_ung['weights'][0]
    assert_allclose(np.corrcoef(np.abs(w_nai).ravel(),
                                np.abs(w_ung).ravel()), 1, atol=1e-7)

    # Test whether spatial filter contains src_type
    assert 'src_type' in filters

    fname = op.join(str(tmpdir), 'filters-dics.h5')
    filters.save(fname)
    filters_read = read_beamformer(fname)
    assert isinstance(filters, Beamformer)
    assert isinstance(filters_read, Beamformer)
    for key in ['tmin', 'tmax']:  # deal with strictness of object_diff
        setattr(filters['csd'], key, np.float(getattr(filters['csd'], key)))
    assert object_diff(filters, filters_read) == ''


def _fwd_dist(power, fwd, vertices, source_ind, tidx=1):
    idx = np.argmax(power.data[:, tidx])
    rr_got = fwd['src'][0]['rr'][vertices[idx]]
    rr_want = fwd['src'][0]['rr'][vertices[source_ind]]
    return np.linalg.norm(rr_got - rr_want)


@idx_param
def test_apply_dics_csd(_load_forward, idx, mat_tol, vol_tol):
    """Test applying a DICS beamformer to a CSD matrix."""
    fwd_free, fwd_surf, fwd_fixed, _ = _load_forward
    epochs, _, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    reg = 1  # Lots of regularization for our toy dataset

    with pytest.raises(RuntimeError, match='several sensor types'):
        make_dics(epochs.info, fwd_free, csd)
    epochs.pick_types(meg='grad')

    # Try different types of forward models
    assert label.hemi == 'lh'
    for fwd in [fwd_free, fwd_surf, fwd_fixed]:
        filters = make_dics(epochs.info, fwd, csd, label=label, reg=reg,
                            inversion='single')
        power, f = apply_dics_csd(csd, filters)
        assert f == [10, 20]

        # Did we find the true source at 20 Hz?
        dist = _fwd_dist(power, fwd_free, vertices, source_ind)
        assert dist == 0.

        # Is the signal stronger at 20 Hz than 10?
        assert power.data[source_ind, 1] > power.data[source_ind, 0]


@pytest.mark.parametrize('pick_ori', [None, 'normal', 'max-power'])
@pytest.mark.parametrize('inversion', ['single', 'matrix'])
@idx_param
def test_apply_dics_ori_inv(_load_forward, pick_ori, inversion, idx,
                            mat_tol, vol_tol):
    """Test picking different orientations and inversion modes."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, _, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    epochs.pick_types('grad')

    reg_ = 5 if inversion == 'matrix' else 1
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        reg=reg_, pick_ori=pick_ori,
                        inversion=inversion, normalize_fwd=False,
                        weight_norm='unit-noise-gain')
    power, f = apply_dics_csd(csd, filters)
    assert f == [10, 20]
    dist = _fwd_dist(power, fwd_surf, vertices, source_ind)
    assert dist <= (0.03 if inversion == 'matrix' else 0.)
    assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Test unit-noise-gain weighting
    csd_noise = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    csd_noise._data[...] = np.eye(csd.n_channels)[inds][:, np.newaxis]
    noise_power, f = apply_dics_csd(csd_noise, filters)
    assert_allclose(noise_power.data, 1., atol=1e-7)

    # Test filter with forward normalization instead of weight
    # normalization
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        reg=reg_, pick_ori=pick_ori,
                        inversion=inversion, weight_norm=None,
                        normalize_fwd=True)
    power, f = apply_dics_csd(csd, filters)
    assert f == [10, 20]
    dist = _fwd_dist(power, fwd_surf, vertices, source_ind)
    max_ = (mat_tol if inversion == 'matrix' else 0.)
    assert 0 <= dist <= max_
    assert power.data[source_ind, 1] > power.data[source_ind, 0]


def _nearest_vol_ind(fwd_vol, fwd, vertices, source_ind):
    return _compute_nearest(
        fwd_vol['source_rr'],
        fwd['src'][0]['rr'][vertices][source_ind][np.newaxis])[0]


@idx_param
def test_real(_load_forward, idx, mat_tol, vol_tol):
    """Test using a real-valued filter."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, _, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    epochs.pick_types('grad')
    reg = 1  # Lots of regularization for our toy dataset
    filters_real = make_dics(epochs.info, fwd_surf, csd, label=label, reg=reg,
                             real_filter=True)
    # Also test here that no warings are thrown - implemented to check whether
    # src should not be None warning occurs:
    with pytest.warns(None) as w:
        power, f = apply_dics_csd(csd, filters_real)
    assert len(w) == 0

    assert f == [10, 20]
    dist = _fwd_dist(power, fwd_surf, vertices, source_ind)
    assert dist == 0
    assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Test rank reduction
    filters_real = make_dics(epochs.info, fwd_surf, csd, label=label, reg=5,
                             pick_ori='max-power', inversion='matrix',
                             reduce_rank=True)
    power, f = apply_dics_csd(csd, filters_real)
    assert f == [10, 20]
    dist = _fwd_dist(power, fwd_surf, vertices, source_ind)
    assert dist == 0
    assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Test computing source power on a volume source space
    filters_vol = make_dics(epochs.info, fwd_vol, csd, reg=reg)
    power, f = apply_dics_csd(csd, filters_vol)
    vol_source_ind = _nearest_vol_ind(fwd_vol, fwd_surf, vertices, source_ind)
    assert f == [10, 20]
    dist = _fwd_dist(
        power, fwd_vol, fwd_vol['src'][0]['vertno'], vol_source_ind)
    assert dist <= vol_tol
    assert power.data[vol_source_ind, 1] > power.data[vol_source_ind, 0]

    # check whether a filters object without src_type throws expected warning
    del filters_vol['src_type']  # emulate 0.16 behaviour to cause warning
    with pytest.warns(RuntimeWarning, match='spatial filter does not contain '
                      'src_type'):
        apply_dics_csd(csd, filters_vol)


@pytest.mark.filterwarnings("ignore:The use of several sensor types with the"
                            ":RuntimeWarning")
@idx_param
def test_apply_dics_timeseries(_load_forward, idx, mat_tol, vol_tol):
    """Test DICS applied to timeseries data."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, evoked, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    reg = 5  # Lots of regularization for our toy dataset

    with pytest.raises(RuntimeError, match='several sensor types'):
        make_dics(evoked.info, fwd_surf, csd)
    evoked.pick_types(meg='grad')

    multiple_filters = make_dics(evoked.info, fwd_surf, csd, label=label,
                                 reg=reg)

    # Sanity checks on the resulting STC after applying DICS on evoked
    stcs = apply_dics(evoked, multiple_filters)
    assert isinstance(stcs, list)
    assert len(stcs) == len(multiple_filters['weights'])
    assert_array_equal(stcs[0].vertices[0], multiple_filters['vertices'][0])
    assert_array_equal(stcs[0].vertices[1], multiple_filters['vertices'][1])
    assert_allclose(stcs[0].times, evoked.times)

    # Applying filters for multiple frequencies on epoch data should fail
    with pytest.raises(ValueError, match='computed for a single frequency'):
        apply_dics_epochs(epochs, multiple_filters)

    # From now on, only apply filters with a single frequency (20 Hz).
    csd20 = csd.pick_frequency(20)
    filters = make_dics(evoked.info, fwd_surf, csd20, label=label, reg=reg)

    # Sanity checks on the resulting STC after applying DICS on epochs.
    # Also test here that no warnings are thrown - implemented to check whether
    # src should not be None warning occurs
    with pytest.warns(None) as w:
        stcs = apply_dics_epochs(epochs, filters)
    assert len(w) == 0

    assert isinstance(stcs, list)
    assert len(stcs) == 1
    assert_array_equal(stcs[0].vertices[0], filters['vertices'][0])
    assert_array_equal(stcs[0].vertices[1], filters['vertices'][1])
    assert_allclose(stcs[0].times, epochs.times)

    # Did we find the source?
    stc = (stcs[0] ** 2).mean()
    dist = _fwd_dist(stc, fwd_surf, vertices, source_ind, tidx=0)
    assert dist == 0

    # Apply filters to evoked
    stc = apply_dics(evoked, filters)
    stc = (stc ** 2).mean()
    dist = _fwd_dist(stc, fwd_surf, vertices, source_ind, tidx=0)
    assert dist == 0

    # Test if wrong channel selection is detected in application of filter
    evoked_ch = cp.deepcopy(evoked)
    evoked_ch.pick_channels(evoked_ch.ch_names[:-1])
    with pytest.raises(ValueError, match='MEG 2633 which is not present'):
        apply_dics(evoked_ch, filters)

    # Test whether projections are applied, by adding a custom projection
    filters_noproj = make_dics(evoked.info, fwd_surf, csd20, label=label)
    stc_noproj = apply_dics(evoked, filters_noproj)
    evoked_proj = evoked.copy()
    p = compute_proj_evoked(evoked_proj, n_grad=1, n_mag=0, n_eeg=0)
    proj_matrix = make_projector(p, evoked_proj.ch_names)[0]
    evoked_proj.info['projs'] += p
    filters_proj = make_dics(evoked_proj.info, fwd_surf, csd20, label=label)
    assert_array_equal(filters_proj['proj'], proj_matrix)
    stc_proj = apply_dics(evoked_proj, filters_proj)
    assert np.any(np.not_equal(stc_noproj.data, stc_proj.data))

    # Test detecting incompatible projections
    filters_proj['proj'] = filters_proj['proj'][:-1, :-1]
    with pytest.raises(ValueError, match='operands could not be broadcast'):
        apply_dics(evoked_proj, filters_proj)

    # Test returning a generator
    stcs = apply_dics_epochs(epochs, filters, return_generator=False)
    stcs_gen = apply_dics_epochs(epochs, filters, return_generator=True)
    assert_array_equal(stcs[0].data, next(stcs_gen).data)

    # Test computing timecourses on a volume source space
    filters_vol = make_dics(evoked.info, fwd_vol, csd20, reg=reg)
    stc = apply_dics(evoked, filters_vol)
    stc = (stc ** 2).mean()
    assert stc.data.shape[1] == 1
    vol_source_ind = _nearest_vol_ind(fwd_vol, fwd_surf, vertices, source_ind)
    dist = _fwd_dist(stc, fwd_vol, fwd_vol['src'][0]['vertno'], vol_source_ind,
                     tidx=0)
    assert dist <= vol_tol

    # check whether a filters object without src_type throws expected warning
    del filters_vol['src_type']  # emulate 0.16 behaviour to cause warning
    with pytest.warns(RuntimeWarning, match='filter does not contain src_typ'):
        apply_dics_epochs(epochs, filters_vol)


@pytest.mark.slowtest
@testing.requires_testing_data
@idx_param
def test_tf_dics(_load_forward, idx, mat_tol, vol_tol):
    """Test 5D time-frequency beamforming based on DICS."""
    fwd_free, fwd_surf, fwd_fixed, _ = _load_forward
    epochs, _, _, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    reg = 1  # Lots of regularization for our toy dataset

    tmin = 0
    tmax = 9
    tstep = 4
    win_lengths = [5, 5]
    frequencies = [10, 20]
    freq_bins = [(8, 12), (18, 22)]

    with pytest.raises(RuntimeError, match='several sensor types'):
        stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                       freq_bins=freq_bins, frequencies=frequencies,
                       decim=10, reg=reg, label=label)
    epochs.pick_types(meg='grad')
    # Compute DICS for two time windows and two frequencies
    for mode in ['fourier', 'multitaper', 'cwt_morlet']:
        stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                       mode=mode, freq_bins=freq_bins, frequencies=frequencies,
                       decim=10, reg=reg, label=label)

        # Did we find the true source at 20 Hz?
        dist = _fwd_dist(stcs[1], fwd_surf, vertices, source_ind, tidx=0)
        assert dist == 0
        dist = _fwd_dist(stcs[1], fwd_surf, vertices, source_ind, tidx=1)
        assert dist == 0

        # 20 Hz power should decrease over time
        assert stcs[1].data[source_ind, 0] > stcs[1].data[source_ind, 1]

        # 20 Hz power should be more than 10 Hz power at the true source
        assert stcs[1].data[source_ind, 0] > stcs[0].data[source_ind, 0]

    # Manually compute source power and compare with the last tf_dics result.
    source_power = []
    time_windows = [(0, 5), (4, 9)]
    for time_window in time_windows:
        csd = csd_morlet(epochs, frequencies=[frequencies[1]],
                         tmin=time_window[0], tmax=time_window[1], decim=10)
        csd = csd.sum()
        csd._data /= csd.n_fft
        filters = make_dics(epochs.info, fwd_surf, csd, reg=reg, label=label)
        stc_source_power, _ = apply_dics_csd(csd, filters)
        source_power.append(stc_source_power.data)

    # Comparing tf_dics results with dics_source_power results
    assert_allclose(stcs[1].data, np.array(source_power).squeeze().T, atol=0)

    # Test using noise csds. We're going to use identity matrices. That way,
    # since we're using unit-noise-gain weight normalization, there should be
    # no effect.
    stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                   mode='cwt_morlet', frequencies=frequencies, decim=10,
                   reg=reg, label=label, normalize_fwd=False,
                   weight_norm='unit-noise-gain')
    noise_csd = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    # Using [:, :] syntax for in-place broadcasting
    noise_csd._data[:, :] = 2 * np.eye(csd.n_channels)[inds][:, np.newaxis]
    noise_csd.n_fft = 2  # Dividing by n_fft should yield an identity CSD
    noise_csds = [noise_csd, noise_csd]  # Two frequency bins
    stcs_norm = tf_dics(epochs, fwd_surf, noise_csds, tmin, tmax, tstep,
                        win_lengths, mode='cwt_morlet',
                        frequencies=frequencies, decim=10, reg=reg,
                        label=label, normalize_fwd=False,
                        weight_norm='unit-noise-gain')
    assert_allclose(stcs_norm[0].data, stcs[0].data, atol=0)
    assert_allclose(stcs_norm[1].data, stcs[1].data, atol=0)

    # Test invalid parameter combinations
    with pytest.raises(ValueError, match='fourier.*freq_bins" parameter'):
        tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                mode='fourier', freq_bins=None)
    with pytest.raises(ValueError, match='cwt_morlet.*frequencies" param'):
        tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                mode='cwt_morlet', frequencies=None)

    # Test if incorrect number of noise CSDs is detected
    with pytest.raises(ValueError, match='One noise CSD object expected per'):
        tf_dics(epochs, fwd_surf, [noise_csds[0]], tmin, tmax, tstep,
                win_lengths, freq_bins=freq_bins)

    # Test if freq_bins and win_lengths incompatibility is detected
    with pytest.raises(ValueError, match='One time window length expected'):
        tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep,
                win_lengths=[0, 1, 2], freq_bins=freq_bins)

    # Test if time step exceeding window lengths is detected
    with pytest.raises(ValueError, match='Time step should not be larger'):
        tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep=0.15,
                win_lengths=[0.2, 0.1], freq_bins=freq_bins)

    # Test if incorrect number of n_ffts is detected
    with pytest.raises(ValueError, match='When specifying number of FFT'):
        tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep,
                win_lengths, freq_bins=freq_bins, n_ffts=[1])

    # Test if incorrect number of mt_bandwidths is detected
    with pytest.raises(ValueError, match='When using multitaper mode and'):
        tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep,
                win_lengths=win_lengths, freq_bins=freq_bins,
                mode='multitaper', mt_bandwidths=[20])

    # Test if subtracting evoked responses yields NaN's, since we only have one
    # epoch. Suppress division warnings.
    assert len(epochs) == 1, len(epochs)
    with np.errstate(invalid='ignore'):
        stcs = tf_dics(epochs, fwd_surf, None, tmin, tmax, tstep, win_lengths,
                       mode='cwt_morlet', frequencies=frequencies,
                       subtract_evoked=True, reg=reg, label=label, decim=20)
    assert np.all(np.isnan(stcs[0].data))


run_tests_if_main()
