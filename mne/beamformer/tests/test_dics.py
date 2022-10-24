# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD-3-Clause

import copy as cp
import os.path as op

import pytest
from numpy.testing import (assert_array_equal, assert_allclose,
                           assert_array_less)
import numpy as np

import mne
from mne import pick_types
from mne.beamformer import (make_dics, apply_dics, apply_dics_epochs,
                            apply_dics_tfr_epochs, apply_dics_csd,
                            read_beamformer, Beamformer)
from mne.beamformer._compute_beamformer import _prepare_beamformer_input
from mne.beamformer._dics import _prepare_noise_csd
from mne.beamformer.tests.test_lcmv import _assert_weight_norm
from mne.datasets import testing
from mne.io.constants import FIFF
from mne.io import read_info
from mne.io.pick import pick_info
from mne.proj import compute_proj_evoked, make_projector
from mne.surface import _compute_nearest
from mne.time_frequency import (CrossSpectralDensity, csd_morlet, EpochsTFR,
                                csd_tfr)
from mne.time_frequency.csd import _sym_mat_to_vector
from mne.transforms import invert_transform, apply_trans
from mne.utils import object_diff, requires_version, catch_logging

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
    with info._unlock():
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
    csd = csd_morlet(epochs, frequencies=[10, 20], n_cycles=[5, 10], decim=5)

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


idx_param = pytest.mark.parametrize('idx', [
    0,
    pytest.param(100, marks=pytest.mark.slowtest),
    200,
    pytest.param(233, marks=pytest.mark.slowtest),
])


def _rand_csd(rng, info):
    scales = mne.make_ad_hoc_cov(info).data
    n = scales.size
    # Some random complex correlation structure (with channel scalings)
    data = rng.randn(n, n) + 1j * rng.randn(n, n)
    data = data @ data.conj().T
    data *= scales
    data *= scales[:, np.newaxis]
    data.flat[::n + 1] = scales
    return data


def _make_rand_csd(info, csd):
    rng = np.random.RandomState(0)
    data = _rand_csd(rng, info)
    # now we need to have the same null space as the data csd
    s, u = np.linalg.eigh(csd.get_data(csd.frequencies[0]))
    mask = np.abs(s) >= s[-1] * 1e-7
    rank = mask.sum()
    assert rank == len(data) == len(info['ch_names'])
    noise_csd = CrossSpectralDensity(
        _sym_mat_to_vector(data), info['ch_names'], 0., csd.n_fft)
    return noise_csd, rank


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_version('h5io')
@idx_param
@pytest.mark.parametrize('whiten', [
    pytest.param(False, marks=pytest.mark.slowtest),
    True,
])
def test_make_dics(tmp_path, _load_forward, idx, whiten):
    """Test making DICS beamformer filters."""
    # We only test proper handling of parameters here. Testing the results is
    # done in test_apply_dics_timeseries and test_apply_dics_csd.

    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, _, csd, _, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    with pytest.raises(ValueError, match='several sensor types'):
        make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None)
    if whiten:
        noise_csd, rank = _make_rand_csd(epochs.info, csd)
        assert rank == len(epochs.info['ch_names']) == 62
    else:
        noise_csd = None
        epochs.pick_types(meg='grad')

    with pytest.raises(ValueError, match="Invalid value for the 'pick_ori'"):
        make_dics(epochs.info, fwd_fixed, csd, pick_ori="notexistent",
                  noise_csd=noise_csd)
    with pytest.raises(ValueError, match='rank, if str'):
        make_dics(epochs.info, fwd_fixed, csd, rank='foo', noise_csd=noise_csd)
    with pytest.raises(TypeError, match='rank must be'):
        make_dics(epochs.info, fwd_fixed, csd, rank=1., noise_csd=noise_csd)

    # Test if fixed forward operator is detected when picking normal
    # orientation
    with pytest.raises(ValueError, match='forward operator with free ori'):
        make_dics(epochs.info, fwd_fixed, csd, pick_ori="normal",
                  noise_csd=noise_csd)

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    with pytest.raises(ValueError, match='oriented in surface coordinates'):
        make_dics(epochs.info, fwd_free, csd, pick_ori="normal",
                  noise_csd=noise_csd)

    # Test if volume forward operator is detected when picking normal
    # orientation
    with pytest.raises(ValueError, match='oriented in surface coordinates'):
        make_dics(epochs.info, fwd_vol, csd, pick_ori="normal",
                  noise_csd=noise_csd)

    # Test invalid combinations of parameters
    with pytest.raises(ValueError, match='reduce_rank cannot be used with'):
        make_dics(epochs.info, fwd_free, csd, inversion='single',
                  reduce_rank=True, noise_csd=noise_csd)
    # TODO: Restore this?
    # with pytest.raises(ValueError, match='not stable with depth'):
    #     make_dics(epochs.info, fwd_free, csd, weight_norm='unit-noise-gain',
    #               inversion='single', depth=None)

    # Sanity checks on the returned filters
    n_freq = len(csd.frequencies)
    vertices = np.intersect1d(label.vertices, fwd_free['src'][0]['vertno'])
    n_verts = len(vertices)
    n_orient = 3

    n_channels = len(epochs.ch_names)
    # Test return values
    weight_norm = 'unit-noise-gain'
    inversion = 'single'
    filters = make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None,
                        weight_norm=weight_norm, depth=None, real_filter=False,
                        noise_csd=noise_csd, inversion=inversion)
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert np.iscomplexobj(filters['weights'])
    assert filters['csd'].ch_names == epochs.ch_names
    assert isinstance(filters['csd'], CrossSpectralDensity)
    assert filters['ch_names'] == epochs.ch_names
    assert_array_equal(filters['proj'], np.eye(n_channels))
    assert_array_equal(filters['vertices'][0], vertices)
    assert_array_equal(filters['vertices'][1], [])  # Label was on the LH
    assert filters['subject'] == fwd_free['src']._subject
    assert filters['pick_ori'] is None
    assert filters['is_free_ori']
    assert filters['inversion'] == inversion
    assert filters['weight_norm'] == weight_norm
    assert 'DICS' in repr(filters)
    assert 'subject "sample"' in repr(filters)
    assert str(len(vertices)) in repr(filters)
    assert str(n_channels) in repr(filters)
    assert 'rank' not in repr(filters)
    _, noise_cov = _prepare_noise_csd(csd, noise_csd, real_filter=False)
    _, _, _, _, G, _, _, _ = _prepare_beamformer_input(
        epochs.info, fwd_surf, label, 'vector', combine_xyz=False, exp=None,
        noise_cov=noise_cov)
    G.shape = (n_channels, n_verts, n_orient)
    G = G.transpose(1, 2, 0).conj()  # verts, orient, ch
    _assert_weight_norm(filters, G)

    inversion = 'matrix'
    filters = make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None,
                        weight_norm=weight_norm, depth=None,
                        noise_csd=noise_csd, inversion=inversion)
    _assert_weight_norm(filters, G)

    weight_norm = 'unit-noise-gain-invariant'
    inversion = 'single'
    filters = make_dics(epochs.info, fwd_surf, csd, label=label, pick_ori=None,
                        weight_norm=weight_norm, depth=None,
                        noise_csd=noise_csd, inversion=inversion)
    _assert_weight_norm(filters, G)

    # Test picking orientations. Also test weight norming under these different
    # conditions.
    weight_norm = 'unit-noise-gain'
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', weight_norm=weight_norm,
                        depth=None, noise_csd=noise_csd, inversion=inversion)
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert not filters['is_free_ori']
    _assert_weight_norm(filters, G)

    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='max-power', weight_norm=weight_norm,
                        depth=None, noise_csd=noise_csd, inversion=inversion)
    n_orient = 1
    assert filters['weights'].shape == (n_freq, n_verts * n_orient, n_channels)
    assert not filters['is_free_ori']
    _assert_weight_norm(filters, G)

    # From here on, only work on a single frequency
    csd = csd[0]

    # Test using a real-valued filter
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        pick_ori='normal', real_filter=True,
                        noise_csd=noise_csd)
    assert not np.iscomplexobj(filters['weights'])

    # Test forward normalization. When inversion='single', the power of a
    # unit-noise CSD should be 1, even without weight normalization.
    if not whiten:
        csd_noise = csd.copy()
        inds = np.triu_indices(csd.n_channels)
        # Using [:, :] syntax for in-place broadcasting
        csd_noise._data[:, :] = np.eye(csd.n_channels)[inds][:, np.newaxis]
        filters = make_dics(epochs.info, fwd_surf, csd_noise, label=label,
                            weight_norm=None, depth=1., noise_csd=noise_csd,
                            inversion='single')
        w = filters['weights'][0][:3]
        assert_allclose(np.diag(w.dot(w.conjugate().T)), 1.0, rtol=1e-6,
                        atol=0)

    # Test turning off both forward and weight normalization
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        weight_norm=None, depth=None, noise_csd=noise_csd)
    w = filters['weights'][0][:3]
    assert not np.allclose(np.diag(w.dot(w.conjugate().T)), 1.0,
                           rtol=1e-2, atol=0)

    # Test neural-activity-index weight normalization. It should be a scaled
    # version of the unit-noise-gain beamformer.
    filters_nai = make_dics(
        epochs.info, fwd_surf, csd, label=label, pick_ori='max-power',
        weight_norm='nai', depth=None, noise_csd=noise_csd)
    w_nai = filters_nai['weights'][0]
    filters_ung = make_dics(
        epochs.info, fwd_surf, csd, label=label, pick_ori='max-power',
        weight_norm='unit-noise-gain', depth=None, noise_csd=noise_csd)
    w_ung = filters_ung['weights'][0]
    assert_allclose(np.corrcoef(np.abs(w_nai).ravel(),
                                np.abs(w_ung).ravel()), 1, atol=1e-7)

    # Test whether spatial filter contains src_type
    assert 'src_type' in filters

    fname = op.join(str(tmp_path), 'filters-dics.h5')
    filters.save(fname)
    filters_read = read_beamformer(fname)
    assert isinstance(filters, Beamformer)
    assert isinstance(filters_read, Beamformer)
    for key in ['tmin', 'tmax']:  # deal with strictness of object_diff
        setattr(filters['csd'], key, np.float64(getattr(filters['csd'], key)))
    assert object_diff(filters, filters_read) == ''


def _fwd_dist(power, fwd, vertices, source_ind, tidx=1):
    idx = np.argmax(power.data[:, tidx])
    rr_got = fwd['src'][0]['rr'][vertices[idx]]
    rr_want = fwd['src'][0]['rr'][vertices[source_ind]]
    return np.linalg.norm(rr_got - rr_want)


@idx_param
@pytest.mark.parametrize('inversion, weight_norm', [
    ('single', None),
    ('matrix', 'unit-noise-gain'),
])
def test_apply_dics_csd(_load_forward, idx, inversion, weight_norm):
    """Test applying a DICS beamformer to a CSD matrix."""
    fwd_free, fwd_surf, fwd_fixed, _ = _load_forward
    epochs, _, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    reg = 1  # Lots of regularization for our toy dataset

    with pytest.raises(ValueError, match='several sensor types'):
        make_dics(epochs.info, fwd_free, csd)
    epochs.pick_types(meg='grad')

    # Try different types of forward models
    assert label.hemi == 'lh'
    for fwd in [fwd_free, fwd_surf, fwd_fixed]:
        filters = make_dics(epochs.info, fwd, csd, label=label, reg=reg,
                            inversion=inversion, weight_norm=weight_norm)
        power, f = apply_dics_csd(csd, filters)
        assert f == [10, 20]

        # Did we find the true source at 20 Hz?
        dist = _fwd_dist(power, fwd_free, vertices, source_ind)
        assert dist == 0.

        # Is the signal stronger at 20 Hz than 10?
        assert power.data[source_ind, 1] > power.data[source_ind, 0]


@pytest.mark.parametrize('pick_ori', [None, 'normal', 'max-power', 'vector'])
@pytest.mark.parametrize('inversion', ['single', 'matrix'])
@idx_param
def test_apply_dics_ori_inv(_load_forward, pick_ori, inversion, idx):
    """Test picking different orientations and inversion modes."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, _, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    epochs.pick_types(meg='grad')

    reg_ = 5 if inversion == 'matrix' else 1
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        reg=reg_, pick_ori=pick_ori,
                        inversion=inversion, depth=None,
                        weight_norm='unit-noise-gain')
    power, f = apply_dics_csd(csd, filters)
    assert f == [10, 20]
    dist = _fwd_dist(power, fwd_surf, vertices, source_ind)
    # This is 0. for unit-noise-gain-invariant:
    assert dist <= (0.02 if inversion == 'matrix' else 0.)
    assert power.data[source_ind, 1] > power.data[source_ind, 0]

    # Test unit-noise-gain weighting
    csd_noise = csd.copy()
    inds = np.triu_indices(csd.n_channels)
    csd_noise._data[...] = np.eye(csd.n_channels)[inds][:, np.newaxis]
    noise_power, f = apply_dics_csd(csd_noise, filters)
    want_norm = 3 if pick_ori in (None, 'vector') else 1
    assert_allclose(noise_power.data, want_norm, atol=1e-7)

    # Test filter with forward normalization instead of weight
    # normalization
    filters = make_dics(epochs.info, fwd_surf, csd, label=label,
                        reg=reg_, pick_ori=pick_ori,
                        inversion=inversion, weight_norm=None,
                        depth=1.)
    power, f = apply_dics_csd(csd, filters)
    assert f == [10, 20]
    dist = _fwd_dist(power, fwd_surf, vertices, source_ind)
    mat_tol = {0: 0.055, 100: 0.20, 200: 0.015, 233: 0.035}[idx]
    max_ = (mat_tol if inversion == 'matrix' else 0.)
    assert 0 <= dist <= max_
    assert power.data[source_ind, 1] > power.data[source_ind, 0]


def _nearest_vol_ind(fwd_vol, fwd, vertices, source_ind):
    return _compute_nearest(
        fwd_vol['source_rr'],
        fwd['src'][0]['rr'][vertices][source_ind][np.newaxis])[0]


@idx_param
def test_real(_load_forward, idx):
    """Test using a real-valued filter."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, _, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    epochs.pick_types(meg='grad')
    reg = 1  # Lots of regularization for our toy dataset
    filters_real = make_dics(epochs.info, fwd_surf, csd, label=label, reg=reg,
                             real_filter=True, inversion='single')
    # Also test here that no warnings are thrown - implemented to check whether
    # src should not be None warning occurs:
    power, f = apply_dics_csd(csd, filters_real)

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
    filters_vol = make_dics(epochs.info, fwd_vol, csd, reg=reg,
                            inversion='single')
    power, f = apply_dics_csd(csd, filters_vol)
    vol_source_ind = _nearest_vol_ind(fwd_vol, fwd_surf, vertices, source_ind)
    assert f == [10, 20]
    dist = _fwd_dist(
        power, fwd_vol, fwd_vol['src'][0]['vertno'], vol_source_ind)
    vol_tols = {100: 0.008, 200: 0.008}
    assert dist <= vol_tols.get(idx, 0.)
    assert power.data[vol_source_ind, 1] > power.data[vol_source_ind, 0]

    # check whether a filters object without src_type throws expected warning
    del filters_vol['src_type']  # emulate 0.16 behaviour to cause warning
    with pytest.warns(RuntimeWarning, match='spatial filter does not contain '
                      'src_type'):
        apply_dics_csd(csd, filters_vol)


@pytest.mark.filterwarnings("ignore:The use of several sensor types with the"
                            ":RuntimeWarning")
@idx_param
def test_apply_dics_timeseries(_load_forward, idx):
    """Test DICS applied to timeseries data."""
    fwd_free, fwd_surf, fwd_fixed, fwd_vol = _load_forward
    epochs, evoked, csd, source_vertno, label, vertices, source_ind = \
        _simulate_data(fwd_fixed, idx)
    reg = 5  # Lots of regularization for our toy dataset

    with pytest.raises(ValueError, match='several sensor types'):
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
    filters = make_dics(evoked.info, fwd_surf, csd20, label=label, reg=reg,
                        inversion='single')

    # Sanity checks on the resulting STC after applying DICS on epochs.
    # Also test here that no warnings are thrown - implemented to check whether
    # src should not be None warning occurs
    stcs = apply_dics_epochs(epochs, filters)

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
    evoked_proj.add_proj(p)
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
    filters_vol = make_dics(evoked.info, fwd_vol, csd20, reg=reg,
                            inversion='single')
    stc = apply_dics(evoked, filters_vol)
    stc = (stc ** 2).mean()
    assert stc.data.shape[1] == 1
    vol_source_ind = _nearest_vol_ind(fwd_vol, fwd_surf, vertices, source_ind)
    dist = _fwd_dist(stc, fwd_vol, fwd_vol['src'][0]['vertno'], vol_source_ind,
                     tidx=0)
    vol_tols = {100: 0.008, 200: 0.015}
    vol_tol = vol_tols.get(idx, 0.)
    assert dist <= vol_tol

    # check whether a filters object without src_type throws expected warning
    del filters_vol['src_type']  # emulate 0.16 behaviour to cause warning
    with pytest.warns(RuntimeWarning, match='filter does not contain src_typ'):
        apply_dics_epochs(epochs, filters_vol)


@testing.requires_testing_data
@pytest.mark.parametrize('return_generator', (True, False))
def test_apply_dics_tfr(return_generator):
    """Test DICS applied to time-frequency objects."""
    info = read_info(fname_raw)
    info = pick_info(info, pick_types(info, meg='grad'))
    forward = mne.read_forward_solution(fname_fwd)
    rng = np.random.default_rng(11)

    # Construct an EpochsTFR object filled with random data.
    n_epochs = 8
    n_chans = len(info.ch_names)
    freqs = [8, 9]
    n_times = 300
    times = np.arange(n_times) / info['sfreq']
    data = rng.random((n_epochs, n_chans, len(freqs), n_times))
    data *= 1e-6
    data = data + data * 1j  # add imag. component to simulate phase
    epochs_tfr = EpochsTFR(info, data, times=times, freqs=freqs)

    # Create a DICS beamformer and convert the EpochsTFR to source space.
    csd = csd_tfr(epochs_tfr)
    filters = make_dics(epochs_tfr.info, forward, csd, reg=0.05)
    stcs = apply_dics_tfr_epochs(epochs_tfr, filters, return_generator)

    # Check some basic properties of the returned SourceEstimate objects.
    if return_generator:
        stcs = list(stcs)
    assert_allclose(stcs[0][0].times, times)
    assert len(stcs) == len(epochs_tfr)  # check same number of epochs
    assert all([len(s) == len(freqs) for s in stcs])  # check nested freqs
    assert all([s.data.shape == (forward['nsource'], n_times)
                for these_stcs in stcs for s in these_stcs])

    # Compute power from the source space TFR. This should yield the same
    # result as the apply_dics_csd function.
    source_power = np.zeros((forward['nsource'], len(freqs)))
    for stcs_epoch in stcs:
        for i, stc_freq in enumerate(stcs_epoch):
            power = (stc_freq.data * np.conj(stc_freq.data)).real
            power = power.mean(axis=-1)  # mean over time
            # Scaling by sampling frequency for compatibility with Matlab
            power /= epochs_tfr.info['sfreq']
            source_power[:, i] += power.T
    source_power /= n_epochs

    ref_source_power, ref_freqs = apply_dics_csd(csd, filters)
    assert_allclose(freqs, ref_freqs)
    assert_allclose(ref_source_power.data, source_power)

    # Test that real-value only data fails, due to non-linearity of computing
    # power, it is recommended to transform to source-space first before
    # converting to power.
    with pytest.raises(RuntimeError,
                       match='Time-frequency data must be complex'):
        epochs_tfr_real = epochs_tfr.copy()
        epochs_tfr_real.data = epochs_tfr_real.data.real
        stcs = apply_dics_tfr_epochs(epochs_tfr_real, filters)

    filters_vector = filters.copy()
    filters_vector['pick_ori'] = 'vector'
    with pytest.warns(match='vector solution'):
        apply_dics_tfr_epochs(epochs_tfr, filters_vector)


def _cov_as_csd(cov, info):
    rng = np.random.RandomState(0)
    assert cov['data'].ndim == 2
    assert len(cov['data']) == len(cov['names'])
    # we need to make this have at least some complex structure
    data = cov['data'] + 1e-1 * _rand_csd(rng, info)
    assert data.dtype == np.complex128
    return CrossSpectralDensity(_sym_mat_to_vector(data), cov['names'], 0., 16)


# Just test free ori here (assume fixed is same as LCMV if these are)
# Changes here should be synced with test_lcmv.py
@pytest.mark.slowtest
@pytest.mark.parametrize(
    'reg, pick_ori, weight_norm, use_cov, depth, lower, upper, real_filter', [
        (0.05, 'vector', 'unit-noise-gain-invariant',
         False, None, 26, 28, True),
        (0.05, 'vector', 'unit-noise-gain', False, None, 13, 15, True),
        (0.05, 'vector', 'nai', False, None, 13, 15, True),
        (0.05, None, 'unit-noise-gain-invariant', False, None, 26, 28, False),
        (0.05, None, 'unit-noise-gain-invariant', True, None, 40, 42, False),
        (0.05, None, 'unit-noise-gain-invariant', True, None, 40, 42, True),
        (0.05, None, 'unit-noise-gain', False, None, 13, 14, False),
        (0.05, None, 'unit-noise-gain', True, None, 35, 37, False),
        (0.05, None, 'nai', True, None, 35, 37, False),
        (0.05, None, None, True, None, 12, 14, False),
        (0.05, None, None, True, 0.8, 39, 43, False),
        (0.05, 'max-power', 'unit-noise-gain-invariant', False, None, 17, 20,
         False),
        (0.05, 'max-power', 'unit-noise-gain', False, None, 17, 20, False),
        (0.05, 'max-power', 'unit-noise-gain', False, None, 17, 20, True),
        (0.05, 'max-power', 'nai', True, None, 21, 24, False),
        (0.05, 'max-power', None, True, None, 7, 10, False),
        (0.05, 'max-power', None, True, 0.8, 15, 18, False),
        # skip most no-reg tests, assume others are equal to LCMV if these are
        (0.00, None, None, True, None, 21, 32, False),
        (0.00, 'max-power', None, True, None, 13, 19, False),
    ])
def test_localization_bias_free(bias_params_free, reg, pick_ori, weight_norm,
                                use_cov, depth, lower, upper, real_filter):
    """Test localization bias for free-orientation DICS."""
    evoked, fwd, noise_cov, data_cov, want = bias_params_free
    noise_csd = _cov_as_csd(noise_cov, evoked.info)
    data_csd = _cov_as_csd(data_cov, evoked.info)
    del noise_cov, data_cov
    if not use_cov:
        evoked.pick_types(meg='grad')
        noise_csd = None
    filters = make_dics(
        evoked.info, fwd, data_csd, reg, noise_csd, pick_ori=pick_ori,
        weight_norm=weight_norm, depth=depth, real_filter=real_filter)
    loc = apply_dics(evoked, filters).data
    loc = np.linalg.norm(loc, axis=1) if pick_ori == 'vector' else np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper


@pytest.mark.parametrize(
    'weight_norm, lower, upper, lower_ori, upper_ori, real_filter', [
        ('unit-noise-gain-invariant', 57, 58, 0.60, 0.61, False),
        ('unit-noise-gain', 57, 58, 0.60, 0.61, False),
        ('unit-noise-gain', 57, 58, 0.60, 0.61, True),
        (None, 27, 28, 0.56, 0.57, False),
    ])
def test_orientation_max_power(bias_params_fixed, bias_params_free,
                               weight_norm, lower, upper, lower_ori, upper_ori,
                               real_filter):
    """Test orientation selection for bias for max-power DICS."""
    # we simulate data for the fixed orientation forward and beamform using
    # the free orientation forward, and check the orientation match at the end
    evoked, _, noise_cov, data_cov, want = bias_params_fixed
    noise_csd = _cov_as_csd(noise_cov, evoked.info)
    data_csd = _cov_as_csd(data_cov, evoked.info)
    del data_cov, noise_cov
    fwd = bias_params_free[1]
    filters = make_dics(evoked.info, fwd, data_csd, 0.05, noise_csd,
                        pick_ori='max-power', weight_norm=weight_norm,
                        depth=None, real_filter=real_filter)
    loc = np.abs(apply_dics(evoked, filters).data)
    ori = filters['max_power_ori'][0]
    assert ori.shape == (246, 3)
    loc = np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    max_idx = np.argmax(loc, axis=0)
    mask = want == max_idx  # ones that localized properly
    perc = mask.mean() * 100
    assert lower <= perc <= upper
    # Compute the dot products of our forward normals and
    # assert we get some hopefully reasonable agreement
    assert fwd['coord_frame'] == FIFF.FIFFV_COORD_HEAD
    nn = np.concatenate(
        [s['nn'][v] for s, v in zip(fwd['src'], filters['vertices'])])
    nn = nn[want]
    nn = apply_trans(invert_transform(fwd['mri_head_t']), nn, move=False)
    assert_allclose(np.linalg.norm(nn, axis=1), 1, atol=1e-6)
    assert_allclose(np.linalg.norm(ori, axis=1), 1, atol=1e-12)
    dots = np.abs((nn[mask] * ori[mask]).sum(-1))
    assert_array_less(dots, 1)
    assert_array_less(0, dots)
    got = np.mean(dots)
    assert lower_ori < got < upper_ori


@testing.requires_testing_data
@idx_param
@pytest.mark.parametrize('whiten', (False, True))
def test_make_dics_rank(_load_forward, idx, whiten):
    """Test making DICS beamformer filters with rank param."""
    _, fwd_surf, fwd_fixed, _ = _load_forward
    epochs, _, csd, _, label, _, _ = _simulate_data(fwd_fixed, idx)
    if whiten:
        noise_csd, want_rank = _make_rand_csd(epochs.info, csd)
        kind = 'mag + grad'
    else:
        noise_csd = None
        epochs.pick_types(meg='grad')
        want_rank = len(epochs.ch_names)
        assert want_rank == 41
        kind = 'grad'

    with catch_logging() as log:
        filters = make_dics(
            epochs.info, fwd_surf, csd, label=label, noise_csd=noise_csd,
            verbose=True)
    log = log.getvalue()
    assert f'Estimated rank ({kind}): {want_rank}' in log, log
    stc, _ = apply_dics_csd(csd, filters)
    other_rank = want_rank - 1  # shouldn't make a huge difference
    use_rank = dict(meg=other_rank)
    if not whiten:
        # XXX it's a bug that our rank functions don't treat "meg"
        # properly here...
        use_rank['grad'] = use_rank.pop('meg')
    with catch_logging() as log:
        filters_2 = make_dics(
            epochs.info, fwd_surf, csd, label=label, noise_csd=noise_csd,
            rank=use_rank, verbose=True)
    log = log.getvalue()
    assert f'Computing rank from covariance with rank={use_rank}' in log, log
    stc_2, _ = apply_dics_csd(csd, filters_2)
    corr = np.corrcoef(stc_2.data.ravel(), stc.data.ravel())[0, 1]
    assert 0.8 < corr < 0.999999

    # degenerate conditions
    if whiten:
        # make rank deficient
        data = noise_csd.get_data(0.)
        data[0] = data[:0] = 0
        noise_csd._data[:, 0] = _sym_mat_to_vector(data)
        with pytest.raises(ValueError, match='meg data rank.*the noise rank'):
            filters = make_dics(
                epochs.info, fwd_surf, csd, label=label, noise_csd=noise_csd,
                verbose=True)
