from copy import deepcopy
from inspect import signature
import os.path as op

import pytest
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_array_less)

import mne
from mne.transforms import apply_trans, invert_transform
from mne import (convert_forward_solution, read_forward_solution, compute_rank,
                 VolVectorSourceEstimate, VolSourceEstimate, EvokedArray,
                 pick_channels_cov, read_vectorview_selection)
from mne.beamformer import (make_lcmv, apply_lcmv, apply_lcmv_epochs,
                            apply_lcmv_raw, Beamformer,
                            read_beamformer, apply_lcmv_cov, make_dics)
from mne.beamformer._compute_beamformer import _prepare_beamformer_input
from mne.datasets import testing
from mne.io.compensator import set_current_comp
from mne.io.constants import FIFF
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.minimum_norm.tests.test_inverse import _assert_free_ori_match
from mne.simulation import simulate_evoked
from mne.utils import (object_diff, requires_version, catch_logging,
                       _record_warnings)


data_path = testing.data_path(download=False)
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_trunc_raw-eve.fif')
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', 'Aud-lh.label')
ctf_fname = op.join(data_path, 'CTF', 'somMDYO-18av.ds')

reject = dict(grad=4000e-13, mag=4e-12)


def _read_forward_solution_meg(*args, **kwargs):
    fwd = read_forward_solution(*args)
    fwd = convert_forward_solution(fwd, **kwargs)
    return mne.pick_types_forward(fwd, meg=True, eeg=False)


def _get_data(tmin=-0.1, tmax=0.15, all_forward=True, epochs=True,
              epochs_preload=True, data_cov=True, proj=True):
    """Read in data used in tests."""
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    forward = mne.read_forward_solution(fname_fwd)
    if all_forward:
        forward_surf_ori = _read_forward_solution_meg(
            fname_fwd, surf_ori=True)
        forward_fixed = _read_forward_solution_meg(
            fname_fwd, force_fixed=True, surf_ori=True, use_cps=False)
        forward_vol = _read_forward_solution_meg(fname_fwd_vol)
    else:
        forward_surf_ori = None
        forward_fixed = None
        forward_vol = None

    event_id, tmin, tmax = 1, tmin, tmax

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bad channels
    # Set up pick list: MEG - bad channels
    left_temporal_channels = read_vectorview_selection('Left-temporal')
    picks = mne.pick_types(raw.info, meg=True,
                           selection=left_temporal_channels)
    picks = picks[::2]  # decimate for speed
    # add a couple channels we will consider bad
    bad_picks = [100, 101]
    bads = [raw.ch_names[pick] for pick in bad_picks]
    assert not any(pick in picks for pick in bad_picks)
    picks = np.concatenate([picks, bad_picks])
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    del picks

    raw.info['bads'] = bads  # add more bads
    if proj:
        raw.info.normalize_proj()  # avoid projection warnings
    else:
        raw.del_proj()

    if epochs:
        # Read epochs
        epochs = mne.Epochs(
            raw, events, event_id, tmin, tmax, proj=True,
            baseline=(None, 0), preload=epochs_preload, reject=reject)
        if epochs_preload:
            epochs.resample(200, npad=0)
        epochs.crop(0, None)
        evoked = epochs.average()
        info = evoked.info
    else:
        epochs = None
        evoked = None
        info = raw.info

    noise_cov = mne.read_cov(fname_cov)
    noise_cov['projs'] = []  # avoid warning
    noise_cov = mne.cov.regularize(noise_cov, info, mag=0.05, grad=0.05,
                                   eeg=0.1, proj=True, rank=None)
    if data_cov:
        data_cov = mne.compute_covariance(
            epochs, tmin=0.04, tmax=0.145, verbose='error')  # baseline warning
    else:
        data_cov = None

    return raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol


@pytest.mark.slowtest
@testing.requires_testing_data
def test_lcmv_vector():
    """Test vector LCMV solutions."""
    info = mne.io.read_raw_fif(fname_raw).info

    # For speed and for rank-deficiency calculation simplicity,
    # just use grads
    info = mne.pick_info(info, mne.pick_types(info, meg='grad', exclude=()))
    with info._unlock():
        info.update(bads=[], projs=[])

    forward = mne.read_forward_solution(fname_fwd)
    forward = mne.pick_channels_forward(forward, info['ch_names'])
    vertices = [s['vertno'][::200] for s in forward['src']]
    n_vertices = sum(len(v) for v in vertices)
    assert n_vertices == 4

    amplitude = 100e-9
    stc = mne.SourceEstimate(amplitude * np.eye(n_vertices), vertices,
                             0, 1. / info['sfreq'])
    forward_sim = mne.convert_forward_solution(forward, force_fixed=True,
                                               use_cps=True, copy=True)
    forward_sim = mne.forward.restrict_forward_to_stc(forward_sim, stc)
    noise_cov = mne.make_ad_hoc_cov(info)
    noise_cov.update(data=np.diag(noise_cov['data']), diag=False)
    evoked = simulate_evoked(forward_sim, stc, info, noise_cov, nave=1)
    source_nn = forward_sim['source_nn']
    source_rr = forward_sim['source_rr']

    # Figure out our indices
    mask = np.concatenate([np.in1d(s['vertno'], v)
                           for s, v in zip(forward['src'], vertices)])
    mapping = np.where(mask)[0]
    assert_array_equal(source_rr, forward['source_rr'][mapping])

    # Don't check NN because we didn't rotate to surf ori
    del forward_sim

    # Let's do minimum norm as a sanity check (dipole_fit is slower)
    inv = make_inverse_operator(info, forward, noise_cov, loose=1.)
    stc_vector_mne = apply_inverse(evoked, inv, pick_ori='vector')
    mne_ori = stc_vector_mne.data[mapping, :, np.arange(n_vertices)]
    mne_ori /= np.linalg.norm(mne_ori, axis=-1)[:, np.newaxis]
    mne_angles = np.rad2deg(np.arccos(np.sum(mne_ori * source_nn, axis=-1)))
    assert np.mean(mne_angles) < 35

    # Now let's do LCMV
    data_cov = mne.make_ad_hoc_cov(info)  # just a stub for later
    with pytest.raises(ValueError, match="pick_ori"):
        make_lcmv(info, forward, data_cov, 0.05, noise_cov, pick_ori='bad')

    lcmv_ori = list()
    for ti in range(n_vertices):
        this_evoked = evoked.copy().crop(evoked.times[ti], evoked.times[ti])
        data_cov['diag'] = False
        data_cov['data'] = (np.outer(this_evoked.data, this_evoked.data) +
                            noise_cov['data'])
        vals = linalg.svdvals(data_cov['data'])
        assert vals[0] / vals[-1] < 1e5  # not rank deficient

        with catch_logging() as log:
            filters = make_lcmv(info, forward, data_cov, 0.05, noise_cov,
                                verbose=True)
        log = log.getvalue()
        assert '498 sources' in log
        with catch_logging() as log:
            filters_vector = make_lcmv(info, forward, data_cov, 0.05,
                                       noise_cov, pick_ori='vector',
                                       verbose=True)
        log = log.getvalue()
        assert '498 sources' in log
        stc = apply_lcmv(this_evoked, filters)
        stc_vector = apply_lcmv(this_evoked, filters_vector)
        assert isinstance(stc, mne.SourceEstimate)
        assert isinstance(stc_vector, mne.VectorSourceEstimate)
        assert_allclose(stc.data, stc_vector.magnitude().data)

        # Check the orientation by pooling across some neighbors, as LCMV can
        # have some "holes" at the points of interest
        idx = np.where(cdist(forward['source_rr'], source_rr[[ti]]) < 0.02)[0]
        lcmv_ori.append(np.mean(stc_vector.data[idx, :, 0], axis=0))
        lcmv_ori[-1] /= np.linalg.norm(lcmv_ori[-1])

    lcmv_angles = np.rad2deg(np.arccos(np.sum(lcmv_ori * source_nn, axis=-1)))
    assert np.mean(lcmv_angles) < 55


@pytest.mark.slowtest
@requires_version('h5io')
@testing.requires_testing_data
@pytest.mark.parametrize('reg, proj, kind', [
    (0.01, True, 'volume'),
    (0., False, 'volume'),
    (0.01, False, 'surface'),
    (0., True, 'surface'),
])
def test_make_lcmv_bem(tmp_path, reg, proj, kind):
    """Test LCMV with evoked data and single trials."""
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data(proj=proj)

    if kind == 'surface':
        fwd = forward
    else:
        fwd = forward_vol
        assert kind == 'volume'

    filters = make_lcmv(evoked.info, fwd, data_cov, reg=reg,
                        noise_cov=noise_cov)
    stc = apply_lcmv(evoked, filters)
    stc.crop(0.02, None)

    stc_pow = np.sum(np.abs(stc.data), axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert 0.08 < tmax < 0.15, tmax
    assert 0.9 < np.max(max_stc) < 3.5, np.max(max_stc)

    if kind == 'surface':
        # Test picking normal orientation (surface source space only).
        filters = make_lcmv(evoked.info, forward_surf_ori, data_cov,
                            reg=reg, noise_cov=noise_cov,
                            pick_ori='normal', weight_norm=None)
        stc_normal = apply_lcmv(evoked, filters)
        stc_normal.crop(0.02, None)

        stc_pow = np.sum(np.abs(stc_normal.data), axis=1)
        idx = np.argmax(stc_pow)
        max_stc = stc_normal.data[idx]
        tmax = stc_normal.times[np.argmax(max_stc)]

        lower = 0.04 if proj else 0.025
        assert lower < tmax < 0.14, tmax
        lower = 3e-7 if proj else 2e-7
        assert lower < np.max(max_stc) < 3e-6, np.max(max_stc)

        # No weight normalization was applied, so the amplitude of normal
        # orientation results should always be smaller than free
        # orientation results.
        assert (np.abs(stc_normal.data) <= stc.data).all()

    # Test picking source orientation maximizing output source power
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=reg,
                        noise_cov=noise_cov, pick_ori='max-power')
    stc_max_power = apply_lcmv(evoked, filters)
    stc_max_power.crop(0.02, None)
    stc_pow = np.sum(np.abs(stc_max_power.data), axis=1)
    idx = np.argmax(stc_pow)
    max_stc = np.abs(stc_max_power.data[idx])
    tmax = stc.times[np.argmax(max_stc)]

    lower = 0.08 if proj else 0.04
    assert lower < tmax < 0.15, tmax
    assert 0.8 < np.max(max_stc) < 3., np.max(max_stc)

    stc_max_power.data[:, :] = np.abs(stc_max_power.data)

    if kind == 'surface':
        # Maximum output source power orientation results should be
        # similar to free orientation results in areas with channel
        # coverage
        label = mne.read_label(fname_label)
        mean_stc = stc.extract_label_time_course(
            label, fwd['src'], mode='mean')
        mean_stc_max_pow = \
            stc_max_power.extract_label_time_course(
                label, fwd['src'], mode='mean')
        assert_array_less(np.abs(mean_stc - mean_stc_max_pow), 1.0)

    # Test if spatial filter contains src_type
    assert filters['src_type'] == kind

    # __repr__
    assert len(evoked.ch_names) == 22
    assert len(evoked.info['projs']) == (3 if proj else 0)
    assert len(evoked.info['bads']) == 2
    rank = 17 if proj else 20
    assert 'LCMV' in repr(filters)
    assert 'unknown subject' not in repr(filters)
    assert f'{fwd["nsource"]} vert' in repr(filters)
    assert '20 ch' in repr(filters)
    assert 'rank %s' % rank in repr(filters)

    # I/O
    fname = op.join(str(tmp_path), 'filters.h5')
    with pytest.warns(RuntimeWarning, match='-lcmv.h5'):
        filters.save(fname)
    filters_read = read_beamformer(fname)
    assert isinstance(filters, Beamformer)
    assert isinstance(filters_read, Beamformer)
    # deal with object_diff strictness
    filters_read['rank'] = int(filters_read['rank'])
    filters['rank'] = int(filters['rank'])
    assert object_diff(filters, filters_read) == ''

    if kind != 'surface':
        return

    # Test if fixed forward operator is detected when picking normal or
    # max-power orientation
    pytest.raises(ValueError, make_lcmv, evoked.info, forward_fixed, data_cov,
                  reg=0.01, noise_cov=noise_cov, pick_ori='normal')
    pytest.raises(ValueError, make_lcmv, evoked.info, forward_fixed, data_cov,
                  reg=0.01, noise_cov=noise_cov, pick_ori='max-power')

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    pytest.raises(ValueError, make_lcmv, evoked.info, forward, data_cov,
                  reg=0.01, noise_cov=noise_cov, pick_ori='normal')

    # Test if volume forward operator is detected when picking normal
    # orientation
    pytest.raises(ValueError, make_lcmv, evoked.info, forward_vol, data_cov,
                  reg=0.01, noise_cov=noise_cov, pick_ori='normal')

    # Test if missing of noise covariance matrix is detected when more than
    # one channel type is present in the data
    pytest.raises(ValueError, make_lcmv, evoked.info, forward_vol,
                  data_cov=data_cov, reg=0.01, noise_cov=None,
                  pick_ori='max-power')

    # Test if wrong channel selection is detected in application of filter
    evoked_ch = deepcopy(evoked)
    evoked_ch.pick_channels(evoked_ch.ch_names[1:])
    filters = make_lcmv(evoked.info, forward_vol, data_cov, reg=0.01,
                        noise_cov=noise_cov)

    # Test if discrepancies in channel selection of data and fwd model are
    # handled correctly in apply_lcmv
    # make filter with data where first channel was removed
    filters = make_lcmv(evoked_ch.info, forward_vol, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    # applying that filter to the full data set should automatically exclude
    # this channel from the data
    # also test here that no warnings are thrown - implemented to check whether
    # src should not be None warning occurs
    stc = apply_lcmv(evoked, filters)

    # the result should be equal to applying this filter to a dataset without
    # this channel:
    stc_ch = apply_lcmv(evoked_ch, filters)
    assert_array_almost_equal(stc.data, stc_ch.data)

    # Test if non-matching SSP projection is detected in application of filter
    if proj:
        raw_proj = raw.copy().del_proj()
        with pytest.raises(ValueError, match='do not match the projections'):
            apply_lcmv_raw(raw_proj, filters)

    # Test apply_lcmv_raw
    use_raw = raw.copy().crop(0, 1)
    stc = apply_lcmv_raw(use_raw, filters)
    assert_allclose(stc.times, use_raw.times)
    assert_array_equal(stc.vertices[0], forward_vol['src'][0]['vertno'])

    # Test if spatial filter contains src_type
    assert 'src_type' in filters

    # check whether a filters object without src_type throws expected warning
    del filters['src_type']  # emulate 0.16 behaviour to cause warning
    with pytest.warns(RuntimeWarning, match='spatial filter does not contain '
                      'src_type'):
        apply_lcmv(evoked, filters)

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    filters = make_lcmv(epochs.info, forward_fixed, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    stcs = apply_lcmv_epochs(epochs, filters)
    stcs_ = apply_lcmv_epochs(epochs, filters, return_generator=True)
    assert_array_equal(stcs[0].data, next(stcs_).data)

    epochs.drop_bad()
    assert (len(epochs.events) == len(stcs))

    # average the single trial estimates
    stc_avg = np.zeros_like(stcs[0].data)
    for this_stc in stcs:
        stc_avg += this_stc.data
    stc_avg /= len(stcs)

    # compare it to the solution using evoked with fixed orientation
    filters = make_lcmv(evoked.info, forward_fixed, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    stc_fixed = apply_lcmv(evoked, filters)
    assert_array_almost_equal(stc_avg, stc_fixed.data)

    # use a label so we have few source vertices and delayed computation is
    # not used
    filters = make_lcmv(epochs.info, forward_fixed, data_cov, reg=0.01,
                        noise_cov=noise_cov, label=label)
    stcs_label = apply_lcmv_epochs(epochs, filters)

    assert_array_almost_equal(stcs_label[0].data, stcs[0].in_label(label).data)

    # Test condition where the filters weights are zero. There should not be
    # any divide-by-zero errors
    zero_cov = data_cov.copy()
    zero_cov['data'][:] = 0
    filters = make_lcmv(epochs.info, forward_fixed, zero_cov, reg=0.01,
                        noise_cov=noise_cov)
    assert_array_equal(filters['weights'], 0)

    # Test condition where one channel type is picked
    # (avoid "grad data rank (13) did not match the noise rank (None)")
    data_cov_grad = pick_channels_cov(
        data_cov, [ch_name for ch_name in epochs.info['ch_names']
                   if ch_name.endswith(('2', '3'))])
    assert len(data_cov_grad['names']) > 4
    make_lcmv(epochs.info, forward_fixed, data_cov_grad, reg=0.01,
              noise_cov=noise_cov)


@testing.requires_testing_data
@pytest.mark.slowtest
@pytest.mark.parametrize('weight_norm, pick_ori', [
    ('unit-noise-gain', 'max-power'),
    ('unit-noise-gain', 'vector'),
    ('unit-noise-gain', None),
    ('nai', 'vector'),
    (None, 'max-power'),
])
def test_make_lcmv_sphere(pick_ori, weight_norm):
    """Test LCMV with sphere head model."""
    # unit-noise gain beamformer and orientation
    # selection and rank reduction of the leadfield
    _, _, evoked, data_cov, noise_cov, _, _, _, _, _ = _get_data(proj=True)
    assert 'eeg' not in evoked
    assert 'meg' in evoked
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)
    src = mne.setup_volume_source_space(
        pos=25., sphere=sphere, mindist=5.0, exclude=2.0)
    fwd_sphere = mne.make_forward_solution(evoked.info, None, src, sphere)

    # Test that we get an error if not reducing rank
    with pytest.raises(ValueError, match='Singular matrix detected'):
        with pytest.warns(RuntimeWarning, match='positive semidefinite'):
            make_lcmv(
                evoked.info, fwd_sphere, data_cov, reg=0.1,
                noise_cov=noise_cov, weight_norm=weight_norm,
                pick_ori=pick_ori, reduce_rank=False, rank='full')

    # Now let's reduce it
    filters = make_lcmv(evoked.info, fwd_sphere, data_cov, reg=0.1,
                        noise_cov=noise_cov, weight_norm=weight_norm,
                        pick_ori=pick_ori, reduce_rank=True)
    stc_sphere = apply_lcmv(evoked, filters)
    if isinstance(stc_sphere, VolVectorSourceEstimate):
        stc_sphere = stc_sphere.magnitude()
    else:
        stc_sphere = abs(stc_sphere)
    assert isinstance(stc_sphere, VolSourceEstimate)
    stc_sphere.crop(0.02, None)

    stc_pow = np.sum(stc_sphere.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc_sphere.data[idx]
    tmax = stc_sphere.times[np.argmax(max_stc)]
    assert 0.08 < tmax < 0.15, tmax
    min_, max_ = 1.0, 4.5
    if weight_norm is None:
        min_ *= 2e-7
        max_ *= 2e-7
    assert min_ < np.max(max_stc) < max_, (min_, np.max(max_stc), max_)


@testing.requires_testing_data
@pytest.mark.parametrize('weight_norm', (None, 'unit-noise-gain'))
@pytest.mark.parametrize('pick_ori', ('max-power', 'normal'))
def test_lcmv_cov(weight_norm, pick_ori):
    """Test LCMV source power computation."""
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()
    convert_forward_solution(forward, surf_ori=True, copy=False)
    filters = make_lcmv(evoked.info, forward, data_cov, noise_cov=noise_cov,
                        weight_norm=weight_norm, pick_ori=pick_ori)
    for cov in (data_cov, noise_cov):
        this_cov = pick_channels_cov(cov, evoked.ch_names)
        this_evoked = evoked.copy().pick_channels(this_cov['names'])
        this_cov['projs'] = this_evoked.info['projs']
        assert this_evoked.ch_names == this_cov['names']
        stc = apply_lcmv_cov(this_cov, filters)
        assert stc.data.min() > 0
        assert stc.shape == (498, 1)
        ev = EvokedArray(this_cov.data, this_evoked.info)
        stc_1 = apply_lcmv(ev, filters)
        assert stc_1.data.min() < 0
        ev = EvokedArray(stc_1.data.T, this_evoked.info)
        stc_2 = apply_lcmv(ev, filters)
        assert stc_2.data.shape == (498, 498)
        data = np.diag(stc_2.data)[:, np.newaxis]
        assert data.min() > 0
        assert_allclose(data, stc.data, rtol=1e-12)


@pytest.mark.slowtest
@testing.requires_testing_data
def test_lcmv_ctf_comp():
    """Test interpolation with compensated CTF data."""
    raw = mne.io.read_raw_ctf(ctf_fname, preload=True)
    raw.pick(raw.ch_names[:70])

    events = mne.make_fixed_length_events(raw, duration=0.2)[:2]
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.2)
    evoked = epochs.average()

    data_cov = mne.compute_covariance(epochs)
    fwd = mne.make_forward_solution(evoked.info, None,
                                    mne.setup_volume_source_space(pos=30.0),
                                    mne.make_sphere_model())
    with pytest.raises(ValueError, match='reduce_rank'):
        make_lcmv(evoked.info, fwd, data_cov)
    filters = make_lcmv(evoked.info, fwd, data_cov, reduce_rank=True)
    assert 'weights' in filters

    # test whether different compensations throw error
    info_comp = evoked.info.copy()
    set_current_comp(info_comp, 1)
    with pytest.raises(RuntimeError, match='Compensation grade .* not match'):
        make_lcmv(info_comp, fwd, data_cov)


@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize('proj, weight_norm', [
    (True, 'unit-noise-gain'),
    (False, 'unit-noise-gain'),
    (True, None),
    (True, 'nai'),
])
def test_lcmv_reg_proj(proj, weight_norm):
    """Test LCMV with and without proj."""
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    events = mne.find_events(raw)
    raw.pick_types(meg=True)
    assert len(raw.ch_names) == 305
    epochs = mne.Epochs(raw, events, None, preload=True, proj=proj)
    with pytest.warns(RuntimeWarning, match='Too few samples'):
        noise_cov = mne.compute_covariance(epochs, tmax=0)
        data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15)
    forward = mne.read_forward_solution(fname_fwd)
    filters = make_lcmv(epochs.info, forward, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='nai', rank=None, verbose=True)
    want_rank = 302  # 305 good channels - 3 MEG projs
    assert filters['rank'] == want_rank
    # And also with and without noise_cov
    with pytest.raises(ValueError, match='several sensor types'):
        make_lcmv(epochs.info, forward, data_cov, reg=0.05,
                  noise_cov=None)
    epochs.pick_types(meg='grad')
    kwargs = dict(reg=0.05, pick_ori=None, weight_norm=weight_norm)
    filters_cov = make_lcmv(epochs.info, forward, data_cov,
                            noise_cov=noise_cov, **kwargs)
    filters_nocov = make_lcmv(epochs.info, forward, data_cov,
                              noise_cov=None, **kwargs)
    ad_hoc = mne.make_ad_hoc_cov(epochs.info)
    filters_adhoc = make_lcmv(epochs.info, forward, data_cov,
                              noise_cov=ad_hoc, **kwargs)
    evoked = epochs.average()
    stc_cov = apply_lcmv(evoked, filters_cov)
    stc_nocov = apply_lcmv(evoked, filters_nocov)
    stc_adhoc = apply_lcmv(evoked, filters_adhoc)

    # Compare adhoc and nocov: scale difference is necessitated by using std=1.
    if weight_norm == 'unit-noise-gain':
        scale = np.sqrt(ad_hoc['data'][0])
    else:
        scale = 1.
    assert_allclose(stc_nocov.data, stc_adhoc.data * scale)
    a = np.dot(filters_nocov['weights'], filters_nocov['whitener'])
    b = np.dot(filters_adhoc['weights'], filters_adhoc['whitener']) * scale
    atol = np.mean(np.sqrt(a * a)) * 1e-7
    assert_allclose(a, b, atol=atol, rtol=1e-7)

    # Compare adhoc and cov: locs might not be equivalent, but the same
    # general profile should persist, so look at the std and be lenient:
    if weight_norm == 'unit-noise-gain':
        adhoc_scale = 0.12
    else:
        adhoc_scale = 1.
    assert_allclose(
        np.linalg.norm(stc_adhoc.data, axis=0) * adhoc_scale,
        np.linalg.norm(stc_cov.data, axis=0), rtol=0.3)
    assert_allclose(
        np.linalg.norm(stc_nocov.data, axis=0) / scale * adhoc_scale,
        np.linalg.norm(stc_cov.data, axis=0), rtol=0.3)

    if weight_norm == 'nai':
        # NAI is always normalized by noise-level (based on eigenvalues)
        for stc in (stc_nocov, stc_cov):
            assert_allclose(stc.data.std(), 0.584, rtol=0.2)
    elif weight_norm is None:
        # None always represents something not normalized, reflecting channel
        # weights
        for stc in (stc_nocov, stc_cov):
            assert_allclose(stc.data.std(), 2.8e-8, rtol=0.1)
    else:
        assert weight_norm == 'unit-noise-gain'
        # Channel scalings depend on presence of noise_cov
        assert_allclose(stc_nocov.data.std(), 7.8e-13, rtol=0.1)
        assert_allclose(stc_cov.data.std(), 0.187, rtol=0.2)


@pytest.mark.parametrize('reg, weight_norm, use_cov, depth, lower, upper', [
    (0.05, 'unit-noise-gain', True, None, 97, 98),
    (0.05, 'nai', True, None, 96, 98),
    (0.05, 'nai', True, 0.8, 96, 98),
    (0.05, None, True, None, 74, 76),
    (0.05, None, True, 0.8, 90, 93),  # depth improves weight_norm=None
    (0.05, 'unit-noise-gain', False, None, 83, 86),
    (0.05, 'unit-noise-gain', False, 0.8, 83, 86),  # depth same for wn != None
    # no reg
    (0.00, 'unit-noise-gain', True, None, 35, 99),  # TODO: Still not stable
])
def test_localization_bias_fixed(bias_params_fixed, reg, weight_norm, use_cov,
                                 depth, lower, upper):
    """Test localization bias for fixed-orientation LCMV."""
    evoked, fwd, noise_cov, data_cov, want = bias_params_fixed
    if not use_cov:
        evoked.pick_types(meg='grad')
        noise_cov = None
    assert data_cov['data'].shape[0] == len(data_cov['names'])
    loc = apply_lcmv(evoked, make_lcmv(evoked.info, fwd, data_cov, reg,
                                       noise_cov, depth=depth,
                                       weight_norm=weight_norm)).data
    loc = np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper


# Changes here should be synced with test_dics.py
@pytest.mark.parametrize(
    'reg, pick_ori, weight_norm, use_cov, depth, lower, upper, '
    'lower_ori, upper_ori', [
        (0.05, 'vector', 'unit-noise-gain-invariant', False, None, 26, 28, 0.82, 0.84),  # noqa: E501
        (0.05, 'vector', 'unit-noise-gain-invariant', True, None, 40, 42, 0.96, 0.98),  # noqa: E501
        (0.05, 'vector', 'unit-noise-gain', False, None, 13, 14, 0.79, 0.81),
        (0.05, 'vector', 'unit-noise-gain', True, None, 35, 37, 0.98, 0.99),
        (0.05, 'vector', 'nai', True, None, 35, 37, 0.98, 0.99),
        (0.05, 'vector', None, True, None, 12, 14, 0.97, 0.98),
        (0.05, 'vector', None, True, 0.8, 39, 43, 0.97, 0.98),
        (0.05, 'max-power', 'unit-noise-gain-invariant', False, None, 17, 20, 0, 0),  # noqa: E501
        (0.05, 'max-power', 'unit-noise-gain', False, None, 17, 20, 0, 0),
        (0.05, 'max-power', 'nai', True, None, 21, 24, 0, 0),
        (0.05, 'max-power', None, True, None, 7, 10, 0, 0),
        (0.05, 'max-power', None, True, 0.8, 15, 18, 0, 0),
        (0.05, None, None, True, 0.8, 40, 42, 0, 0),
        # no reg
        (0.00, 'vector', None, True, None, 23, 24, 0.96, 0.97),
        (0.00, 'vector', 'unit-noise-gain-invariant', True, None, 52, 54, 0.95, 0.96),  # noqa: E501
        (0.00, 'vector', 'unit-noise-gain', True, None, 44, 48, 0.97, 0.99),
        (0.00, 'vector', 'nai', True, None, 44, 48, 0.97, 0.99),
        (0.00, 'max-power', None, True, None, 14, 15, 0, 0),
        (0.00, 'max-power', 'unit-noise-gain-invariant', True, None, 35, 37, 0, 0),  # noqa: E501
        (0.00, 'max-power', 'unit-noise-gain', True, None, 35, 37, 0, 0),
        (0.00, 'max-power', 'nai', True, None, 35, 37, 0, 0),
    ])
def test_localization_bias_free(bias_params_free, reg, pick_ori, weight_norm,
                                use_cov, depth, lower, upper,
                                lower_ori, upper_ori):
    """Test localization bias for free-orientation LCMV."""
    evoked, fwd, noise_cov, data_cov, want = bias_params_free
    if not use_cov:
        evoked.pick_types(meg='grad')
        noise_cov = None
    with _record_warnings():  # rank deficiency of data_cov
        filters = make_lcmv(evoked.info, fwd, data_cov, reg,
                            noise_cov, pick_ori=pick_ori,
                            weight_norm=weight_norm,
                            depth=depth)
    loc = apply_lcmv(evoked, filters).data
    if pick_ori == 'vector':
        ori = loc.copy() / np.linalg.norm(loc, axis=1, keepdims=True)
    else:
        # doesn't make sense for pooled (None) or max-power (can't be all 3)
        ori = None
    loc = np.linalg.norm(loc, axis=1) if pick_ori == 'vector' else np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    max_idx = np.argmax(loc, axis=0)
    perc = (want == max_idx).mean() * 100
    assert lower <= perc <= upper
    _assert_free_ori_match(ori, max_idx, lower_ori, upper_ori)


# Changes here should be synced with the ones above, but these have meaningful
# orientation values
@pytest.mark.parametrize(
    'reg, weight_norm, use_cov, depth, lower, upper, lower_ori, upper_ori', [
        (0.05, 'unit-noise-gain-invariant', False, None, 38, 40, 0.54, 0.55),
        (0.05, 'unit-noise-gain', False, None, 38, 40, 0.54, 0.55),
        (0.05, 'nai', True, None, 56, 57, 0.59, 0.61),
        (0.05, None, True, None, 27, 28, 0.56, 0.57),
        (0.05, None, True, 0.8, 42, 43, 0.56, 0.57),
        # no reg
        (0.00, None, True, None, 50, 51, 0.58, 0.59),
        (0.00, 'unit-noise-gain-invariant', True, None, 73, 75, 0.59, 0.61),
        (0.00, 'unit-noise-gain', True, None, 73, 75, 0.59, 0.61),
        (0.00, 'nai', True, None, 73, 75, 0.59, 0.61),
    ])
def test_orientation_max_power(bias_params_fixed, bias_params_free,
                               reg, weight_norm, use_cov, depth, lower, upper,
                               lower_ori, upper_ori):
    """Test orientation selection for bias for max-power LCMV."""
    # we simulate data for the fixed orientation forward and beamform using
    # the free orientation forward, and check the orientation match at the end
    evoked, _, noise_cov, data_cov, want = bias_params_fixed
    fwd = bias_params_free[1]
    if not use_cov:
        evoked.pick_types(meg='grad')
        noise_cov = None
    filters = make_lcmv(evoked.info, fwd, data_cov, reg,
                        noise_cov, pick_ori='max-power',
                        weight_norm=weight_norm,
                        depth=depth)
    loc = apply_lcmv(evoked, filters).data
    ori = filters['max_power_ori']
    assert ori.shape == (246, 3)
    loc = np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    max_idx = np.argmax(loc, axis=0)
    mask = want == max_idx  # ones that localized properly
    perc = mask.mean() * 100
    assert lower <= perc <= upper
    # Compute the dot products of our forward normals and
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


@pytest.mark.parametrize('weight_norm, pick_ori', [
    pytest.param('nai', 'max-power', marks=pytest.mark.slowtest),
    ('unit-noise-gain', 'vector'),
    ('unit-noise-gain', 'max-power'),
    pytest.param('unit-noise-gain', None, marks=pytest.mark.slowtest),
])
def test_depth_does_not_matter(bias_params_free, weight_norm, pick_ori):
    """Test that depth weighting does not matter for normalized filters."""
    evoked, fwd, noise_cov, data_cov, _ = bias_params_free
    data = apply_lcmv(evoked, make_lcmv(
        evoked.info, fwd, data_cov, 0.05, noise_cov, pick_ori=pick_ori,
        weight_norm=weight_norm, depth=0.)).data
    data_depth = apply_lcmv(evoked, make_lcmv(
        evoked.info, fwd, data_cov, 0.05, noise_cov, pick_ori=pick_ori,
        weight_norm=weight_norm, depth=1.)).data
    assert data.shape == data_depth.shape
    for d1, d2 in zip(data, data_depth):
        # Sign flips can change when nearly orthogonal to the normal direction
        d2 *= np.sign(np.dot(d1.ravel(), d2.ravel()))
        atol = np.linalg.norm(d1) * 1e-7
        assert_allclose(d1, d2, atol=atol)


@testing.requires_testing_data
def test_lcmv_maxfiltered():
    """Test LCMV on maxfiltered data."""
    raw = mne.io.read_raw_fif(fname_raw).fix_mag_coil_types()
    raw_sss = mne.preprocessing.maxwell_filter(raw)
    events = mne.find_events(raw_sss)
    del raw
    raw_sss.pick_types(meg='mag')
    assert len(raw_sss.ch_names) == 102
    epochs = mne.Epochs(raw_sss, events)
    data_cov = mne.compute_covariance(epochs, tmin=0)
    fwd = mne.read_forward_solution(fname_fwd)
    rank = compute_rank(data_cov, info=epochs.info)
    assert rank == {'mag': 71}
    for use_rank in ('info', rank, 'full', None):
        make_lcmv(epochs.info, fwd, data_cov, rank=use_rank)


# To reduce test time, only test combinations that should matter rather than
# all of them
@testing.requires_testing_data
@pytest.mark.parametrize('pick_ori, weight_norm, reg, inversion', [
    ('vector', 'unit-noise-gain-invariant', 0.05, 'matrix'),
    ('vector', 'unit-noise-gain-invariant', 0.05, 'single'),
    ('vector', 'unit-noise-gain', 0.05, 'matrix'),
    ('vector', 'unit-noise-gain', 0.05, 'single'),
    ('vector', 'unit-noise-gain', 0.0, 'matrix'),
    ('vector', 'unit-noise-gain', 0.0, 'single'),
    ('vector', 'nai', 0.05, 'matrix'),
    ('max-power', 'unit-noise-gain', 0.05, 'matrix'),
    ('max-power', 'unit-noise-gain', 0.0, 'single'),
    ('max-power', 'unit-noise-gain', 0.05, 'single'),
    ('max-power', 'unit-noise-gain-invariant', 0.05, 'matrix'),
    ('normal', 'unit-noise-gain', 0.05, 'matrix'),
    ('normal', 'nai', 0.0, 'matrix'),
])
def test_unit_noise_gain_formula(pick_ori, weight_norm, reg, inversion):
    """Test unit-noise-gain filter against formula."""
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    events = mne.find_events(raw)
    raw.pick_types(meg='mag')
    assert len(raw.ch_names) == 102
    epochs = mne.Epochs(raw, events, None, preload=True)
    data_cov = mne.compute_covariance(epochs, tmin=0.04, tmax=0.15)
    # for now, avoid whitening to make life easier
    noise_cov = mne.make_ad_hoc_cov(epochs.info, std=dict(grad=1., mag=1.))
    forward = mne.read_forward_solution(fname_fwd)
    convert_forward_solution(forward, surf_ori=True, copy=False)
    rank = None
    kwargs = dict(reg=reg, noise_cov=noise_cov, pick_ori=pick_ori,
                  weight_norm=weight_norm, rank=rank, inversion=inversion)
    if inversion == 'single' and pick_ori == 'vector' and \
            weight_norm == 'unit-noise-gain-invariant':
        with pytest.raises(ValueError, match='Cannot use'):
            make_lcmv(epochs.info, forward, data_cov, **kwargs)
        return
    filters = make_lcmv(epochs.info, forward, data_cov, **kwargs)
    _, _, _, _, G, _, _, _ = _prepare_beamformer_input(
        epochs.info, forward, None, 'vector', noise_cov=noise_cov, rank=rank,
        pca=False, exp=None)
    n_channels, n_sources = G.shape
    n_sources //= 3
    G.shape = (n_channels, n_sources, 3)
    G = G.transpose(1, 2, 0)  # verts, orient, ch
    _assert_weight_norm(filters, G)


def _assert_weight_norm(filters, G):
    """Check the result of the chosen weight normalization strategy."""
    weights, max_power_ori = filters['weights'], filters['max_power_ori']

    # Make the dimensions of the weight matrix equal for both DICS (which
    # defines weights for multiple frequencies) and LCMV (which does not).
    if filters['kind'] == 'LCMV':
        weights = weights[np.newaxis]
        if max_power_ori is not None:
            max_power_ori = max_power_ori[np.newaxis]
    if max_power_ori is not None:
        max_power_ori = max_power_ori[..., np.newaxis]

    weight_norm = filters['weight_norm']
    inversion = filters['inversion']
    n_channels = weights.shape[2]

    if inversion == 'matrix':
        # Dipoles are grouped in groups with size n_orient
        n_sources = filters['n_sources']
        n_orient = 3 if filters['is_free_ori'] else 1
    elif inversion == 'single':
        # Every dipole is treated as a unique source
        n_sources = weights.shape[1]
        n_orient = 1

    for wi, w in enumerate(weights):
        w = w.reshape(n_sources, n_orient, n_channels)

        # Compute leadfield in the direction chosen during the computation of
        # the beamformer.
        if filters['pick_ori'] == 'max-power':
            use_G = np.sum(G * max_power_ori[wi], axis=1, keepdims=True)
        elif filters['pick_ori'] == 'normal':
            use_G = G[:, -1:]
        else:
            use_G = G
        if inversion == 'single':
            # Every dipole is treated as a unique source
            use_G = use_G.reshape(n_sources, 1, n_channels)
        assert w.shape == use_G.shape == (n_sources, n_orient, n_channels)

        # Test weight normalization scheme
        got = np.matmul(w, w.conj().swapaxes(-2, -1))
        desired = np.repeat(np.eye(n_orient)[np.newaxis], w.shape[0], axis=0)
        if n_orient == 3 and weight_norm in ('unit-noise-gain', 'nai'):
            # only the diagonal is correct!
            assert not np.allclose(got, desired, atol=1e-7)
            got = got.reshape(n_sources, -1)[:, ::n_orient + 1]
            desired = np.ones_like(got)
        if weight_norm == 'nai':  # additional scale factor, should be fixed
            atol = 1e-7 * got.flat[0]
            desired *= got.flat[0]
        else:
            atol = 1e-7
        assert_allclose(got, desired, atol=atol, err_msg='w @ w.conj().T = I')

        # Check that the result here is a diagonal matrix for Sekihara
        if n_orient > 1 and weight_norm != 'unit-noise-gain-invariant':
            got = w @ use_G.swapaxes(-2, -1)
            diags = np.diagonal(got, 0, -2, -1)
            want = np.apply_along_axis(np.diagflat, 1, diags)
            atol = np.mean(diags).real * 1e-12
            assert_allclose(got, want, atol=atol, err_msg='G.T @ w = θI')


def test_api():
    """Test LCMV/DICS API equivalence."""
    lcmv_names = list(signature(make_lcmv).parameters)
    dics_names = list(signature(make_dics).parameters)
    dics_names[dics_names.index('csd')] = 'data_cov'
    dics_names[dics_names.index('noise_csd')] = 'noise_cov'
    dics_names.pop(dics_names.index('real_filter'))  # not a thing for LCMV
    assert lcmv_names == dics_names
