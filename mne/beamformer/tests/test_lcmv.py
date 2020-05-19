from copy import deepcopy
import os.path as op

import pytest
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal, assert_allclose,
                           assert_array_less)

import mne
from mne import (convert_forward_solution, read_forward_solution, compute_rank,
                 VolVectorSourceEstimate, VolSourceEstimate)
from mne.datasets import testing
from mne.beamformer import (make_lcmv, apply_lcmv, apply_lcmv_epochs,
                            apply_lcmv_raw, tf_lcmv, Beamformer,
                            read_beamformer)
from mne.beamformer._lcmv import _lcmv_source_power
from mne.io.compensator import set_current_comp
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.simulation import simulate_evoked
from mne.utils import (run_tests_if_main, object_diff, requires_h5py,
                       catch_logging)


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
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, selection=left_temporal_channels)
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
        with pytest.warns(RuntimeWarning, match='baseline = None'):
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


@testing.requires_testing_data
def test_lcmv_vector():
    """Test vector LCMV solutions."""
    info = mne.io.read_raw_fif(fname_raw).info

    # For speed and for rank-deficiency calculation simplicity,
    # just use grads
    info = mne.pick_info(info, mne.pick_types(info, meg='grad', exclude=()))
    info.update(bads=[], projs=[])

    forward = mne.read_forward_solution(fname_fwd)
    forward = mne.pick_channels_forward(forward, info['ch_names'])
    vertices = [s['vertno'][::100] for s in forward['src']]
    n_vertices = sum(len(v) for v in vertices)
    assert 5 < n_vertices < 20

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
@requires_h5py
@testing.requires_testing_data
@pytest.mark.parametrize('reg', (0.01, 0.))
@pytest.mark.parametrize('proj', (True, False))
def test_make_lcmv(tmpdir, reg, proj):
    """Test LCMV with evoked data and single trials."""
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data(proj=proj)

    for fwd in [forward, forward_vol]:
        filters = make_lcmv(evoked.info, fwd, data_cov, reg=reg,
                            noise_cov=noise_cov)
        stc = apply_lcmv(evoked, filters, max_ori_out='signed')
        stc.crop(0.02, None)

        stc_pow = np.sum(np.abs(stc.data), axis=1)
        idx = np.argmax(stc_pow)
        max_stc = stc.data[idx]
        tmax = stc.times[np.argmax(max_stc)]

        assert 0.08 < tmax < 0.14, tmax
        assert 0.9 < np.max(max_stc) < 3., np.max(max_stc)

        if fwd is forward:
            # Test picking normal orientation (surface source space only).
            filters = make_lcmv(evoked.info, forward_surf_ori, data_cov,
                                reg=reg, noise_cov=noise_cov,
                                pick_ori='normal', weight_norm=None)
            stc_normal = apply_lcmv(evoked, filters, max_ori_out='signed')
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
        stc_max_power = apply_lcmv(evoked, filters, max_ori_out='signed')
        stc_max_power.crop(0.02, None)
        stc_pow = np.sum(np.abs(stc_max_power.data), axis=1)
        idx = np.argmax(stc_pow)
        max_stc = np.abs(stc_max_power.data[idx])
        tmax = stc.times[np.argmax(max_stc)]

        lower = 0.08 if proj else 0.04
        assert lower < tmax < 0.12, tmax
        assert 0.8 < np.max(max_stc) < 3., np.max(max_stc)

        stc_max_power.data[:, :] = np.abs(stc_max_power.data)

        if fwd is forward:
            # Maximum output source power orientation results should be
            # similar to free orientation results in areas with channel
            # coverage
            label = mne.read_label(fname_label)
            mean_stc = stc.extract_label_time_course(label, fwd['src'],
                                                     mode='mean')
            mean_stc_max_pow = \
                stc_max_power.extract_label_time_course(label, fwd['src'],
                                                        mode='mean')
            assert_array_less(np.abs(mean_stc - mean_stc_max_pow), 1.0)

        # Test NAI weight normalization:
        filters = make_lcmv(evoked.info, fwd, data_cov, reg=reg,
                            noise_cov=noise_cov, pick_ori='max-power',
                            weight_norm='nai')
        stc_nai = apply_lcmv(evoked, filters, max_ori_out='signed')
        stc_nai.crop(0.02, None)

        # Test whether unit-noise-gain solution is a scaled version of NAI
        pearsoncorr = np.corrcoef(np.concatenate(np.abs(stc_nai.data)),
                                  np.concatenate(stc_max_power.data))
        assert_almost_equal(pearsoncorr[0, 1], 1.)

    # Test if spatial filter contains src_type
    assert 'src_type' in filters

    # __repr__
    assert len(evoked.ch_names) == 22
    assert len(evoked.info['projs']) == (4 if proj else 0)
    assert len(evoked.info['bads']) == 2
    rank = 17 if proj else 20
    assert 'LCMV' in repr(filters)
    assert 'unknown subject' not in repr(filters)
    assert '4157 vert' in repr(filters)
    assert '20 ch' in repr(filters)
    assert 'rank %s' % rank in repr(filters)

    # I/O
    fname = op.join(str(tmpdir), 'filters.h5')
    with pytest.warns(RuntimeWarning, match='-lcmv.h5'):
        filters.save(fname)
    filters_read = read_beamformer(fname)
    assert isinstance(filters, Beamformer)
    assert isinstance(filters_read, Beamformer)
    # deal with object_diff strictness
    filters_read['rank'] = int(filters_read['rank'])
    filters['rank'] = int(filters['rank'])
    assert object_diff(filters, filters_read) == ''

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
    pytest.raises(ValueError, apply_lcmv, evoked_ch, filters,
                  max_ori_out='signed')

    # Test if discrepancies in channel selection of data and fwd model are
    # handled correctly in apply_lcmv
    # make filter with data where first channel was removed
    filters = make_lcmv(evoked_ch.info, forward_vol, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    # applying that filter to the full data set should automatically exclude
    # this channel from the data
    # also test here that no warnings are thrown - implemented to check whether
    # src should not be None warning occurs
    with pytest.warns(None) as w:
        stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    assert len(w) == 0
    # the result should be equal to applying this filter to a dataset without
    # this channel:
    stc_ch = apply_lcmv(evoked_ch, filters, max_ori_out='signed')
    assert_array_almost_equal(stc.data, stc_ch.data)

    # Test if non-matching SSP projection is detected in application of filter
    if proj:
        raw_proj = deepcopy(raw)
        raw_proj.del_proj()
        with pytest.raises(ValueError, match='do not match the projections'):
            apply_lcmv_raw(raw_proj, filters, max_ori_out='signed')

    # Test if spatial filter contains src_type
    assert 'src_type' in filters

    # check whether a filters object without src_type throws expected warning
    del filters['src_type']  # emulate 0.16 behaviour to cause warning
    with pytest.warns(RuntimeWarning, match='spatial filter does not contain '
                      'src_type'):
        apply_lcmv(evoked, filters, max_ori_out='signed')

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    filters = make_lcmv(epochs.info, forward_fixed, data_cov, reg=0.01,
                        noise_cov=noise_cov)
    stcs = apply_lcmv_epochs(epochs, filters, max_ori_out='signed')
    stcs_ = apply_lcmv_epochs(epochs, filters, return_generator=True,
                              max_ori_out='signed')
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
    stc_fixed = apply_lcmv(evoked, filters, max_ori_out='signed')
    assert_array_almost_equal(stc_avg, stc_fixed.data)

    # use a label so we have few source vertices and delayed computation is
    # not used
    filters = make_lcmv(epochs.info, forward_fixed, data_cov, reg=0.01,
                        noise_cov=noise_cov, label=label)
    stcs_label = apply_lcmv_epochs(epochs, filters, max_ori_out='signed')

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
    data_cov_grad = mne.pick_channels_cov(
        data_cov, [ch_name for ch_name in epochs.info['ch_names']
                   if ch_name.endswith(('2', '3'))])
    assert len(data_cov_grad['names']) > 4
    make_lcmv(epochs.info, forward_fixed, data_cov_grad, reg=0.01,
              noise_cov=noise_cov)


@pytest.mark.parametrize('weight_norm', (None, 'unit-noise-gain', 'nai'))
@pytest.mark.parametrize('pick_ori', (None, 'max-power', 'vector'))
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
        make_lcmv(
            evoked.info, fwd_sphere, data_cov, reg=0.1,
            noise_cov=noise_cov, weight_norm=weight_norm,
            pick_ori=pick_ori, reduce_rank=False, rank='full')

    # Now let's reduce it
    filters = make_lcmv(evoked.info, fwd_sphere, data_cov, reg=0.1,
                        noise_cov=noise_cov, weight_norm=weight_norm,
                        pick_ori=pick_ori, reduce_rank=True)
    stc_sphere = apply_lcmv(evoked, filters, max_ori_out='signed')
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
    min_, max_ = 0.4, 3.0
    if weight_norm is None:
        min_ *= 2e-7
        max_ *= 2e-7
    assert min_ < np.max(max_stc) < max_, (min_, np.max(max_stc), max_)


@testing.requires_testing_data
def test_lcmv_raw():
    """Test LCMV with raw data."""
    raw, _, _, _, noise_cov, label, forward, _, _, _ =\
        _get_data(all_forward=False, epochs=False, data_cov=False)

    tmin, tmax = 0, 20
    start, stop = raw.time_as_index([tmin, tmax])

    # use only the left-temporal MEG channels for LCMV
    data_cov = mne.compute_raw_covariance(raw, tmin=tmin, tmax=tmax)
    filters = make_lcmv(raw.info, forward, data_cov, reg=0.01,
                        noise_cov=noise_cov, label=label)
    stc = apply_lcmv_raw(raw, filters, start=start, stop=stop,
                         max_ori_out='signed')

    assert_array_almost_equal(np.array([tmin, tmax]),
                              np.array([stc.times[0], stc.times[-1]]),
                              decimal=2)

    # make sure we get an stc with vertices only in the lh
    vertno = [forward['src'][0]['vertno'], forward['src'][1]['vertno']]
    assert len(stc.vertices[0]) == len(np.intersect1d(vertno[0],
                                                      label.vertices))
    assert len(stc.vertices[1]) == 0


@testing.requires_testing_data
def test_lcmv_source_power():
    """Test LCMV source power computation."""
    raw, epochs, evoked, data_cov, noise_cov, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = _get_data()

    stc_source_power = _lcmv_source_power(epochs.info, forward, noise_cov,
                                          data_cov, label=label,
                                          weight_norm='unit-noise-gain')

    max_source_idx = np.argmax(stc_source_power.data)
    max_source_power = np.max(stc_source_power.data)

    assert max_source_idx == 0, max_source_idx
    assert 0.4 < max_source_power < 2.4, max_source_power

    # Test picking normal orientation and using a list of CSD matrices
    stc_normal = _lcmv_source_power(
        epochs.info, forward_surf_ori, noise_cov, data_cov,
        pick_ori="normal", label=label, weight_norm='unit-noise-gain')

    # The normal orientation results should always be smaller than free
    # orientation results
    assert (np.abs(stc_normal.data[:, 0]) <= stc_source_power.data[:, 0]).all()

    # Test if fixed forward operator is detected when picking normal
    # orientation
    pytest.raises(ValueError, _lcmv_source_power, raw.info, forward_fixed,
                  noise_cov, data_cov, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    pytest.raises(ValueError, _lcmv_source_power, raw.info, forward, noise_cov,
                  data_cov, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    pytest.raises(ValueError, _lcmv_source_power, epochs.info, forward_vol,
                  noise_cov, data_cov, pick_ori="normal")


@testing.requires_testing_data
def test_tf_lcmv():
    """Test TF beamforming based on LCMV."""
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    forward = mne.read_forward_solution(fname_fwd)

    event_id, tmin, tmax = 1, -0.2, 0.2

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.pick_types(raw.info, selection=left_temporal_channels)
    picks = picks[::2]  # decimate for speed
    raw.pick_channels([raw.ch_names[ii] for ii in picks])
    raw.info.normalize_proj()  # avoid projection warnings
    del picks

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        baseline=None, preload=False, reject=reject)
    epochs.load_data()

    freq_bins = [(4, 12), (15, 40)]
    time_windows = [(-0.1, 0.1), (0.0, 0.2)]
    win_lengths = [0.2, 0.2]
    tstep = 0.1
    reg = 0.05

    source_power = []
    noise_covs = []
    for (l_freq, h_freq), win_length in zip(freq_bins, win_lengths):
        raw_band = raw.copy()
        raw_band.filter(l_freq, h_freq, method='iir', n_jobs=1,
                        iir_params=dict(output='ba'))
        epochs_band = mne.Epochs(
            raw_band, epochs.events, epochs.event_id, tmin=tmin, tmax=tmax,
            baseline=None, proj=True)
        noise_cov = mne.compute_covariance(
            epochs_band, tmin=tmin, tmax=tmin + win_length)
        noise_cov = mne.cov.regularize(
            noise_cov, epochs_band.info, mag=reg, grad=reg, eeg=reg,
            proj=True, rank=None)
        noise_covs.append(noise_cov)
        del raw_band  # to save memory

        # Manually calculating source power in on frequency band and several
        # time windows to compare to tf_lcmv results and test overlapping
        if (l_freq, h_freq) == freq_bins[0]:
            for time_window in time_windows:
                data_cov = mne.compute_covariance(
                    epochs_band, tmin=time_window[0], tmax=time_window[1])
                stc_source_power = _lcmv_source_power(
                    epochs.info, forward, noise_cov, data_cov,
                    reg=reg, label=label, weight_norm='unit-noise-gain')
                source_power.append(stc_source_power.data)

    pytest.raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths, freq_bins, reg=reg, label=label)
    stcs = tf_lcmv(epochs, forward, noise_covs, tmin, tmax, tstep,
                   win_lengths, freq_bins, reg=reg, label=label, raw=raw)

    assert (len(stcs) == len(freq_bins))
    assert (stcs[0].shape[1] == 4)

    # Averaging all time windows that overlap the time period 0 to 100 ms
    source_power = np.mean(source_power, axis=0)

    # Selecting the first frequency bin in tf_lcmv results
    stc = stcs[0]

    # Comparing tf_lcmv results with _lcmv_source_power results
    assert_array_almost_equal(stc.data[:, 2], source_power[:, 0])

    # Test if using unsupported max-power orientation is detected
    pytest.raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths, freq_bins=freq_bins,
                  pick_ori='max-power')

    # Test if incorrect number of noise CSDs is detected
    # Test if incorrect number of noise covariances is detected
    pytest.raises(ValueError, tf_lcmv, epochs, forward, [noise_covs[0]], tmin,
                  tmax, tstep, win_lengths, freq_bins)

    # Test if freq_bins and win_lengths incompatibility is detected
    pytest.raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths=[0, 1, 2], freq_bins=freq_bins)

    # Test if time step exceeding window lengths is detected
    pytest.raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep=0.15, win_lengths=[0.2, 0.1], freq_bins=freq_bins)

    # Test if missing of noise covariance matrix is detected when more than
    # one channel type is present in the data
    pytest.raises(ValueError, tf_lcmv, epochs, forward, noise_covs=None,
                  tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths,
                  freq_bins=freq_bins)

    # Test if unsupported weight normalization specification is detected
    pytest.raises(ValueError, tf_lcmv, epochs, forward, noise_covs, tmin, tmax,
                  tstep, win_lengths, freq_bins, weight_norm='nai')

    # Test unsupported pick_ori (vector not supported here)
    with pytest.raises(ValueError, match='pick_ori'):
        tf_lcmv(epochs, forward, noise_covs, tmin, tmax, tstep, win_lengths,
                freq_bins, pick_ori='vector')
    # Test correct detection of preloaded epochs objects that do not contain
    # the underlying raw object
    epochs_preloaded = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                                  baseline=(None, 0), preload=True)
    epochs_preloaded._raw = None
    pytest.raises(ValueError, tf_lcmv, epochs_preloaded, forward,
                  noise_covs, tmin, tmax, tstep, win_lengths, freq_bins)

    # Pass only one epoch to test if subtracting evoked
    # responses yields zeros
    with pytest.warns(RuntimeWarning,
                      match='Too few samples .* estimate may be unreliable'):
        stcs = tf_lcmv(epochs[0], forward, noise_covs, tmin, tmax, tstep,
                       win_lengths, freq_bins, subtract_evoked=True, reg=reg,
                       label=label, raw=raw)

    assert_array_almost_equal(stcs[0].data, np.zeros_like(stcs[0].data))


@testing.requires_testing_data
def test_lcmv_ctf_comp():
    """Test interpolation with compensated CTF data."""
    ctf_dir = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(ctf_dir, 'somMDYO-18av.ds')
    raw = mne.io.read_raw_ctf(raw_fname, preload=True)

    events = mne.make_fixed_length_events(raw, duration=0.2)[:2]
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.2)
    evoked = epochs.average()

    with pytest.warns(RuntimeWarning,
                      match='Too few samples .* estimate may be unreliable'):
        data_cov = mne.compute_covariance(epochs)
    fwd = mne.make_forward_solution(evoked.info, None,
                                    mne.setup_volume_source_space(pos=15.0),
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


@testing.requires_testing_data
@pytest.mark.parametrize('proj', [False, True])
@pytest.mark.parametrize('weight_norm', (None, 'nai', 'unit-noise-gain'))
def test_lcmv_reg_proj(proj, weight_norm):
    """Test LCMV with and without proj."""
    raw = mne.io.read_raw_fif(fname_raw, preload=True)
    events = mne.find_events(raw)
    raw.pick_types()
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
    epochs.pick_types('grad')
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
            assert_allclose(stc.data.std(), 0.39, rtol=0.1)
    elif weight_norm is None:
        # None always represents something not normalized, reflecting channel
        # weights
        for stc in (stc_nocov, stc_cov):
            assert_allclose(stc.data.std(), 2.8e-8, rtol=0.1)
    else:
        assert weight_norm == 'unit-noise-gain'
        # Channel scalings depend on presence of noise_cov
        assert_allclose(stc_nocov.data.std(), 5.3e-13, rtol=0.1)
        assert_allclose(stc_cov.data.std(), 0.13, rtol=0.1)


@pytest.mark.parametrize('reg, weight_norm, use_cov, depth, lower, upper', [
    # the 0 reg is not so stable, can produce a wide range of scores
    (0.00, 'unit-noise-gain', True, None, 44, 90),
    (0.05, 'unit-noise-gain', True, None, 97, 98),
    (0.05, 'nai', True, None, 96, 98),
    (0.05, 'nai', True, 0.8, 96, 98),
    (0.05, None, True, None, 74, 76),
    (0.05, None, True, 0.8, 90, 93),  # depth improves weight_norm=None
    (0.05, 'unit-noise-gain', False, None, 83, 86),
    (0.05, 'unit-noise-gain', False, 0.8, 83, 86),  # depth same for wn != None
])
def test_localization_bias_fixed(bias_params_fixed, reg, weight_norm, use_cov,
                                 depth, lower, upper):
    """Test localization bias for fixed-orientation LCMV."""
    evoked, fwd, noise_cov, data_cov, want = bias_params_fixed
    if not use_cov:
        evoked.pick_types('grad')
        noise_cov = None
    assert data_cov['data'].shape[0] == len(data_cov['names'])
    loc = apply_lcmv(evoked, make_lcmv(evoked.info, fwd, data_cov, reg,
                                       noise_cov, depth=depth,
                                       weight_norm=weight_norm)).data
    loc = np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper


@pytest.mark.parametrize(
    'reg, pick_ori, weight_norm, use_cov, depth, lower, upper', [
        (0.05, 'vector', 'unit-noise-gain', True, None, 36, 39),
        (0.05, 'vector', 'unit-noise-gain', False, None, 11, 13),
        (0.05, 'vector', 'nai', True, None, 36, 39),
        (0.05, 'vector', None, True, None, 12, 14),
        (0.05, 'vector', None, True, 0.8, 39, 43),
        # (0.00, 'vector', 'unit-noise-gain', True, None, 43, 46),  # complex
        (0.05, 'max-power', 'unit-noise-gain', True, None, 20, 24),
        # (0., 'max-power', 'unit-noise-gain', True, None, 37, 40),  # complex
        (0.05, 'max-power', 'unit-noise-gain', False, None, 17, 19),
        (0.05, 'max-power', 'nai', True, None, 20, 24),
        (0.05, 'max-power', None, True, None, 7, 9),
        (0.05, 'max-power', None, True, 0.8, 16, 19),
        (0.05, None, None, True, 0.8, 40, 42),
    ])
def test_localization_bias_free(bias_params_free, reg, pick_ori, weight_norm,
                                use_cov, depth, lower, upper):
    """Test localization bias for free-orientation LCMV."""
    evoked, fwd, noise_cov, data_cov, want = bias_params_free
    if not use_cov:
        evoked.pick_types('grad')
        noise_cov = None
    loc = apply_lcmv(evoked, make_lcmv(evoked.info, fwd, data_cov, reg,
                                       noise_cov, pick_ori=pick_ori,
                                       weight_norm=weight_norm,
                                       depth=depth)).data
    loc = np.linalg.norm(loc, axis=1) if pick_ori == 'vector' else np.abs(loc)
    # Compute the percentage of sources for which there is no loc bias:
    perc = (want == np.argmax(loc, axis=0)).mean() * 100
    assert lower <= perc <= upper


@pytest.mark.parametrize('weight_norm', ('nai', 'unit-noise-gain'))
@pytest.mark.parametrize('pick_ori', ('vector', 'max-power', None))
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


def test_lcmv_maxfiltered():
    """Test LCMV on maxfiltered data."""
    raw = mne.io.read_raw_fif(fname_raw).fix_mag_coil_types()
    raw_sss = mne.preprocessing.maxwell_filter(raw)
    events = mne.find_events(raw_sss)
    del raw
    raw_sss.pick_types('mag')
    assert len(raw_sss.ch_names) == 102
    epochs = mne.Epochs(raw_sss, events)
    data_cov = mne.compute_covariance(epochs, tmin=0)
    fwd = mne.read_forward_solution(fname_fwd)
    rank = compute_rank(data_cov, info=epochs.info)
    assert rank == {'mag': 71}
    for use_rank in ('info', rank, 'full', None):
        make_lcmv(epochs.info, fwd, data_cov, rank=use_rank)


run_tests_if_main()
