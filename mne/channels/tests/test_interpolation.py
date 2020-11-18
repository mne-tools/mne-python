import os.path as op

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from itertools import compress

from mne import io, pick_types, pick_channels, read_events, Epochs
from mne.channels.interpolation import _make_interpolation_matrix
from mne.datasets import testing
from mne.utils import run_tests_if_main
from mne.preprocessing.nirs import optical_density, scalp_coupling_index
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.io.proj import _has_eeg_average_ref_proj

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
raw_fname_ctf = op.join(base_dir, 'test_ctf_raw.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2


def _load_data(kind):
    """Load data."""
    # It is more memory efficient to load data in a separate
    # function so it's loaded on-demand
    raw = io.read_raw_fif(raw_fname)
    events = read_events(event_name)
    # subselect channels for speed
    if kind == 'eeg':
        picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])[:15]
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        preload=True, reject=dict(eeg=80e-6))
    else:
        picks = pick_types(raw.info, meg=True, eeg=False, exclude=[])[1:200:2]
        assert kind == 'meg'
        with pytest.warns(RuntimeWarning, match='projection'):
            epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                            preload=True,
                            reject=dict(grad=1000e-12, mag=4e-12))
    return raw, epochs


@pytest.mark.parametrize('offset', (0., 0.1))
@pytest.mark.parametrize('avg_proj, ctol', [
    (True, (0.86, 0.93)),
    (False, (0.97, 0.99)),
])
@pytest.mark.parametrize('method, atol', [
    (None, 3e-6),
    (dict(eeg='MNE'), 4e-6),
])
@pytest.mark.filterwarnings('ignore:.*than 20 mm from head frame origin.*')
def test_interpolation_eeg(offset, avg_proj, ctol, atol, method):
    """Test interpolation of EEG channels."""
    raw, epochs_eeg = _load_data('eeg')
    epochs_eeg = epochs_eeg.copy()
    assert not _has_eeg_average_ref_proj(epochs_eeg.info['projs'])
    # Offsetting the coordinate frame should have no effect on the output
    for inst in (raw, epochs_eeg):
        for ch in inst.info['chs']:
            if ch['kind'] == io.constants.FIFF.FIFFV_EEG_CH:
                ch['loc'][:3] += offset
                ch['loc'][3:6] += offset
        for d in inst.info['dig']:
            d['r'] += offset

    # check that interpolation does nothing if no bads are marked
    epochs_eeg.info['bads'] = []
    evoked_eeg = epochs_eeg.average()
    kw = dict(method=method)
    with pytest.warns(RuntimeWarning, match='Doing nothing'):
        evoked_eeg.interpolate_bads(**kw)

    # create good and bad channels for EEG
    epochs_eeg.info['bads'] = []
    goods_idx = np.ones(len(epochs_eeg.ch_names), dtype=bool)
    goods_idx[epochs_eeg.ch_names.index('EEG 012')] = False
    bads_idx = ~goods_idx
    pos = epochs_eeg._get_channel_positions()

    evoked_eeg = epochs_eeg.average()
    if avg_proj:
        evoked_eeg.set_eeg_reference(projection=True).apply_proj()
        assert_allclose(evoked_eeg.data.mean(0), 0., atol=1e-20)
    ave_before = evoked_eeg.data[bads_idx]

    # interpolate bad channels for EEG
    epochs_eeg.info['bads'] = ['EEG 012']
    evoked_eeg = epochs_eeg.average()
    if avg_proj:
        evoked_eeg.set_eeg_reference(projection=True).apply_proj()
        good_picks = pick_types(evoked_eeg.info, meg=False, eeg=True)
        assert_allclose(evoked_eeg.data[good_picks].mean(0), 0., atol=1e-20)
    evoked_eeg_bad = evoked_eeg.copy()
    evoked_eeg_bad.data[
        evoked_eeg.ch_names.index(epochs_eeg.info['bads'][0])] = 1e10
    evoked_eeg_interp = evoked_eeg_bad.copy().interpolate_bads(
        origin=(0., 0., 0.), **kw)
    if avg_proj:
        assert_allclose(evoked_eeg_interp.data.mean(0), 0., atol=1e-6)
    interp_zero = evoked_eeg_interp.data[bads_idx]
    if method is None:  # using
        pos_good = pos[goods_idx]
        pos_bad = pos[bads_idx]
        interpolation = _make_interpolation_matrix(pos_good, pos_bad)
        assert interpolation.shape == (1, len(epochs_eeg.ch_names) - 1)
        interp_manual = np.dot(interpolation, evoked_eeg_bad.data[goods_idx])
        assert_array_equal(interp_manual, interp_zero)
        del interp_manual, interpolation, pos, pos_good, pos_bad
    assert_allclose(ave_before, interp_zero, atol=atol)
    assert ctol[0] < np.corrcoef(ave_before, interp_zero)[0, 1] < ctol[1]
    interp_fit = evoked_eeg_bad.copy().interpolate_bads(**kw).data[bads_idx]
    assert_allclose(ave_before, interp_fit, atol=2.5e-6)
    assert ctol[1] < np.corrcoef(ave_before, interp_fit)[0, 1]  # better

    # check that interpolation fails when preload is False
    epochs_eeg.preload = False
    with pytest.raises(RuntimeError, match='requires epochs data to be loade'):
        epochs_eeg.interpolate_bads(**kw)
    epochs_eeg.preload = True

    # check that interpolation changes the data in raw
    raw_eeg = io.RawArray(data=epochs_eeg._data[0], info=epochs_eeg.info)
    raw_before = raw_eeg._data[bads_idx]
    raw_after = raw_eeg.interpolate_bads(**kw)._data[bads_idx]
    assert not np.all(raw_before == raw_after)

    # check that interpolation fails when preload is False
    for inst in [raw, epochs_eeg]:
        assert hasattr(inst, 'preload')
        inst.preload = False
        inst.info['bads'] = [inst.ch_names[1]]
        with pytest.raises(RuntimeError, match='requires.*data to be loaded'):
            inst.interpolate_bads(**kw)

    # check that interpolation works with few channels
    raw_few = raw.copy().crop(0, 0.1).load_data()
    raw_few.pick_channels(raw_few.ch_names[:1] + raw_few.ch_names[3:4])
    assert len(raw_few.ch_names) == 2
    raw_few.del_proj()
    raw_few.info['bads'] = [raw_few.ch_names[-1]]
    orig_data = raw_few[1][0]
    with pytest.warns(None) as w:
        raw_few.interpolate_bads(reset_bads=False, **kw)
    assert len([ww for ww in w if 'more than' not in str(ww.message)]) == 0
    new_data = raw_few[1][0]
    assert (new_data == 0).mean() < 0.5
    assert np.corrcoef(new_data, orig_data)[0, 1] > 0.2


@pytest.mark.slowtest
def test_interpolation_meg():
    """Test interpolation of MEG channels."""
    # speed accuracy tradeoff: channel subselection is faster but the
    # correlation drops
    thresh = 0.68

    raw, epochs_meg = _load_data('meg')

    # check that interpolation works when non M/EEG channels are present
    # before MEG channels
    raw.crop(0, 0.1).load_data().pick_channels(epochs_meg.ch_names)
    raw.info.normalize_proj()
    with pytest.warns(RuntimeWarning, match='unit .* changed from .* to .*'):
        raw.set_channel_types({raw.ch_names[0]: 'stim'})
    raw.info['bads'] = [raw.ch_names[1]]
    raw.load_data()
    raw.interpolate_bads(mode='fast')
    del raw

    # check that interpolation works for MEG
    epochs_meg.info['bads'] = ['MEG 0141']
    evoked = epochs_meg.average()
    pick = pick_channels(epochs_meg.info['ch_names'], epochs_meg.info['bads'])

    # MEG -- raw
    raw_meg = io.RawArray(data=epochs_meg._data[0], info=epochs_meg.info)
    raw_meg.info['bads'] = ['MEG 0141']
    data1 = raw_meg[pick, :][0][0]

    raw_meg.info.normalize_proj()
    data2 = raw_meg.interpolate_bads(reset_bads=False,
                                     mode='fast')[pick, :][0][0]
    assert np.corrcoef(data1, data2)[0, 1] > thresh
    # the same number of bads as before
    assert len(raw_meg.info['bads']) == len(raw_meg.info['bads'])

    # MEG -- epochs
    data1 = epochs_meg.get_data()[:, pick, :].ravel()
    epochs_meg.info.normalize_proj()
    epochs_meg.interpolate_bads(mode='fast')
    data2 = epochs_meg.get_data()[:, pick, :].ravel()
    assert np.corrcoef(data1, data2)[0, 1] > thresh
    assert len(epochs_meg.info['bads']) == 0

    # MEG -- evoked (plus auto origin)
    data1 = evoked.data[pick]
    evoked.info.normalize_proj()
    data2 = evoked.interpolate_bads(origin='auto').data[pick]
    assert np.corrcoef(data1, data2)[0, 1] > thresh


def _this_interpol(inst, ref_meg=False):
    from mne.channels.interpolation import _interpolate_bads_meg
    _interpolate_bads_meg(inst, ref_meg=ref_meg, mode='fast')
    return inst


@pytest.mark.slowtest
def test_interpolate_meg_ctf():
    """Test interpolation of MEG channels from CTF system."""
    thresh = .85
    tol = .05  # assert the new interpol correlates at least .05 "better"
    bad = 'MLC22-2622'  # select a good channel to test the interpolation

    raw = io.read_raw_fif(raw_fname_ctf).crop(0, 1.0).load_data()  # 3 secs
    raw.apply_gradient_compensation(3)

    # Show that we have to exclude ref_meg for interpolating CTF MEG-channels
    # (fixed in #5965):
    raw.info['bads'] = [bad]
    pick_bad = pick_channels(raw.info['ch_names'], raw.info['bads'])
    data_orig = raw[pick_bad, :][0]
    # mimic old behavior (the ref_meg-arg in _interpolate_bads_meg only serves
    # this purpose):
    data_interp_refmeg = _this_interpol(raw, ref_meg=True)[pick_bad, :][0]
    # new:
    data_interp_no_refmeg = _this_interpol(raw, ref_meg=False)[pick_bad, :][0]

    R = dict()
    R['no_refmeg'] = np.corrcoef(data_orig, data_interp_no_refmeg)[0, 1]
    R['with_refmeg'] = np.corrcoef(data_orig, data_interp_refmeg)[0, 1]

    print('Corrcoef of interpolated with original channel: ', R)
    assert R['no_refmeg'] > R['with_refmeg'] + tol
    assert R['no_refmeg'] > thresh


@testing.requires_testing_data
def test_interpolation_ctf_comp():
    """Test interpolation with compensated CTF data."""
    ctf_dir = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(ctf_dir, 'somMDYO-18av.ds')
    raw = io.read_raw_ctf(raw_fname, preload=True)
    raw.info['bads'] = [raw.ch_names[5], raw.ch_names[-5]]
    raw.interpolate_bads(mode='fast', origin=(0., 0., 0.04))
    assert raw.info['bads'] == []


@testing.requires_testing_data
def test_interpolation_nirs():
    """Test interpolating bad nirs channels."""
    fname = op.join(data_path(download=False),
                    'NIRx', 'nirscout', 'nirx_15_2_recording_w_overlap')
    raw_intensity = read_raw_nirx(fname, preload=False)
    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(raw_od)
    raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))
    bad_0 = np.where([name == raw_od.info['bads'][0] for
                      name in raw_od.ch_names])[0][0]
    bad_0_std_pre_interp = np.std(raw_od._data[bad_0])
    raw_od.interpolate_bads()
    assert raw_od.info['bads'] == []
    assert bad_0_std_pre_interp > np.std(raw_od._data[bad_0])


run_tests_if_main()
