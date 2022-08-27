import itertools
import os.path as op

import numpy as np
import pytest

from mne import create_info, io, pick_types, read_events, Epochs
from mne.channels import make_standard_montage
from mne.preprocessing import equalize_bads, interpolate_bridged_electrodes
from mne.preprocessing.interpolate import _find_centroid_sphere
from mne.transforms import _cart_to_sph

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
raw_fname_ctf = op.join(base_dir, 'test_ctf_raw.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2


def _load_data():
    """Load data."""
    # It is more memory efficient to load data in a separate
    # function so it's loaded on-demand
    raw = io.read_raw_fif(raw_fname).pick(['eeg', 'stim'])
    events = read_events(event_name)
    # subselect channels for speed
    picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])[:15]
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, reject=dict(eeg=80e-6))
    evoked = epochs.average()
    return raw.load_data(), epochs.load_data(), evoked


@pytest.mark.parametrize('interp_thresh', [0., 0.5, 1.])
@pytest.mark.parametrize('inst_type', ['raw', 'epochs', 'evoked'])
def test_equalize_bads(interp_thresh, inst_type):
    """Test equalize_bads function."""
    raw, epochs, evoked = _load_data()

    if inst_type == 'raw':
        insts = [raw.copy().crop(0, 1), raw.copy().crop(0, 2)]
    elif inst_type == 'epochs':
        insts = [epochs.copy()[:1], epochs.copy()[:2]]
    else:
        insts = [evoked.copy().crop(0, 0.1), raw.copy().crop(0, 0.2)]

    with pytest.raises(ValueError, match='between 0'):
        equalize_bads(insts, interp_thresh=2.)

    bads = insts[0].copy().pick('eeg').ch_names[:3]
    insts[0].info['bads'] = bads[:2]
    insts[1].info['bads'] = bads[1:]

    insts_ok = equalize_bads(insts, interp_thresh=interp_thresh)
    if interp_thresh == 0:
        bads_ok = []
    elif interp_thresh == 1:
        bads_ok = bads
    else:  # interp_thresh == 0.5
        bads_ok = bads[1:]

    for inst in insts_ok:
        assert set(inst.info['bads']) == set(bads_ok)


def test_interpolate_bridged_electrodes():
    """Test interpolate_bridged_electrodes function."""
    raw, epochs, evoked = _load_data()
    for inst in (raw, epochs, evoked):
        idx0 = inst.ch_names.index('EEG 001')
        idx1 = inst.ch_names.index('EEG 002')
        ch_names_orig = inst.ch_names.copy()
        bads_orig = inst.info['bads'].copy()
        inst2 = inst.copy()
        inst2.info['bads'] = ['EEG 001', 'EEG 002']
        inst2.interpolate_bads()
        data_interp_reg = inst2.get_data(picks=['EEG 001', 'EEG 002'])
        inst = interpolate_bridged_electrodes(inst, [(idx0, idx1)])
        data_interp = inst.get_data(picks=['EEG 001', 'EEG 002'])
        assert not any(['virtual' in ch for ch in inst.ch_names])
        assert inst.ch_names == ch_names_orig
        assert inst.info['bads'] == bads_orig
        # check closer to regular interpolation than original data
        assert 1e-6 < np.mean(np.abs(data_interp - data_interp_reg)) < 5.4e-5

    for inst in (raw, epochs, evoked):
        idx0 = inst.ch_names.index('EEG 001')
        idx1 = inst.ch_names.index('EEG 002')
        idx2 = inst.ch_names.index('EEG 003')
        ch_names_orig = inst.ch_names.copy()
        bads_orig = inst.info['bads'].copy()
        inst2 = inst.copy()
        inst2.info['bads'] = ['EEG 001', 'EEG 002', 'EEG 003']
        inst2.interpolate_bads()
        data_interp_reg = inst2.get_data(
            picks=['EEG 001', 'EEG 002', 'EEG 003']
        )
        inst = interpolate_bridged_electrodes(
            inst, [(idx0, idx1), (idx0, idx2), (idx1, idx2)]
        )
        data_interp = inst.get_data(picks=['EEG 001', 'EEG 002', 'EEG 003'])
        assert not any(['virtual' in ch for ch in inst.ch_names])
        assert inst.ch_names == ch_names_orig
        assert inst.info['bads'] == bads_orig
        # check closer to regular interpolation than original data
        assert 1e-6 < np.mean(np.abs(data_interp - data_interp_reg)) < 5.4e-5

    # test bad_limit
    montage = make_standard_montage("standard_1020")
    ch_names = [ch for ch in montage.ch_names
                if ch not in ["P7", "P8", "T3", "T4", "T5", "T4", "T6"]]
    info = create_info(ch_names, sfreq=1024, ch_types="eeg")
    data = np.random.randn(len(ch_names), 1024)
    data[:5, :] = np.ones((5, 1024))
    raw = io.RawArray(data, info)
    raw.set_montage("standard_1020")
    bridged_idx = list(itertools.combinations(range(5), 2))
    with pytest.raises(
        RuntimeError,
        match="The channels Fp1, Fpz, Fp2, AF9, AF7 are bridged "
        "together and form a large area of bridged electrodes."
    ):
        interpolate_bridged_electrodes(raw, bridged_idx, bad_limit=4)
    # increase the limit to prevent raising
    interpolate_bridged_electrodes(raw, bridged_idx, bad_limit=5)
    # invalid argument
    with pytest.raises(
        ValueError,
        match="Argument 'bad_limit' should be a strictly positive integer."
    ):
        interpolate_bridged_electrodes(raw, bridged_idx, bad_limit=-4)


def test_find_centroid():
    """Test that the centroid is correct."""
    montage = make_standard_montage("standard_1020")
    ch_names = [ch for ch in montage.ch_names
                if ch not in ["P7", "P8", "T3", "T4", "T5", "T4", "T6"]]
    info = create_info(ch_names, sfreq=1024, ch_types="eeg")
    info.set_montage(montage)
    montage = info.get_montage()
    pos = montage.get_positions()
    assert pos["coord_frame"] == "head"

    # look for centroid between T7 and TP7, an average in spehrical coordinate
    # fails and places the average on the wrong side of the head between T8 and
    # TP8
    ch_names = ["T7", "TP7"]
    pos_centroid = _find_centroid_sphere(pos["ch_pos"], ch_names)
    _check_centroid_position(pos, ch_names, pos_centroid)

    # check other positions
    pairs = [("CPz", "CP2"), ("CPz", "Cz"), ("Fpz", "AFz"), ("AF7", "F7"),
             ("O1", "O2"), ("M2", "A2"), ("P5", "P9")]
    for ch_names in pairs:
        pos_centroid = _find_centroid_sphere(pos["ch_pos"], ch_names)
        _check_centroid_position(pos, ch_names, pos_centroid)
    triplets = [("CPz", "Cz", "FCz"), ("AF9", "Fpz", "AF10"),
                ("FT10", "FT8", "T10")]
    for ch_names in triplets:
        pos_centroid = _find_centroid_sphere(pos["ch_pos"], ch_names)
        _check_centroid_position(pos, ch_names, pos_centroid)


def _check_centroid_position(pos, ch_names, pos_centroid):
    """Check the centroid distance.

    The cartesian average should be distanced from pos_centroid by the
    difference between the radii.
    """
    radii = list()
    cartesian_positions = np.zeros((len(ch_names), 3))
    for i, ch in enumerate(ch_names):
        radii.append(_cart_to_sph(pos["ch_pos"][ch])[0, 0])
        cartesian_positions[i, :] = pos["ch_pos"][ch]
    avg_radius = np.average(radii)
    avg_cartesian_position = np.average(cartesian_positions, axis=0)
    avg_cartesian_position_radius = _cart_to_sph(avg_cartesian_position)[0, 0]
    radius_diff = np.abs(avg_radius - avg_cartesian_position_radius)
    # distance
    distance = np.linalg.norm(pos_centroid - avg_cartesian_position)
    assert np.isclose(radius_diff, distance, atol=1e-6)
