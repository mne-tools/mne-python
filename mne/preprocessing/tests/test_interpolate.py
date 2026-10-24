# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import itertools
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne import Epochs, create_info, io, pick_types, read_events
from mne.channels import make_standard_montage
from mne.preprocessing import interpolate_bridged_electrodes
from mne.preprocessing.interpolate import _find_centroid_sphere
from mne.transforms import _cart_to_sph

base_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"
event_name = base_dir / "test-eve.fif"
raw_fname_ctf = base_dir / "test_ctf_raw.fif"
event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2


def _load_data():
    """Load data."""
    # It is more memory efficient to load data in a separate
    # function so it's loaded on-demand
    raw = io.read_raw_fif(raw_fname).pick(["eeg", "stim"])
    events = read_events(event_name)
    # subselect channels for speed
    picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])[:15]
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        preload=True,
        reject=dict(eeg=80e-6),
    )
    evoked = epochs.average()
    return raw.load_data(), epochs.load_data(), evoked


@pytest.mark.parametrize("kind", ["raw", "epochs", "evoked"])
@pytest.mark.parametrize(
    "bridged", [("EEG 001", "EEG 002"), ("EEG 001", "EEG 002", "EEG 003")]
)
def test_interpolate_bridged_electrodes(kind, bridged):
    """Bridged groups are interpolated for all data types."""
    raw, epochs, evoked = _load_data()
    inst = dict(raw=raw, epochs=epochs, evoked=evoked)[kind]
    ch_names, bads = inst.ch_names.copy(), inst.info["bads"].copy()
    picks = list(bridged)
    # all pairs within the group, so a 3-channel group must be merged into one
    idx = [inst.ch_names.index(ch) for ch in bridged]
    pairs = list(itertools.combinations(idx, 2))

    reference = inst.copy()
    reference.info["bads"] = picks
    reference.interpolate_bads()

    inst = interpolate_bridged_electrodes(inst, pairs)
    assert inst.ch_names == ch_names  # virtual channel removed, order kept
    assert inst.info["bads"] == bads
    # close to, but not identical with, regular interpolation
    diff = np.abs(inst.get_data(picks=picks) - reference.get_data(picks=picks))
    assert 1e-6 < diff.mean() < 5.4e-5


def test_interpolate_bridged_electrodes_bads():
    """Pre-existing bads are preserved and do not contaminate the result."""
    raw = _load_data()[0]
    bridged = ["EEG 001", "EEG 002"]
    pairs = [tuple(raw.ch_names.index(ch) for ch in bridged)]
    raw.info["bads"] = ["EEG 001", "EEG 003"]  # bridged+bad, and bad only
    clean = interpolate_bridged_electrodes(raw.copy(), pairs)

    # a non-bridged bad must be ignored as a source and left untouched
    raw.apply_function(lambda x: np.full_like(x, 1.0), picks="EEG 003")
    out = interpolate_bridged_electrodes(raw.copy(), pairs)
    assert_array_equal(out.get_data(picks=bridged), clean.get_data(picks=bridged))
    assert_array_equal(out.get_data(picks="EEG 003"), 1.0)
    assert out.info["bads"] == ["EEG 001", "EEG 003"]

    # a bridged bad must be repaired, with no trace of its 1 V bad data
    raw.apply_function(lambda x: np.full_like(x, 2.0), picks="EEG 001")
    out = interpolate_bridged_electrodes(raw.copy(), pairs)
    assert np.abs(out.get_data(picks="EEG 001")).max() < 1e-3
    assert out.info["bads"] == ["EEG 001", "EEG 003"]


def test_interpolate_bridged_electrodes_bad_limit():
    """Groups larger than bad_limit raise, and bad_limit is validated."""
    montage = make_standard_montage("spherical_1005")
    # drop old/new alias names (e.g. T3/T7) that share positions
    ch_names = [
        ch for ch in montage.ch_names if ch not in ("P7", "P8", "T3", "T4", "T5", "T6")
    ]
    info = create_info(ch_names, sfreq=100, ch_types="eeg")
    data = np.random.default_rng(0).standard_normal((len(ch_names), 100)) * 1e-5
    data[:5] = 1e-5  # first five channels identical -> one bridged group of 5
    raw = io.RawArray(data, info).set_montage(montage)
    raw.info["bads"] = [ch_names[10]]
    pairs = list(itertools.combinations(range(5), 2))

    with pytest.raises(RuntimeError, match="form a large area of bridged"):
        interpolate_bridged_electrodes(raw, pairs, bad_limit=4)
    assert raw.info["bads"] == [ch_names[10]]  # bads restored after the error
    interpolate_bridged_electrodes(raw, pairs, bad_limit=5)  # at the limit: fine
    for bad_limit in (0, -4):
        with pytest.raises(ValueError, match="strictly positive integer"):
            interpolate_bridged_electrodes(raw, pairs, bad_limit=bad_limit)


def test_find_centroid():
    """Test that the centroid is correct."""
    montage = make_standard_montage("spherical_1005")
    ch_names = [
        ch
        for ch in montage.ch_names
        if ch not in ["P7", "P8", "T3", "T4", "T5", "T4", "T6"]
    ]
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
    pairs = [
        ("CPz", "CP2"),
        ("CPz", "Cz"),
        ("Fpz", "AFz"),
        ("AF7", "F7"),
        ("O1", "O2"),
        ("TP9", "TP9h"),
        ("P5", "P9"),
    ]
    for ch_names in pairs:
        pos_centroid = _find_centroid_sphere(pos["ch_pos"], ch_names)
        _check_centroid_position(pos, ch_names, pos_centroid)
    triplets = [("CPz", "Cz", "FCz"), ("AF9", "Fpz", "AF10"), ("FT10", "FT8", "T10")]
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
