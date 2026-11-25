# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne.datasets import testing
from mne.io import read_raw_edf, read_raw_nihon
from mne.io.nihon import nihon
from mne.io.nihon.nihon import (
    _read_nihon_annotations,
    _read_nihon_header,
    _read_nihon_metadata,
)
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)


@testing.requires_testing_data
def test_nihon_eeg():
    """Test reading Nihon Kohden EEG files."""
    fname = data_path / "NihonKohden" / "MB0400FU.EEG"
    raw = read_raw_nihon(fname.as_posix(), preload=True)
    assert "RawNihon" in repr(raw)
    _test_raw_reader(read_raw_nihon, fname=fname)
    fname_edf = data_path / "NihonKohden" / "MB0400FU.EDF"
    raw_edf = read_raw_edf(fname_edf, preload=True)
    raw_edf.drop_channels(["Events/Markers"])

    assert raw._data.shape == raw_edf._data.shape
    assert raw.info["sfreq"] == raw_edf.info["sfreq"]
    # a couple of ch names differ in the EDF
    edf_ch_names = {"EEG Mark1": "$A2", "EEG Mark2": "$A1"}
    raw_edf.rename_channels(edf_ch_names)
    assert raw.ch_names == raw_edf.ch_names

    assert len(raw.annotations) == len(raw_edf.annotations)
    for an1, an2 in zip(raw.annotations, raw_edf.annotations):
        assert an1["onset"] == an2["onset"]
        assert an1["duration"] == an2["duration"]
        assert an1["description"] == an2["description"].rstrip()

    assert_allclose(raw.get_data(), raw_edf.get_data())

    with pytest.raises(ValueError, match="Not a valid Nihon Kohden EEG file"):
        raw = read_raw_nihon(fname_edf, preload=True)

    with pytest.raises(ValueError, match="Not a valid Nihon Kohden EEG file"):
        header, _ = _read_nihon_header(fname_edf)

    bad_fname = data_path / "eximia" / "text_eximia.nxe"

    msg = "No PNT file exists. Metadata will be blank"
    with pytest.warns(RuntimeWarning, match=msg):
        meta = _read_nihon_metadata(bad_fname)
        assert len(meta) == 0

    msg = "No LOG file exists. Annotations will not be read"
    with pytest.warns(RuntimeWarning, match=msg):
        annot = _read_nihon_annotations(bad_fname)
        assert all(len(x) == 0 for x in annot.values())

    # the nihon test file has $A1 and $A2 in it, which are not EEG
    assert "$A1" in raw.ch_names

    # assert that channels with $ are 'misc'
    picks = [ch for ch in raw.ch_names if ch.startswith("$")]
    ch_types = raw.get_channel_types(picks=picks)
    assert all(ch == "misc" for ch in ch_types)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "stem, n_chan, blocks",
    [
        ("DA001000", 265, [4000]),
        ("DA001001", 265, [4000, 4000]),
        ("DA001002", 66, [10000]),
        ("DA001003", 66, [10000, 4000, 3000]),
    ],
)
@pytest.mark.filterwarnings("ignore:Channel names are not unique:RuntimeWarning")
def test_nihon_extended_blocks(stem, n_chan, blocks):
    """Test EEG-1200A high-density files with an extended block layout."""
    fname = data_path / "NihonKohden" / f"{stem}.EEG"
    raw = read_raw_nihon(fname, preload=True)
    assert len(raw.ch_names) == n_chan
    assert raw.n_times == sum(blocks)
    assert raw.info["sfreq"] == 1000.0
    # Each acquisition gap is annotated as in the single-layout files
    boundaries = [d for d in raw.annotations.description if d.endswith("boundary")]
    assert len(boundaries) == 2 * (len(blocks) - 1)
    header, _ = _read_nihon_header(fname)
    datablocks = header["controlblocks"][0]["datablocks"]
    assert [int(b["n_samples"]) for b in datablocks] == blocks
    for block in datablocks:
        # Every sample ends with a constant sync word, which is only the case if
        # the data offset and the sample stride are right
        n_samples, n_words = int(block["n_samples"]), block["n_channels"] + 1
        words = np.fromfile(
            fname, "<u2", n_samples * n_words, offset=block["data_offset"]
        ).reshape(n_samples, n_words)
        assert len(np.unique(words[:, -1])) == 1
    # Windows straddling one or several block boundaries read the same lazily
    lazy = read_raw_nihon(fname)
    ends = np.cumsum(blocks)
    for start, stop in [(0, ends[-1])] + [(e - 5, e + 5) for e in ends[:-1]]:
        assert_array_equal(lazy[:, start:stop][0], raw[:, start:stop][0])
    if len(blocks) > 2:  # a window spanning a whole middle block
        assert_array_equal(
            lazy[:, ends[0] - 5 : ends[1] + 5][0], raw[:, ends[0] - 5 : ends[1] + 5][0]
        )
    if stem == "DA001000":  # a calibration recording: 50 µV pulses
        assert_allclose(np.ptp(raw.get_data("U1")), 50e-6, rtol=1e-3)
    if stem == "DA001002":  # the generic checks assume one block, unique names
        _test_raw_reader(read_raw_nihon, fname=fname)


@testing.requires_testing_data
def test_nihon_duplicate_channels(monkeypatch):
    """Test deduplication of channel names."""
    fname = data_path / "NihonKohden" / "MB0400FU.EEG"

    def return_channel_duplicates(fname):
        ch_names = nihon._default_chan_labels
        ch_names[1] = ch_names[0]
        return ch_names

    monkeypatch.setattr(nihon, "_read_21e_file", return_channel_duplicates)

    assert len(nihon._read_21e_file(fname)) > len(set(nihon._read_21e_file(fname)))
    msg = (
        "Channel names are not unique, found duplicates for: "
        "{'FP1'}. Applying running numbers for duplicates."
    )
    with pytest.warns(RuntimeWarning, match=msg):
        read_raw_nihon(fname)


@testing.requires_testing_data
def test_nihon_calibration():
    """Test handling of calibration factor and range in Nihon Kohden EEG files."""
    fname = data_path / "NihonKohden" / "DA00100E.EEG"
    raw = read_raw_nihon(fname, preload=True, encoding="cp936")

    Fp1_idx = raw.ch_names.index("Fp1")
    M1_idx = raw.ch_names.index("M1")
    M2_idx = raw.ch_names.index("M2")

    Fp1_info = raw.info["chs"][Fp1_idx]
    M1_info = raw.info["chs"][M1_idx]
    M2_info = raw.info["chs"][M2_idx]

    # M1, M2 are EEG channels, just like Fp1.
    # So they should have the same calibration factor and physical range.
    assert_allclose(M1_info["cal"], Fp1_info["cal"])
    assert_allclose(M2_info["cal"], Fp1_info["cal"])
    assert_allclose(M1_info["range"], Fp1_info["range"])
    assert_allclose(M2_info["range"], Fp1_info["range"])

    fname_edf = data_path / "NihonKohden" / "DA00100E.EDF"
    raw_edf = read_raw_edf(fname_edf, preload=True)
    raw_edf.drop_channels(["Events/Markers"])
    # a couple of ch names differ in the EDF
    edf_ch_names = {"EEG Mark1": "$M1", "EEG Mark2": "$M2"}
    raw_edf.rename_channels(edf_ch_names)

    assert raw.ch_names == raw_edf.ch_names
    assert raw._data.shape == raw_edf._data.shape
    assert raw.info["sfreq"] == raw_edf.info["sfreq"]

    assert_allclose(raw.get_data(), raw_edf.get_data())
