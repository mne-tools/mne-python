# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne.datasets import testing
from mne.io import read_raw_edf, read_raw_nihon
from mne.io.nihon import nihon
from mne.io.nihon.nihon import (
    _default_chan_labels,
    _map_ch_to_specs,
    _read_nihon_annotations,
    _read_nihon_header,
    _read_nihon_metadata,
)
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)


def _write_uint(buf, offset, value, dtype):
    buf[offset : offset + np.dtype(dtype).itemsize] = np.array(
        value, dtype=dtype
    ).tobytes()


def _write_str(buf, offset, s):
    encoded = s.encode("ascii")
    buf[offset : offset + len(encoded)] = encoded


def _write_synthetic_1200a_eeg(tmp_path, ch_indices, sfreq, samples):
    """Write a minimal synthetic EEG-1200A file set (.EEG/.PNT/.21E/.LOG).

    EEG-1200A systems store the real channel count/order and the start of
    the raw samples in a chain of "extended blocks" rather than in the
    data block itself (see ``_read_nihon_1200a_ext_block``). This writes
    just enough of that structure, with made-up (non-patient) sample data,
    to exercise that code path without requiring a real recording.
    """
    version = "EEG-1200A V01.00"
    n_channels = len(ch_indices)
    n_samples = samples.shape[1]
    assert samples.shape == (n_channels + 1, n_samples)  # +1 for marker channel

    ctl_address = 0x400
    # 0x17FE is a fixed, unconditional offset checked in every valid Nihon
    # Kohden file (the "waveform block" signature byte), so the header must
    # extend at least that far regardless of where the other blocks live.
    data_address = 0x17FE
    ext_address = 0x1900
    extblock2_address = 0x1A00
    extblock3_address = 0x1B00
    data_start = extblock3_address + 72 + n_channels * 10

    buf = bytearray(data_start)
    _write_str(buf, 0x0000, version)
    _write_str(buf, 0x0081, version)
    _write_uint(buf, 0x17FE, 1, np.uint8)  # waveform sign
    _write_uint(buf, 0x03EE, ext_address, np.uint32)
    _write_uint(buf, 0x0091, 1, np.uint8)  # n_ctlblocks
    _write_uint(buf, 0x0092, ctl_address, np.uint32)
    _write_uint(buf, ctl_address + 17, 1, np.uint8)  # n_datablocks
    _write_uint(buf, ctl_address + 18, data_address, np.uint32)
    _write_uint(buf, data_address + 0x1A, sfreq, np.uint16)

    _write_uint(buf, ext_address + 18, extblock2_address, np.uint32)
    _write_uint(buf, extblock2_address + 20, extblock3_address, np.uint32)
    _write_uint(buf, extblock3_address + 68, n_channels, np.uint16)
    for i_ch, idx in enumerate(ch_indices):
        _write_uint(buf, extblock3_address + 72 + i_ch * 10, idx, np.uint16)

    # Samples are stored as the "NK int16" encoding: uint16 offset by 0x8000.
    stored = (samples.view(np.uint16) + np.uint16(0x8000)).astype("<u2")

    fname = tmp_path / "SYNTH1200A.EEG"
    with open(fname, "wb") as fid:
        fid.write(bytes(buf))
        fid.write(stored.tobytes(order="F"))

    pnt = bytearray(0x40 + 14)
    _write_str(pnt, 0x0000, version)
    _write_str(pnt, 0x40, "20260101120000")
    with open(fname.with_suffix(".PNT"), "wb") as fid:
        fid.write(bytes(pnt))

    log = bytearray(0x92)
    _write_str(log, 0x0000, version)
    _write_uint(log, 0x91, 0, np.uint8)  # n_logblocks
    with open(fname.with_suffix(".LOG"), "wb") as fid:
        fid.write(bytes(log))

    with open(fname.with_suffix(".21E"), "w", encoding="ascii") as fid:
        fid.write("[ELECTRODE]\n")
        for idx in ch_indices:
            fid.write(f"{idx}={_default_chan_labels[idx]}\n")

    return fname


@testing.requires_testing_data
def test_nihon_eeg():
    """Test reading Nihon Kohden EEG files."""
    fname = data_path / "NihonKohden" / "MB0400FU.EEG"
    raw = read_raw_nihon(fname.as_posix(), preload=True)
    assert "RawNihon" in repr(raw)
    _test_raw_reader(read_raw_nihon, fname=fname, test_scaling=False)
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
def test_nihon_duplicate_channels(monkeypatch):
    """Test deduplication of channel names."""
    fname = data_path / "NihonKohden" / "MB0400FU.EEG"

    def return_channel_duplicates(fname):
        # copy so we don't permanently mutate the shared module-level list
        ch_names = list(nihon._default_chan_labels)
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


def test_nihon_1200a_extended_block(tmp_path):
    """Test reading an EEG-1200A file via the extended channel block."""
    rng = np.random.RandomState(0)
    ch_names = ["FP1", "FP2", "F3", "F4"]
    ch_indices = [_default_chan_labels.index(name) for name in ch_names]
    sfreq = 500
    n_samples = 200
    # last row is the marker channel, which read_raw_nihon drops
    samples = rng.randint(-1000, 1000, size=(len(ch_names) + 1, n_samples))
    samples = samples.astype(np.int16)

    fname = _write_synthetic_1200a_eeg(tmp_path, ch_indices, sfreq, samples)
    raw = read_raw_nihon(fname, preload=True)

    assert raw.ch_names == ch_names
    assert raw.info["sfreq"] == sfreq
    assert raw.n_times == n_samples

    chan_labels_upper = [x.upper() for x in _default_chan_labels]
    specs = [_map_ch_to_specs(name, chan_labels_upper) for name in ch_names]
    expected = np.stack(
        [
            (samples[i].astype(np.float64) * spec["cal"] + spec["offset"])
            * spec["unit"]
            for i, spec in enumerate(specs)
        ]
    )
    assert_allclose(raw.get_data(), expected)


def test_nihon_1200a_multi_datablock_not_supported(tmp_path):
    """EEG-1200A files with >1 data block should raise, not misread."""
    rng = np.random.RandomState(0)
    ch_names = ["FP1", "FP2"]
    ch_indices = [_default_chan_labels.index(name) for name in ch_names]
    sfreq = 500
    samples = rng.randint(-1000, 1000, size=(len(ch_names) + 1, 10)).astype(np.int16)

    fname = _write_synthetic_1200a_eeg(tmp_path, ch_indices, sfreq, samples)
    # Claim a second data block in the (only) control block.
    with open(fname, "r+b") as fid:
        ctl_address = 0x400
        fid.seek(ctl_address + 17)
        fid.write(np.array(2, dtype=np.uint8).tobytes())

    with pytest.raises(NotImplementedError, match="more than one"):
        read_raw_nihon(fname)
