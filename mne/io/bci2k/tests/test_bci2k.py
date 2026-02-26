import os
import struct

import numpy as np
import pytest

from mne.io.bci2k import read_raw_bci2k, RawBCI2k

def _write_demo_bci2k(fname, n_channels=2, n_samples=100, sfreq=256,
                      state_vec_len=1, data_format="int16", header_len=1024):
    """Write a minimal BCI2000 .dat demo file compatible with our reader."""
    header_lines = []

    # First line: key=value pairs
    first_line = (
        f"BCI2000V=3.0 "
        f"HeaderLen={header_len} "
        f"SourceCh={n_channels} "
        f"StatevectorLen={state_vec_len} "
        f"DataFormat={data_format}"
    )
    header_lines.append(first_line)

    # Minimal Parameter Definition section
    header_lines.append("[ Parameter Definition ]")
    header_lines.append(f"SamplingRate= {sfreq}Hz")

    # State Vector Definition section: StimulusCode 8 bits at byte 0, bit 0
    header_lines.append("[ State Vector Definition ]")
    header_lines.append("StimulusCode 8 0 0 0")

    header_text = "\n".join(header_lines) + "\n"
    header_bytes = header_text.encode("utf-8")
    if len(header_bytes) > header_len:
        raise RuntimeError("Header too long for requested HeaderLen")
    header_bytes += b"\x00" * (header_len - len(header_bytes))

    # Signal data on disk
    rng = np.random.default_rng(42)
    if data_format != "int16":
        raise RuntimeError("Demo writer currently assumes int16 data_format")
    signal = rng.integers(-100, 100, size=(n_samples, n_channels),
                          dtype=np.int16)

    # StimulusCode: increases every 10 samples
    state = np.zeros((n_samples,), dtype=np.uint8)
    for i in range(n_samples):
        state[i] = i // 10

    frames = []
    for i in range(n_samples):
        frames.append(signal[i].tobytes() + struct.pack("B", state[i]))
    data_bytes = b"".join(frames)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        f.write(header_bytes)
        f.write(data_bytes)


def test_read_bci2k_demo_file(tmp_path):
    """Basic smoke test for BCI2000 .dat reader."""
    fname = tmp_path / "demo_bci2k.dat"
    _write_demo_bci2k(str(fname))

    raw = read_raw_bci2k(str(fname), preload=True)

    assert isinstance(raw, RawBCI2k)
    assert raw.info["sfreq"] == 256.0
    assert raw.n_times == 100

    # 2 EEG + 1 stim
    assert raw.ch_names == ["EEG1", "EEG2", "STI 014"]

    data = raw.get_data()
    assert data.shape == (3, 100)

    # Check StimulusCode decoding (0..9, each repeated 10 times)
    stim = raw._bci2k_states["StimulusCode"]
    expected = np.repeat(np.arange(10, dtype=np.int32), 10)
    np.testing.assert_array_equal(stim, expected)


def test_bci2k_preload_false_raises(tmp_path):
    """preload=False is not yet supported."""
    fname = tmp_path / "demo_bci2k.dat"
    _write_demo_bci2k(str(fname))

    with pytest.raises(NotImplementedError, match="preload=False"):
        read_raw_bci2k(str(fname), preload=False)