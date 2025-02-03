import os
import math
import pytest
import numpy as np
from datetime import datetime, timezone

from mne.io.quspin import read_raw_quspin_lvm
from mne.io import read_raw_fif


def test_read_raw_quspin_lvm_with_preload():
    """Test reading QuSpin LVM files with preload=True."""
    # ------------------------------------------------------------------
    # 1) Paths / Setup
    # ------------------------------------------------------------------
    test_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    test_file = os.path.join(test_data_dir, 'quspin_N1_test_data.lvm')
    temp_fif = os.path.join(test_data_dir, 'test_raw.fif')

    # ------------------------------------------------------------------
    # 2) Read file with preload=True
    # ------------------------------------------------------------------
    raw = read_raw_quspin_lvm(test_file, preload=True)

    # ------------------------------------------------------------------
    # 3) Basic checks on info
    # ------------------------------------------------------------------
    # Adjust nchan and sfreq as needed to be consistent with the quspin_N1_test_data.lvm:
    expected_nchan = 224
    expected_sfreq = 374.953131

    assert raw.info['nchan'] == expected_nchan, (
        f"Expected {expected_nchan} channels, got {raw.info['nchan']}."
    )
    assert len(raw.ch_names) == expected_nchan, (
        f"Mismatch in number of channel names: expected {expected_nchan}."
    )
    assert math.isclose(raw.info['sfreq'], expected_sfreq, rel_tol=1e-6), (
        f"Sampling frequency mismatch: expected ~{expected_sfreq}, "
        f"got {raw.info['sfreq']}."
    )

    # ------------------------------------------------------------------
    # 4) Check data shape
    # ------------------------------------------------------------------
    data = raw.get_data()  # shape: (n_channels, n_times)
    assert data.shape[0] == expected_nchan, (
        f"Total channels mismatch: expected {expected_nchan}, got {data.shape[0]}."
    )
    assert data.shape[1] > 0, "No data points loaded (data.shape[1] == 0)."
    print(f"Data shape with preload=True: {data.shape}")

    # ------------------------------------------------------------------
    # 5) Round-trip saving/loading
    # ------------------------------------------------------------------
    try:
        raw.save(temp_fif, overwrite=True)
        raw_reloaded = read_raw_fif(temp_fif, preload=True)

        # Compare channel count
        assert raw.info['nchan'] == raw_reloaded.info['nchan'], (
            f"Number of channels mismatch: {raw.info['nchan']} "
            f"!= {raw_reloaded.info['nchan']}"
        )

        # Compare sampling frequency
        assert math.isclose(
            raw.info['sfreq'],
            raw_reloaded.info['sfreq'],
            rel_tol=1e-6
        ), (
            f"Sampling frequency mismatch: {raw.info['sfreq']} "
            f"!= {raw_reloaded.info['sfreq']}"
        )

        # Compare channel names
        assert raw.info['ch_names'] == raw_reloaded.info['ch_names'], (
            f"Channel names mismatch:\n{raw.info['ch_names']}\n"
            f"vs.\n{raw_reloaded.info['ch_names']}"
        )

        # Compare data
        assert np.allclose(raw.get_data(), raw_reloaded.get_data(), atol=1e-6), (
            "Data mismatch after reloading from FIF."
        )
    finally:
        # Clean up temporary file
        if os.path.exists(temp_fif):
            os.remove(temp_fif)

def test_read_raw_quspin_lvm_without_preload():
    """Test reading QuSpin LVM files with preload=False using _read_segment_file."""
    # ------------------------------------------------------------------
    # 1) Paths / Setup
    # ------------------------------------------------------------------
    test_data_dir = os.path.join(os.path.dirname(__file__), 'data')
    test_file = os.path.join(test_data_dir, 'quspin_N1_test_data.lvm')

    # Load with preload=False so that _read_segment_file gets invoked on demand
    raw_no_preload = read_raw_quspin_lvm(test_file, preload=False)

    # Basic sanity checks
    expected_nchan = 224
    expected_sfreq = 374.953131
    assert raw_no_preload.info['nchan'] == expected_nchan, (
        f"Expected {expected_nchan} channels, got {raw_no_preload.info['nchan']}"
    )
    assert math.isclose(raw_no_preload.info['sfreq'], expected_sfreq, rel_tol=1e-6), (
        f"Sampling frequency mismatch: expected {expected_sfreq}, "
        f"got {raw_no_preload.info['sfreq']}"
    )

    # ------------------------------------------------------------------
    # 2) Directly call _read_segment_file to read a partial segment
    # ------------------------------------------------------------------
    start, stop = 0, 10  # read the first 10 samples
    n_samps = stop - start
    n_chans = raw_no_preload._n_channels  # same as expected_nchan

    # This is where we store the data read by _read_segment_file:
    data_segment_on_demand = np.zeros((n_chans, n_samps), dtype=float)

    # Prepare required arguments for _read_segment_file:
    fi = 0  # file index (if you only have one file, it's 0)
    idx = np.arange(n_chans)  # read all channels
    cals = np.ones(n_chans, dtype=float)  # or your real calibration factors
    mult = None  # ignoring any projection matrix

    # Call the custom method
    # (Signature: _read_segment_file(self, data, idx, fi, start, stop, cals, mult))
    raw_no_preload._read_segment_file(
        data=data_segment_on_demand,
        idx=idx,
        fi=fi,
        start=start,
        stop=stop,
        cals=cals,
        mult=mult
    )

    # Check the shape
    assert data_segment_on_demand.shape == (n_chans, n_samps), (
        f"Unexpected shape: got {data_segment_on_demand.shape}, "
        f"expected ({n_chans}, {n_samps})"
    )
    print(f"Data segment shape (on-demand): {data_segment_on_demand.shape}")

    # ------------------------------------------------------------------
    # 3) Compare partial segment to a fully preloaded version
    # ------------------------------------------------------------------
    # Load the file again, but with preload=True
    raw_preloaded = read_raw_quspin_lvm(test_file, preload=True)
    data_segment_preloaded = raw_preloaded.get_data(start=start, stop=stop)

    # They should match closely if your on-demand reading is correct
    assert data_segment_preloaded.shape == (n_chans, n_samps), (
        "Mismatch in shape between preloaded segment and on-demand segment"
    )
    assert np.allclose(data_segment_on_demand, data_segment_preloaded, atol=1e-6), (
        "Data mismatch between on-demand read and preloaded read."
    )
    
    # ------------------------------------------------------------------
    # 4) Test out-of-bounds requests
    # ------------------------------------------------------------------
    # Request a segment beyond the file length (assuming the file doesn't have ~1e6 samples!)
    large_start = 999999
    large_stop = large_start + 10  # 10 samples beyond
    n_samps_oob = large_stop - large_start

    data_segment_oob = np.ones((n_chans, n_samps_oob), dtype=float)  # init with non-zero
    raw_no_preload._read_segment_file(
        data=data_segment_oob,
        idx=idx,
        fi=0,
        start=large_start,
        stop=large_stop,
        cals=cals,
        mult=mult
    )

    # We expect all zeros (because we've run off the end of the file)
    assert data_segment_oob.shape == (n_chans, n_samps_oob), (
        f"Out-of-bounds read shape mismatch: {data_segment_oob.shape}."
    )
    # Check if the entire array is zeros
    assert np.allclose(data_segment_oob, 0.0), (
        "Out-of-bounds samples should be filled with zeros."
    )