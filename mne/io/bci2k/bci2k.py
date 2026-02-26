import os
import re

import numpy as np

from ... import create_info
from ...utils import verbose
from ..base import BaseRaw


def _parse_bci2k_header(fname):
    """Parse minimal BCI2000 .dat header.

    This parser intended to extract:
    - Header length (bytes)
    - Number of source channels
    - Statevector length (bytes)
    - Data format (int16/int32/float32)
    - Sampling rate
    - State definitions (name, length, bytePos, bitPos)
    """
    header = {}
    params = {}
    state_defs = {}

    def _parse_sampling_rate(val):
        # Accept e.g. "256", "256Hz", "256.0 Hz"
        text = str(val).strip()
        text = re.sub(r"\s*Hz\s*$", "", text, flags=re.IGNORECASE)
        # Grab the first float-looking token
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if m is None:
            raise ValueError(f"Could not parse SamplingRate from {val!r}")
        return float(m.group(0))

    current_section = ""

    with open(fname, "rb") as f:
        # First line: key=value pairs
        first_line = f.readline().decode("utf-8", errors="replace").strip()
        for token in first_line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                header[k] = v

        missing = [

            k
            for k in ("HeaderLen", "SourceCh", "StatevectorLen")
            if k not in header
        ]
        if missing:
            raise ValueError(
                f"BCI2000 header is missing required key(s): {', '.join(missing)}"
            )

        header_len = int(header["HeaderLen"])
        n_channels = int(header["SourceCh"])
        state_vec_len = int(header["StatevectorLen"])
        data_format = header.get("DataFormat", "int16")

        # Move through header sections up to header_len bytes
        while f.tell() < header_len:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line
                continue

            # Parameter section: try to parse "Name=Value"
            if "Parameter Definition" in current_section:
                if "=" in line:
                    left, right = line.split("=", 1)
                    name = left.strip().split()[-1]
                    value = right.strip().split()[0]
                    params[name] = value
                continue

            # State section: typically "Name Length Default BytePos BitPos"
            if "State Vector Definition" in current_section:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        length = int(parts[1])
                        byte_pos = int(parts[3])
                        bit_pos = int(parts[4])
                    except ValueError:
                        # Skip malformed or binary-contaminated lines
                        continue
                    state_defs[parts[0]] = {
                        "length": length,
                        "bytePos": byte_pos,
                        "bitPos": bit_pos,
                    }
                continue

    if "SamplingRate" not in params:
        raise ValueError(
            "Could not find 'SamplingRate' in the BCI2000 Parameter Definition section."
        )

    sfreq = _parse_sampling_rate(params["SamplingRate"])

    return {
        "header_len": header_len,
        "n_channels": n_channels,
        "state_vec_len": state_vec_len,
        "data_format": data_format,
        "sfreq": sfreq,
        "params": params,
        "state_defs": state_defs,
    }


def _read_bci2k_data(fname, info_dict):
    """Read binary signal + state data."""
    header_len = info_dict["header_len"]
    n_channels = info_dict["n_channels"]
    state_vec_len = info_dict["state_vec_len"]
    data_format = info_dict["data_format"]

    # Determine dtype
    if data_format == "int16":
        dtype = np.int16
    elif data_format == "int32":
        dtype = np.int32
    elif data_format == "float32":
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported DataFormat: {data_format}")

    bytes_per_sample = np.dtype(dtype).itemsize
    bytes_per_frame = n_channels * bytes_per_sample + state_vec_len

    file_size = os.path.getsize(fname)
    data_bytes = file_size - header_len
    n_samples = data_bytes // bytes_per_frame

    with open(fname, "rb") as f:
        f.seek(header_len)
        raw = f.read(n_samples * bytes_per_frame)

    raw = np.frombuffer(raw, dtype=np.uint8)

    # Separate signal + state
    frame_data = raw.reshape(n_samples, bytes_per_frame)
    sig_bytes = frame_data[:, : n_channels * bytes_per_sample]
    state_bytes = frame_data[:, n_channels * bytes_per_sample :]

    signal = np.frombuffer(sig_bytes.tobytes(), dtype=dtype).reshape(
        n_samples, n_channels
    )

    signal = signal.T.astype(np.float64)  # (n_channels, n_samples)
    state_bytes = state_bytes.T  # (state_vec_len, n_samples), dtype=uint8

    return signal, state_bytes


def _decode_bci2k_states(state_bytes, state_defs):
    """Decode BCI2000 state vector into integer state time series.

    Parameters
    ----------
    state_bytes : array, shape (n_bytes, n_samples), dtype=uint8
        Raw state vector bytes for each sample.
    state_defs : dict
        Mapping state name -> dict(length, bytePos, bitPos).

    Returns
    -------
    states : dict
        Mapping state name -> array, shape (n_samples,), dtype=int32.
    """
    if state_bytes.size == 0 or not state_defs:
        return {}

    n_bytes, n_samples = state_bytes.shape
    states = {}

    for name, sdef in state_defs.items():
        length = int(sdef["length"])
        byte_pos = int(sdef["bytePos"])
        bit_pos = int(sdef["bitPos"])

        vals = np.zeros(n_samples, dtype=np.int32)

        # Decode bit by bit according to spec
        for bit in range(length):
            offset = bit_pos + bit
            this_byte = byte_pos + offset // 8
            this_bit = offset % 8
            if this_byte < 0 or this_byte >= n_bytes:
                continue
            mask = 1 << this_bit
            bit_vals = (state_bytes[this_byte] & mask) >> this_bit
            vals |= bit_vals.astype(np.int32) << bit

        states[name] = vals

    return states


class RawBCI2k(BaseRaw):
    """Raw object for BCI2000 .dat files.

    Parameters
    ----------
    input_fname : path-like
        Path to the BCI2000 .dat file.
    preload : bool
        Must be True. preload=False is not supported.
    verbose : bool | str | int | None
        Control verbosity.
    """

    @verbose
    def __init__(self, input_fname, preload=False, verbose=None):
        # For now we always preload; non-preload would require chunked reading.
        if not preload:
            raise NotImplementedError(
                "preload=False is not yet supported for BCI2000; "
                "use preload=True for now."
            )

        info_dict = _parse_bci2k_header(input_fname)
        signal, state_bytes = _read_bci2k_data(input_fname, info_dict)

        sfreq = info_dict["sfreq"]
        n_channels = info_dict["n_channels"]

        # Channel names: use generic EEG names for now; can be improved later
        ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
        ch_types = ["eeg"] * n_channels

        # Decode states into integer series
        states = _decode_bci2k_states(state_bytes, info_dict["state_defs"])

        stim_data = None
        if "StimulusCode" in states:
            stim_data = states["StimulusCode"].astype(np.float32)[np.newaxis, :]
            ch_names.append("STI 014")
            ch_types.append("stim")
            signal = np.vstack([signal, stim_data])

        info = create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types,
        )

        n_samp = signal.shape[1]
        first_samp = 0
        last_samp = n_samp - 1

        # Proper BaseRaw initialization: pass preloaded data array directly
        super().__init__(
            info,
            preload=signal,  # (n_channels, n_times) array
            filenames=[input_fname],
            first_samps=[first_samp],
            last_samps=[last_samp],
            orig_format="single",
            verbose=verbose,
        )

        self._bci2k_states = states


def read_raw_bci2k(input_fname, preload=False, verbose=None):
    """Reader for BCI2000 .dat files."""
    return RawBCI2k(
        input_fname=input_fname,
        preload=preload,
        verbose=verbose,
    )
