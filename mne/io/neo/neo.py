# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Read data from Neo-supported formats."""

from pathlib import Path

import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ..base import BaseRaw


def _select_analog_signals(analogsignals, stream_id_to_name):
    """Select analog signals to read, preferring neural/amplifier data.

    Parameters
    ----------
    analogsignals : list
        List of Neo AnalogSignal or AnalogSignalProxy objects.
    stream_id_to_name : dict
        Mapping from stream_id to stream name.

    Returns
    -------
    selected : list
        List of selected analog signals.
    """
    if len(analogsignals) == 1:
        return analogsignals

    # Group signals by stream_id
    from collections import defaultdict

    by_stream = defaultdict(list)
    for sig in analogsignals:
        stream_id = str(sig.annotations.get("stream_id", "unknown"))
        by_stream[stream_id].append(sig)

    # Try to find amplifier/neural streams
    amplifier_signals = []
    for stream_id, signals in by_stream.items():
        stream_name = stream_id_to_name.get(stream_id, "").lower()
        if "amplifier" in stream_name:
            amplifier_signals.extend(signals)

    if amplifier_signals:
        # Found amplifier streams - use those
        skipped = len(analogsignals) - len(amplifier_signals)
        if skipped > 0:
            logger.info(
                "Selected %d amplifier signal(s), skipping %d non-neural signal(s)",
                len(amplifier_signals),
                skipped,
            )
        return amplifier_signals

    # No amplifier streams found - group by sampling rate and select highest
    by_sfreq = defaultdict(list)
    for sig in analogsignals:
        sfreq = float(sig.sampling_rate.rescale("Hz").magnitude)
        by_sfreq[sfreq].append(sig)

    if len(by_sfreq) > 1:
        # Multiple sampling rates - select highest (typically neural data)
        highest_sfreq = max(by_sfreq.keys())
        selected = by_sfreq[highest_sfreq]
        skipped_sfreqs = [s for s in by_sfreq.keys() if s != highest_sfreq]
        logger.warning(
            "Multiple sampling rates found. Selecting signals at %.1f Hz. "
            "Skipping signals at %s Hz. Use Neo directly if you need other signals.",
            highest_sfreq,
            skipped_sfreqs,
        )
        return selected

    # All same sampling rate - return all
    return analogsignals


@fill_doc
@verbose
def read_raw_neo(fname, *, neo_io_class=None, preload=False, verbose=None):
    """Read raw data from Neo-supported formats.

    This function provides a generic interface to read electrophysiology data
    using the `Neo <https://neo.readthedocs.io/>`_ library.

    Parameters
    ----------
    fname : path-like
        Path to the file or directory to read.
    neo_io_class : str | None
        Name of the Neo IO class to use (e.g., ``'MicromedIO'``, ``'NWBIO'``,
        ``'IntanIO'``). If ``None``, Neo will attempt to auto-detect the format
        based on the file extension. See the `list of Neo IO classes
        <https://neo.readthedocs.io/en/stable/iolist.html>`__.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawNeo
        A Raw object containing the Neo data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawNeo.

    Notes
    -----
    Data is read using the `Neo package <https://neo.readthedocs.io/>`__.
    Some formats may require additional dependencies (see Neo documentation).

    Channel types default to EEG. Use :meth:`raw.set_channel_types()
    <mne.io.Raw.set_channel_types>` to set appropriate types after loading.

    When a Neo file contains multiple analog signals per segment, the reader
    automatically selects neural/amplifier signals based on the stream metadata.
    For formats like Intan, signals with "amplifier" in the stream name are
    selected. If no amplifier streams are found but multiple sampling rates
    exist, the signals with the highest sampling rate are selected (neural data
    is typically recorded at the highest rate). Other signals (auxiliary inputs,
    digital channels, etc.) are skipped with a warning. Use Neo directly if you
    need access to non-neural signals.

    Examples
    --------
    Read a file with auto-detection::

        >>> raw = mne.io.read_raw_neo('recording.trc')  # doctest: +SKIP

    Read a file with explicit IO class::

        >>> raw = mne.io.read_raw_neo('data.rhd',
        ...                           neo_io_class='IntanIO')  # doctest: +SKIP
    """
    return RawNeo(fname, neo_io_class=neo_io_class, preload=preload, verbose=verbose)


@fill_doc
class RawNeo(BaseRaw):
    """Raw object for Neo-supported formats.

    Parameters
    ----------
    fname : path-like
        Path to the file or directory to read.
    neo_io_class : str | None
        Name of the Neo IO class to use. If ``None``, auto-detects format.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, *, neo_io_class=None, preload=False, verbose=None):
        neo = _soft_import("neo", "reading Neo format files", strict=True)

        # Check if path exists
        fname_path = Path(fname)
        is_dir = fname_path.is_dir()
        must_exist = neo_io_class != "ExampleIO"
        fname = _check_fname(fname, "read", must_exist, "fname", need_dir=is_dir)

        # Get IO class - auto-detect if not specified
        if neo_io_class is None:
            neo_reader = neo.io.get_io(str(fname))
            neo_io_class = type(neo_reader).__name__
            logger.info("Auto-detected Neo IO class: %s", neo_io_class)
        else:
            if not hasattr(neo.io, neo_io_class):
                available = [name for name in dir(neo.io) if name.endswith("IO")]
                raise ValueError(
                    f"Unknown Neo IO class: {neo_io_class!r}. "
                    f"Available classes include: {available[:10]}..."
                )
            io_class = getattr(neo.io, neo_io_class)
            neo_reader = io_class(str(fname))
            logger.info("Using Neo IO class: %s", neo_io_class)

        # Read block structure (lazy)
        blocks = neo_reader.read(lazy=True)
        if not blocks:
            raise ValueError("No data blocks found in file.")

        block = blocks[0]
        if not block.segments:
            raise ValueError("No segments found in data block.")

        logger.info("Found %d segment(s)", len(block.segments))

        # Get signal info from first segment
        segment = block.segments[0]
        if not segment.analogsignals:
            raise ValueError("No analog signals found in segment.")

        # Build stream_id to stream_name mapping from header
        stream_id_to_name = {}
        if hasattr(neo_reader, "header") and "signal_streams" in neo_reader.header:
            for stream in neo_reader.header["signal_streams"]:
                stream_id_to_name[str(stream["id"])] = stream["name"]

        # Select which analogsignals to read based on stream type
        selected_signals = _select_analog_signals(
            segment.analogsignals, stream_id_to_name
        )
        # Store indices for use in _read_segment_file
        selected_indices = [
            segment.analogsignals.index(sig) for sig in selected_signals
        ]

        # Check all selected analogsignals have the same sampling rate
        sfreqs = [
            float(sig.sampling_rate.rescale("Hz").magnitude)
            for sig in selected_signals
        ]
        if len(set(sfreqs)) > 1:
            raise ValueError(
                f"Multiple sampling rates found in selected signals: {set(sfreqs)} Hz. "
                "MNE requires all analog signals to have the same sampling rate. "
                "Use Neo directly to read signals with a specific sampling rate."
            )
        sfreq = sfreqs[0]
        logger.info("Sampling rate: %s Hz", sfreq)

        # Get channel info from selected analogsignals
        n_signals = len(selected_signals)
        logger.info("Reading %d analog signal(s) per segment", n_signals)

        n_channels = sum(sig.shape[1] for sig in selected_signals)
        ch_names = []
        for sig in selected_signals:
            n_ch = sig.shape[1]
            if hasattr(sig, "array_annotations"):
                ann = sig.array_annotations
                if "channel_names" in ann:
                    ch_names.extend(list(ann["channel_names"]))
                elif "channel_ids" in ann:
                    ch_names.extend([str(cid) for cid in ann["channel_ids"]])
                else:
                    offset = len(ch_names)
                    ch_names.extend([f"CH{i + offset:03d}" for i in range(n_ch)])
            else:
                offset = len(ch_names)
                ch_names.extend([f"CH{i + offset:03d}" for i in range(n_ch)])

        logger.info("Found %d channels", n_channels)

        # Calculate total samples (concatenate all segments)
        # Use the first selected signal index to get sample counts
        first_sig_idx = selected_indices[0]
        segment_sizes = []
        for seg in block.segments:
            sig = seg.analogsignals[first_sig_idx]
            segment_sizes.append(sig.shape[0])
        total_samples = sum(segment_sizes)

        logger.info("Total samples: %d", total_samples)

        # Determine unit scale factor (from first selected analogsignal)
        unit_str = str(selected_signals[0].units.dimensionality).lower()
        if "microv" in unit_str or "uv" in unit_str:
            scale_factor = 1e-6
        elif "milliv" in unit_str or "mv" in unit_str:
            scale_factor = 1e-3
        elif unit_str == "v" or "volt" in unit_str:
            scale_factor = 1.0
        else:
            scale_factor = 1e-6  # Default to microvolts

        # Create info (default to EEG, users can change with set_channel_types)
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

        # Store extras for lazy loading
        raw_extras = dict(
            neo_io_class=neo_io_class,
            segment_sizes=segment_sizes,
            scale_factor=scale_factor,
            n_channels=n_channels,
            selected_signal_indices=selected_indices,
        )

        super().__init__(
            info=info,
            last_samps=[total_samples - 1],
            filenames=[str(fname)],
            preload=preload,
            raw_extras=[raw_extras],
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        neo = _soft_import("neo", "reading Neo format files", strict=True)

        extras = self._raw_extras[fi]
        io_class = getattr(neo.io, extras["neo_io_class"])
        neo_reader = io_class(self._filenames[fi])

        blocks = neo_reader.read(lazy=True)
        block = blocks[0]

        segment_sizes = extras["segment_sizes"]
        scale_factor = extras["scale_factor"]
        n_channels = extras["n_channels"]
        selected_indices = extras["selected_signal_indices"]

        # Find which segments contain our samples
        cum_sizes = np.cumsum([0] + segment_sizes)
        seg_start_idx = np.searchsorted(cum_sizes[1:], start, side="right")
        seg_stop_idx = np.searchsorted(cum_sizes[1:], stop - 1, side="right")

        # Load data from relevant segments
        all_data = []
        for rel_si in range(seg_start_idx, seg_stop_idx + 1):
            segment = block.segments[rel_si]

            seg_global_start = cum_sizes[rel_si]
            local_start = max(0, start - seg_global_start)
            local_stop = min(segment_sizes[rel_si], stop - seg_global_start)

            # Concatenate selected analogsignals in this segment
            seg_signals = []
            for sig_idx in selected_indices:
                signal = segment.analogsignals[sig_idx]
                if isinstance(signal, neo.io.proxyobjects.AnalogSignalProxy):
                    sig_data = signal.load().magnitude[local_start:local_stop, :]
                else:
                    sig_data = signal.magnitude[local_start:local_stop, :]
                seg_signals.append(sig_data)

            # Concatenate along channel axis (axis=1)
            seg_data = np.concatenate(seg_signals, axis=1)

            if isinstance(idx, slice):
                seg_data = seg_data[:, idx]
            else:
                seg_data = seg_data[:, idx]

            all_data.append(seg_data)

        # Concatenate and transpose to (n_channels, n_samples)
        if all_data:
            concatenated = np.concatenate(all_data, axis=0).T * scale_factor
        else:
            n_idx = (
                len(range(*idx.indices(n_channels)))
                if isinstance(idx, slice)
                else len(idx)
            )
            concatenated = np.zeros((n_idx, stop - start))

        block_out = np.zeros((n_channels, stop - start), dtype=data.dtype)
        block_out[idx] = concatenated

        _mult_cal_one(data, block_out, idx, cals, mult)
