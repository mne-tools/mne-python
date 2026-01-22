# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Read Micromed TRC files."""

from pathlib import Path

import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ..base import BaseRaw


@fill_doc
@verbose
def read_raw_micromed(fname, *, preload=False, verbose=None):
    """Read raw data from Micromed TRC files.

    Parameters
    ----------
    fname : path-like
        Path to the Micromed ``.trc`` file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawMicromed
        A Raw object containing the Micromed data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawMicromed.

    Notes
    -----
    Data is read using the `Neo package <https://neo.readthedocs.io/>`__.

    Channel types default to sEEG (stereo-EEG). Use :meth:`raw.set_channel_types()
    <mne.io.Raw.set_channel_types>` to set appropriate types after loading.
    Data is assumed to be in microvolts (µV) and is converted to volts (V).

    Examples
    --------
    Read a Micromed TRC file::

        >>> raw = mne.io.read_raw_micromed('recording.trc')  # doctest: +SKIP
    """
    return RawMicromed(fname, preload=preload, verbose=verbose)


@fill_doc
class RawMicromed(BaseRaw):
    """Raw object for Micromed TRC files.

    Parameters
    ----------
    fname : path-like
        Path to the Micromed ``.trc`` file.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, *, preload=False, verbose=None):
        neo = _soft_import("neo", "reading Micromed files", strict=True)

        fname_path = Path(fname)
        is_dir = fname_path.is_dir()
        fname = _check_fname(fname, "read", True, "fname", need_dir=is_dir)

        # Use Neo's MicromedIO
        neo_reader = neo.io.MicromedIO(str(fname))
        logger.info("Reading Micromed TRC file")

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

        analogsignals = segment.analogsignals
        sfreq = float(analogsignals[0].sampling_rate.rescale("Hz").magnitude)
        logger.info("Sampling rate: %s Hz", sfreq)

        # Get channel info from all analogsignals
        n_signals = len(analogsignals)
        logger.info("Reading %d analog signal(s) per segment", n_signals)

        n_channels = sum(sig.shape[1] for sig in analogsignals)
        ch_names = []
        for sig in analogsignals:
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
        segment_sizes = []
        for seg in block.segments:
            segment_sizes.append(seg.analogsignals[0].shape[0])
        total_samples = sum(segment_sizes)

        logger.info("Total samples: %d", total_samples)

        # Create info (default to sEEG for intracranial data)
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")

        # Store extras for lazy loading
        raw_extras = dict(
            segment_sizes=segment_sizes,
            n_channels=n_channels,
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
        neo = _soft_import("neo", "reading Micromed files", strict=True)

        extras = self._raw_extras[fi]
        neo_reader = neo.io.MicromedIO(self._filenames[fi])

        blocks = neo_reader.read(lazy=True)
        block = blocks[0]

        segment_sizes = extras["segment_sizes"]
        n_channels = extras["n_channels"]

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

            # Concatenate all analogsignals in this segment
            seg_signals = []
            for signal in segment.analogsignals:
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
        # Convert from µV to V
        if all_data:
            concatenated = np.concatenate(all_data, axis=0).T * 1e-6
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
