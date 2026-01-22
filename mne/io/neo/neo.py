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

        signal = segment.analogsignals[0]

        # Get sampling rate
        sfreq = float(signal.sampling_rate.rescale("Hz").magnitude)
        logger.info("Sampling rate: %s Hz", sfreq)

        # Get channel names
        n_channels = signal.shape[1]
        if hasattr(signal, "array_annotations"):
            ann = signal.array_annotations
            if "channel_names" in ann:
                ch_names = list(ann["channel_names"])
            elif "channel_ids" in ann:
                ch_names = [str(cid) for cid in ann["channel_ids"]]
            else:
                ch_names = [f"CH{i:03d}" for i in range(n_channels)]
        else:
            ch_names = [f"CH{i:03d}" for i in range(n_channels)]

        logger.info("Found %d channels", n_channels)

        # Calculate total samples (concatenate all segments)
        segment_sizes = []
        for seg in block.segments:
            sig = seg.analogsignals[0]
            segment_sizes.append(sig.shape[0])
        total_samples = sum(segment_sizes)

        logger.info("Total samples: %d", total_samples)

        # Determine unit scale factor
        unit_str = str(signal.units.dimensionality).lower()
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
        from neo.io.proxyobjects import AnalogSignalProxy

        extras = self._raw_extras[fi]
        io_class = getattr(neo.io, extras["neo_io_class"])
        neo_reader = io_class(self._filenames[fi])

        blocks = neo_reader.read(lazy=True)
        block = blocks[0]

        segment_sizes = extras["segment_sizes"]
        scale_factor = extras["scale_factor"]
        n_channels = extras["n_channels"]

        # Find which segments contain our samples
        cum_sizes = np.cumsum([0] + segment_sizes)
        seg_start_idx = np.searchsorted(cum_sizes[1:], start, side="right")
        seg_stop_idx = np.searchsorted(cum_sizes[1:], stop - 1, side="right")

        # Load data from relevant segments
        all_data = []
        for rel_si in range(seg_start_idx, seg_stop_idx + 1):
            segment = block.segments[rel_si]
            signal = segment.analogsignals[0]

            seg_global_start = cum_sizes[rel_si]
            local_start = max(0, start - seg_global_start)
            local_stop = min(segment_sizes[rel_si], stop - seg_global_start)

            # Load signal data
            if isinstance(signal, AnalogSignalProxy):
                sig_data = signal.load().magnitude[local_start:local_stop, :]
            else:
                sig_data = signal.magnitude[local_start:local_stop, :]

            if isinstance(idx, slice):
                sig_data = sig_data[:, idx]
            else:
                sig_data = sig_data[:, idx]

            all_data.append(sig_data)

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
