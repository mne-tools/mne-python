# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Read MEF3 files."""

from pathlib import Path

import numpy as np

from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...utils import _check_fname, _soft_import, fill_doc, logger, verbose
from ..base import BaseRaw


@fill_doc
@verbose
def read_raw_mef(fname, *, preload=False, verbose=None):
    """Read raw data from MEF3 files.

    Parameters
    ----------
    fname : path-like
        Path to the MEF3 ``.mefd`` directory.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawMEF
        A Raw object containing the MEF3 data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawMEF.

    Notes
    -----
    Data is read using the `pymef package <https://github.com/msel-source/pymef>`__.

    Channel types default to sEEG (stereo-EEG). Use :meth:`raw.set_channel_types()
    <mne.io.Raw.set_channel_types>` to set appropriate types after loading.
    Data is assumed to be in microvolts (µV) and is converted to volts (V).

    Examples
    --------
    Read a MEF3 file::

        >>> raw = mne.io.read_raw_mef('recording.mefd')  # doctest: +SKIP
    """
    return RawMEF(fname, preload=preload, verbose=verbose)


@fill_doc
class RawMEF(BaseRaw):
    """Raw object for MEF3 files.

    Parameters
    ----------
    fname : path-like
        Path to the MEF3 ``.mefd`` directory.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, *, preload=False, verbose=None):
        _soft_import("pymef", "reading MEF3 files", strict=True)
        from pymef.mef_session import MefSession

        fname = _check_fname(fname, "read", True, "fname", need_dir=True)
        logger.info("Reading MEF3 file: %s", fname)

        # Open MEF session (empty password)
        session = MefSession(str(fname), "")

        # Get channel info
        ts_channels = session.session_md["time_series_channels"]
        ch_names = list(ts_channels.keys())
        n_channels = len(ch_names)
        logger.info("Found %d channels", n_channels)

        # Get sampling rate and number of samples from first channel
        first_ch_md = ts_channels[ch_names[0]]["section_2"]
        sfreq = float(first_ch_md["sampling_frequency"][0])
        n_samples = int(first_ch_md["number_of_samples"][0])

        logger.info("Sampling rate: %s Hz", sfreq)
        logger.info("Total samples: %d", n_samples)

        # Create info (default to sEEG for intracranial data)
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg")

        # Store extras for lazy loading
        raw_extras = dict(
            n_channels=n_channels,
            ch_names=ch_names,
        )

        super().__init__(
            info=info,
            last_samps=[n_samples - 1],
            filenames=[str(fname)],
            preload=preload,
            raw_extras=[raw_extras],
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        from pymef.mef_session import MefSession

        extras = self._raw_extras[fi]
        n_channels = extras["n_channels"]
        ch_names = extras["ch_names"]

        # Open MEF session
        session = MefSession(str(self._filenames[fi]), "")

        # Determine which channels to read
        if isinstance(idx, slice):
            ch_indices = range(*idx.indices(n_channels))
        else:
            ch_indices = idx

        selected_ch_names = [ch_names[i] for i in ch_indices]

        # Read data [start, stop) - pymef uses inclusive end, so use stop
        raw_data = session.read_ts_channels_sample(selected_ch_names, [start, stop])

        # Convert to numpy array and scale from µV to V
        raw_data = np.array(raw_data, dtype=np.float64) * 1e-6

        # Build output block
        block_out = np.zeros((n_channels, stop - start), dtype=data.dtype)
        block_out[idx] = raw_data

        _mult_cal_one(data, block_out, idx, cals, mult)
