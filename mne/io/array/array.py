"""Tools for creating Raw objects from numpy arrays."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...utils import _check_option, _validate_type, fill_doc, logger, verbose
from ..base import BaseRaw


@fill_doc
class RawArray(BaseRaw):
    """Raw object from numpy array.

    Parameters
    ----------
    data : array, shape (n_channels, n_times)
        The channels' time series. See notes for proper units of measure.
    %(info_not_none)s Consider using :func:`mne.create_info` to populate
        this structure. This may be modified in place by the class.
    first_samp : int
        First sample offset used during recording (default 0).

        .. versionadded:: 0.12
    copy : {'data', 'info', 'both', 'auto', None}
        Determines what gets copied on instantiation. "auto" (default)
        will copy info, and copy "data" only if necessary to get to
        double floating point precision.

        .. versionadded:: 0.18
    %(verbose)s

    See Also
    --------
    mne.EpochsArray
    mne.EvokedArray
    mne.create_info

    Notes
    -----
    Proper units of measure:

    * V: eeg, eog, seeg, dbs, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc
    """

    @verbose
    def __init__(self, data, info, first_samp=0, copy="auto", verbose=None):
        _validate_type(info, "info", "info")
        _check_option("copy", copy, ("data", "info", "both", "auto", None))
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        orig_data = data
        data = np.asanyarray(orig_data, dtype=dtype)
        if data.ndim != 2:
            raise ValueError(
                "Data must be a 2D array of shape (n_channels, n_samples), got shape "
                f"{data.shape}"
            )
        if len(data) != len(info["ch_names"]):
            raise ValueError(
                'len(data) ({}) does not match len(info["ch_names"]) ({})'.format(
                    len(data), len(info["ch_names"])
                )
            )
        assert len(info["ch_names"]) == info["nchan"]
        if copy in ("auto", "info", "both"):
            info = info.copy()
        if copy in ("data", "both"):
            if data is orig_data:
                data = data.copy()
        elif copy != "auto" and data is not orig_data:
            raise ValueError(
                f"data copying was not requested by copy={copy!r} but it was required "
                "to get to double floating point precision"
            )
        logger.info(
            f"Creating RawArray with {dtype.__name__} data, "
            f"n_channels={data.shape[0]}, n_times={data.shape[1]}"
        )
        super().__init__(
            info, data, first_samps=(int(first_samp),), dtype=dtype, verbose=verbose
        )
        logger.info(
            "    Range : %d ... %d =  %9.3f ... %9.3f secs",
            self.first_samp,
            self.last_samp,
            float(self.first_samp) / info["sfreq"],
            float(self.last_samp) / info["sfreq"],
        )
        logger.info("Ready.")
