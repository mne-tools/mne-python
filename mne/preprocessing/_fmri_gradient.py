"""Remove fMRI gradient (imaging) artifacts from EEG recorded during MRI."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from scipy.signal import detrend

from .._fiff.pick import _picks_to_idx
from ..utils import _check_option, _validate_type, verbose


class GradientRemover:
    """Remove the fMRI gradient artifact from EEG data using average templates.

    Implements the average artifact subtraction (AAS) method of
    :footcite:`AllenEtAl2000`: for each imaging volume (TR) epoch the data are
    detrended, an artifact template is formed by averaging neighboring TR
    epochs, and the template is subtracted from the epoch.

    This class operates on a plain :class:`~numpy.ndarray` and exposes the
    intermediate templates and per-TR access, which makes it suitable both for
    offline correction and as a building block for real-time correction. Most
    users should prefer :func:`mne.preprocessing.remove_fmri_gradient_artifact`,
    which operates directly on a :class:`~mne.io.Raw` object.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_times)
        The raw EEG data to perform gradient correction on.
    tr_events : ndarray, shape (n_trs,) | (n_trs, 3)
        The sample numbers at which TRs (imaging volumes) begin. May be a
        1D array of sample numbers, or an ``(n_trs, 3)`` events array as
        returned by :func:`mne.find_events` (the first column is used). TRs
        must be evenly spaced in time, within ``tr_tol`` samples (see below).
    window : int | tuple of int
        The number of neighboring TRs to average into each template. Either an
        even integer giving the total number of TRs (split evenly before and
        after), or a ``(n_before, n_after)`` tuple. For example, ``(4, 0)``
        averages the 4 preceding TRs only. Default ``(4, 4)``.
    tr_tol : int
        The maximum allowed deviation (in samples) of any individual TR
        spacing from the median TR spacing. Real-world trigger timestamps
        (e.g. as reconstructed from wall-clock event times) are often off by
        a sample or two even when the true TR spacing is fixed; this
        tolerance allows such jitter without rejecting the data. Each TR
        epoch is anchored at its own detected onset sample, so small
        deviations do not accumulate across TRs. Default 0.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, eeg_data, tr_events, window=(4, 4), tr_tol=0):
        self._window = GradientRemover._valid_window(window)
        self._tr_events = GradientRemover._valid_tr_events(tr_events, tr_tol)
        if self._tr_events[-1] + self.tr_spacing > eeg_data.shape[1]:
            raise ValueError(
                f"Last TR event is sample {self._tr_events[-1]} but "
                f"eeg data only contains {eeg_data.shape[1]} samples. "
                "Please check your tr event markers."
            )
        self._data = eeg_data
        # Get weights for template
        window_total = self.window[0] + self.window[1]
        self._weight_before = self.window[0] / window_total
        self._weight_after = self.window[1] / window_total
        # Lazy evaluation
        self._corrected = None

    @property
    def corrected(self):
        """The gradient-corrected data (computed on first access)."""
        if self._corrected is not None:
            return self._corrected
        return self.correct()

    @property
    def window(self):
        """The template window as a ``(n_before, n_after)`` tuple."""
        return self._window

    @property
    def tr_spacing(self):
        """The (median) number of samples between consecutive TRs."""
        return int(np.round(np.median(np.diff(self._tr_events))))

    @property
    def n_tr(self):
        """The number of TRs (imaging volumes)."""
        return len(self._tr_events)

    @property
    def n_channels(self):
        """The number of channels."""
        return len(self._data)

    def get_tr(self, n):
        """Get the uncorrected data at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the uncorrected data at (0-indexed).

        Returns
        -------
        data : ndarray, shape (n_channels, tr_spacing)
            The uncorrected data at the given TR.
        """
        this_start, this_end = self._tr_bounds(n)
        return self._data[:, this_start:this_end]

    def get_tr_detrended(self, n):
        """Get the detrended data at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the detrended data at (0-indexed).

        Returns
        -------
        data : ndarray, shape (n_channels, tr_spacing)
            The detrended data at the given TR.
        """
        return detrend(self.get_tr(n))

    def get_tr_template(self, n):
        """Get the gradient artifact template at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the template at (0-indexed).

        Returns
        -------
        template : ndarray, shape (n_channels, tr_spacing)
            The artifact template at the given TR.
        """
        self._check_valid_tr(n)
        if n < self.window[0] or n > (self.n_tr - self.window[1]):
            return np.zeros((self.n_channels, self.tr_spacing))
        if self.window[0]:
            before = self._get_tr_template_part(n - self.window[0], n)
        else:
            before = 0
        if self.window[1]:
            after = self._get_tr_template_part(n + 1, n + self.window[1] - 1)
        else:
            after = 0
        return self._weight_before * before + self._weight_after * after

    def _get_tr_template_part(self, start, stop):
        return np.mean(
            np.asarray([self.get_tr_detrended(tr) for tr in range(start, stop)]),
            axis=0,
        )

    def get_tr_corrected(self, n):
        """Get the gradient-corrected data at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the corrected data at (0-indexed).

        Returns
        -------
        data : ndarray, shape (n_channels, tr_spacing)
            The gradient-corrected data at the given TR.
        """
        detrended = self.get_tr_detrended(n)
        template = self.get_tr_template(n)
        return detrended - template

    def correct(self):
        """Generate the gradient-corrected data.

        Returns
        -------
        corrected : ndarray, shape (n_channels, n_times)
            The gradient-corrected data.
        """
        corrected = self._data.copy()
        for tr in range(self.n_tr):
            this_start, this_end = self._tr_bounds(tr)
            corrected[:, this_start:this_end] = self.get_tr_corrected(tr)
        self._corrected = corrected
        return corrected

    @staticmethod
    def _valid_window(window):
        if isinstance(window, int):
            if not window % 2 == 0:
                raise ValueError(f"Integer windows must be even (received {window}).")
            window = (window // 2, window // 2)
        elif isinstance(window, tuple):
            if not len(window) == 2:
                raise ValueError(
                    f"Tuple windows must contain 2 elements (received {window})."
                )
        else:
            raise TypeError(
                "Window must be a positive, even integer or a tuple of size 2 "
                f"containing a positive integer. (Received {window})."
            )
        if window[0] < 0 or window[1] < 0:
            raise ValueError(
                f"Window must contain a positive integer. (Received {window})."
            )
        if window[0] == 0 and window[1] == 0:
            raise ValueError(
                f"Window must contain a positive integer. (Received {window})."
            )
        return window

    @staticmethod
    def _valid_tr_events(tr_events, tr_tol=0):
        # Check to make sure TRs are (nearly) evenly spaced
        if len(tr_events.shape) == 2:
            if tr_events.shape[1] == 3:
                tr_events = tr_events[:, 0]
            else:
                raise ValueError(
                    "TRs must be a 1D array or a (N, 3) ndarray from mne. "
                    f"Received array of shape {tr_events.shape}."
                )
        elif len(tr_events.shape) != 1:
            raise ValueError(
                "TRs must be a 1D array or a (N, 3) ndarray from mne. "
                f"Received array of shape {tr_events.shape}."
            )
        if tr_tol < 0:
            raise ValueError(
                f"tr_tol must be a non-negative integer (received {tr_tol})."
            )
        diffs = np.diff(tr_events)
        median_spacing = np.median(diffs)
        deviation = np.abs(diffs - median_spacing)
        if np.any(deviation > tr_tol):
            unique = np.unique(diffs)
            raise ValueError(
                "TR spacings are not consistent (median spacing "
                f"{median_spacing}, tolerance {tr_tol} samples); the "
                f"following unique distances were present: {unique}."
            )
        return tr_events

    def _check_valid_tr(self, n):
        if n < 0 or n >= self.n_tr:
            raise ValueError(f"Index {n} not in TR range [0, {self.n_tr - 1}]")

    def _tr_bounds(self, n):
        self._check_valid_tr(n)
        # Anchor each TR at its own detected onset sample (rather than
        # extrapolating from tr_events[0] and a fixed spacing) so that small,
        # per-TR jitter in the trigger timing does not accumulate across TRs.
        this_start = self._tr_events[n]
        this_end = this_start + self.tr_spacing
        return (this_start, this_end)


@verbose
def remove_fmri_gradient_artifact(
    raw,
    tr_events,
    *,
    window=(4, 4),
    tr_tol=0,
    picks=None,
    method="AAS",
    copy=True,
    verbose=None,
):
    """Remove the fMRI gradient (imaging) artifact from EEG data.

    Removes the gradient artifact present in EEG recorded simultaneously with
    functional MRI. The default method is average artifact subtraction (AAS)
    :footcite:`AllenEtAl2000`, which builds an artifact template for each
    imaging volume (TR) by averaging neighboring TR epochs and subtracts it.

    Parameters
    ----------
    raw : instance of Raw
        The raw data recorded during MRI acquisition. Must be preloaded.
    tr_events : ndarray, shape (n_trs,) | (n_trs, 3)
        The sample numbers at which TRs (imaging volumes) begin. May be a 1D
        array of sample numbers, or an ``(n_trs, 3)`` events array as returned
        by :func:`mne.find_events` (the first column is used). TRs must be
        evenly spaced in time, within ``tr_tol`` samples (see below).
    window : int | tuple of int
        The number of neighboring TRs to average into each template. Either an
        even integer giving the total number of TRs (split evenly before and
        after), or a ``(n_before, n_after)`` tuple. Default ``(4, 4)``.
    tr_tol : int
        The maximum allowed deviation (in samples) of any individual TR
        spacing from the median TR spacing, to tolerate jitter in detected TR
        onset times. Default 0.
    %(picks_all_data_noref)s
    method : str
        The artifact removal method. Currently only ``'AAS'`` (average artifact
        subtraction) is supported. Default ``'AAS'``.
    copy : bool
        If True (default), operate on and return a copy of ``raw``. If False,
        modify ``raw`` in place.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The raw data with the gradient artifact removed.

    Notes
    -----
    .. versionadded:: 1.13

    References
    ----------
    .. footbibliography::
    """
    _validate_type(copy, bool, "copy")
    _check_option("method", method, ("AAS",))
    _validate_type(tr_events, np.ndarray, "tr_events")

    if not raw.preload:
        raise RuntimeError(
            "raw data must be preloaded to remove the gradient artifact, use "
            "raw.load_data() or preload=True when reading the data."
        )

    picks = _picks_to_idx(raw.info, picks, none="data", exclude="bads")

    if copy:
        raw = raw.copy()

    data = raw.get_data(picks=picks)
    remover = GradientRemover(data, tr_events, window=window, tr_tol=tr_tol)
    raw._data[picks] = remover.correct()

    return raw
