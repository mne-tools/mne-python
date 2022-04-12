import mne
import numpy as np
from scipy.signal import detrend

class GradientRemover:
    """A class to remove gradients from EEG data using a template approach.
    """
    def __init__(self, eeg_data, tr_events, window=(4, 4)):
        """Constructor for the gradient remover

        Parameters
        ----------
        eeg_data: np.ndarray
            The raw EEG data to perform gradient correction on.
            Expected in shape (channels, time_points).
        tr_events: np.ndarray
            The sample numbers for when TRs begin. The array may be a
            subset of an mne find_events that is a shape (N, 3), or a
            1-dimensional array of sample numbers. TRs must be perfectly
            spaced in time.
        window: int or tuple[int, int]
            The window to use for templates. Must be either an even integer
            to indicate the total size, with an even number of TRs
            templated before and after, OR a tuple containing the number of
            TRs to use in the template before the current TR and then the
            number of TRs to use in the template after the current TR. For
            example, (4, 0) would use 4 TRs before the current and 0 after.
            Default (4, 4).

        Raises
        ------
        ValueError if any inputs are invalid.
        TypeError if any inputs are the wrong type.
        """
        self._window = GradientRemover._valid_window(window)
        self._tr_events = GradientRemover._valid_tr_events(tr_events)
        if self._tr_events[-1] > eeg_data.shape[1]:
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
        if self._corrected:
            return self._corrected
        else:
            return self.correct()
    @property
    def raw(self):
        return self._raw

    @property
    def window(self):
        return self._window

    @property
    def tr_spacing(self):
        return self._tr_events[1] - self._tr_events[0]

    @property
    def n_tr(self):
        return len(self._tr_events)

    @property
    def n_channels(self):
        return len(self._data)

    def get_tr(self, n):
        """Get the uncorrected data at a given TR

        Parameters
        ----------
        n: int
           The TR to get the uncorrected data at (0-indexed).

        Returns
        -------
        np.ndarray of shape (channels, tr_timepoints) representing the
        uncorrected data at the given TR.

        Raises
        ------
        ValueError if an invalid TR index is supplied.
        """
        this_start, this_end = self._tr_bounds(n)
        return self._data[:, this_start:this_end]

    def get_tr_detrended(self, n):
        """Get the detrended data at a given TR

        Parameters
        ----------
        n: int
           The TR to get the detrended data at (0-indexed).

        Returns
        -------
        np.ndarray of shape (channels, tr_timepoints) representing the
        detrended data at the given TR.

        Raises
        ------
        ValueError if an invalid TR index is supplied.
        """
        return detrend(self.get_tr(n))

    def get_tr_template(self, n):
        """Get the gradient template data at a given TR

        Parameters
        ----------
        n: int
           The TR to get the template data at (0-indexed).

        Returns
        -------
        np.ndarray of shape (channels, tr_timepoints) representing the
        template data at the given TR.

        Raises
        ------
        ValueError if an invalid TR index is supplied.
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
            np.asarray(
                [self.get_tr_detrended(tr) for tr in range(start, stop)]
            ),
            axis=0
        )

    def get_tr_corrected(self, n):
        """Get the gradient-corrected data at a given TR

        Parameters
        ----------
        n: int
           The TR to get the corrected data at (0-indexed).

        Returns
        -------
        np.ndarray of shape (channels, tr_timepoints) representing the
        template data at the given TR.

        Raises
        ------
        ValueError if an invalid TR index is supplied.
        """
        detrended = self.get_tr_detrended(n)
        template = self.get_tr_template(n)
        return detrended - template

    def correct(self):
        """Generate the gradient-corrected data."""
        corrected = self._data.copy()
        for tr in range(self.n_tr):
            this_start, this_end = self._tr_bounds(tr)
            corrected[:, this_start:this_end] = self.get_tr_corrected(tr)
        self._corrected = corrected
        return corrected

    def _valid_window(window):
        if isinstance(window, int):
            if not window % 2 == 0:
                raise ValueError(
                    f"Integer windows must be even (received {window})."
                )
            window = (window // 2, window // 2)
        elif isinstance(window, tuple):
            if not len(window) == 2:
                raise ValueError(
                    "Tuple windows must contain 2 elements "
                    f"(received {window})."
                )
        else:
            raise TypeError(
                "Window must be a positive, even integer or a tuple of "
                "size 2 containing a positive integer."
                "(Received {window})."
            )
        if window[0] < 0 or window[1] < 0:
            raise ValueError(
                "Window must contain a positive integer. "
                f"(Received {window})."
            )
        if window[0] == 0 and window[1] == 0:
            raise ValueError(
                "Window must contain a positive integer. "
                f"(Received {window})."
            )
        return window

    def _valid_tr_events(tr_events):
        # Check to make sure TRs are evenly spaced
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
        unique = np.unique(np.diff(tr_events))
        if len(unique) != 1:
            raise ValueError(
                "TR spacings are not consistent; the following unique "
                f"distances were present: {unique}."
            )
        return tr_events

    def _check_valid_tr(self, n):
        if n < 0 or n >= self.n_tr:
            raise ValueError(
                f"Index {n} not in TR range [0, {self.n_tr - 1}]"
            )

    def _tr_bounds(self, n):
        self._check_valid_tr(n)
        offset = self._tr_events[0]
        length = self.tr_spacing
        this_start = offset + n * length
        this_end = offset + (n + 1) * length
        return (this_start, this_end)
