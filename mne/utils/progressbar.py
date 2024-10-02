"""Some utility functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import logging
import os
import os.path as op
import tempfile
import time
from collections.abc import Iterable
from threading import Thread

import numpy as np

from ._logging import logger
from .check import _check_option
from .config import get_config


class ProgressBar:
    """Generate a command-line progressbar.

    Parameters
    ----------
    iterable : iterable | int | None
        The iterable to use. Can also be an int for backward compatibility
        (acts like ``max_value``).
    initial_value : int
        Initial value of process, useful when resuming process from a specific
        value, defaults to 0.
    mesg : str
        Message to include at end of progress bar.
    max_total_width : int | str
        Maximum total message width. Can use "auto" (default) to try to set
        a sane value based on the current terminal width.
    max_value : int | None
        The max value. If None, the length of ``iterable`` will be used.
    which_tqdm : str | None
        Which tqdm module to use. Can be "tqdm", "tqdm.notebook", or "off".
        Defaults to ``None``, which uses the value of the MNE_TQDM environment
        variable, or ``"tqdm.auto"`` if that is not set.
    **kwargs : dict
        Additional keyword arguments for tqdm.
    """

    def __init__(
        self,
        iterable=None,
        initial_value=0,
        mesg=None,
        max_total_width="auto",
        max_value=None,
        *,
        which_tqdm=None,
        **kwargs,
    ):
        # The following mimics this, but with configurable module to use
        # from ..externals.tqdm import auto
        import tqdm

        if which_tqdm is None:
            which_tqdm = get_config("MNE_TQDM", "tqdm.auto")
        _check_option(
            "MNE_TQDM", which_tqdm[:5], ("tqdm", "tqdm.", "off"), extra="beginning"
        )
        logger.debug(f"Using ProgressBar with {which_tqdm}")
        if which_tqdm not in ("tqdm", "off"):
            try:
                __import__(which_tqdm)
            except Exception as exc:
                raise ValueError(
                    f"Unknown tqdm backend {repr(which_tqdm)}, got: {exc}"
                ) from None
            tqdm = getattr(tqdm, which_tqdm.split(".", 1)[1])
        tqdm = tqdm.tqdm
        defaults = dict(
            leave=True,
            mininterval=0.016,
            miniters=1,
            smoothing=0.05,
            bar_format="{percentage:3.0f}%|{bar}| {desc} : {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt:>11}{postfix}]",  # noqa: E501
        )
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs.update({key: val})
        if isinstance(iterable, Iterable):
            self.iterable = iterable
            if max_value is None:
                self.max_value = len(iterable)
            else:
                self.max_value = max_value
        else:  # ignore max_value then
            self.max_value = int(iterable)
            self.iterable = None
        if max_total_width == "auto":
            max_total_width = None  # tqdm's auto
        with tempfile.NamedTemporaryFile("wb", prefix="tmp_mne_prog") as tf:
            self._mmap_fname = tf.name
        del tf  # should remove the file
        self._mmap = None
        disable = logger.level > logging.INFO or which_tqdm == "off"
        self._tqdm = tqdm(
            iterable=self.iterable,
            desc=mesg,
            total=self.max_value,
            initial=initial_value,
            ncols=max_total_width,
            disable=disable,
            **kwargs,
        )

    def update(self, cur_value):
        """Update progressbar with current value of process.

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            ``(cur_value / max_value) * 100``.
        """
        self.update_with_increment_value(cur_value - self._tqdm.n)

    def update_with_increment_value(self, increment_value):
        """Update progressbar with an increment.

        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            ``(self.cur_value + increment_value / max_value) * 100``.
        """
        try:
            self._tqdm.update(increment_value)
        except TypeError:  # can happen during GC on Windows
            pass

    def __iter__(self):
        """Iterate to auto-increment the pbar with 1."""
        yield from self._tqdm

    def subset(self, idx):
        """Make a joblib-friendly index subset updater.

        Parameters
        ----------
        idx : ndarray
            List of indices for this subset.

        Returns
        -------
        updater : instance of PBSubsetUpdater
            Class with a ``.update(ii)`` method.
        """
        return _PBSubsetUpdater(self, idx)

    def __enter__(self):  # noqa: D105
        # This should only be used with pb.subset and parallelization
        if op.isfile(self._mmap_fname):
            os.remove(self._mmap_fname)
        # prevent corner cases where self.max_value == 0
        self._mmap = np.memmap(
            self._mmap_fname, bool, "w+", shape=max(self.max_value, 1)
        )
        self.update(0)  # must be zero as we just created the memmap

        # We need to control how the pickled bars exit: remove print statements
        self._thread = _UpdateThread(self)
        self._thread.start()
        return self

    def __exit__(self, type_, value, traceback):  # noqa: D105
        # Restore exit behavior for our one from the main thread
        self.update(self._mmap.sum())
        self._tqdm.close()
        self._thread._mne_run = False
        self._thread.join()
        self._mmap = None
        if op.isfile(self._mmap_fname):
            try:
                os.remove(self._mmap_fname)
            # happens on Windows sometimes
            except PermissionError:  # pragma: no cover
                pass

    def __del__(self):
        """Ensure output completes."""
        if getattr(self, "_tqdm", None) is not None:
            self._tqdm.close()


class _UpdateThread(Thread):
    def __init__(self, pb):
        super().__init__(daemon=True)
        self._mne_run = True
        self._mne_pb = pb

    def run(self):
        while self._mne_run:
            self._mne_pb.update(self._mne_pb._mmap.sum())
            time.sleep(1.0 / 30.0)  # 30 Hz refresh is plenty


class _PBSubsetUpdater:
    def __init__(self, pb, idx):
        self.mmap = pb._mmap
        self.idx = idx

    def update(self, ii):
        self.mmap[self.idx[ii - 1]] = True
