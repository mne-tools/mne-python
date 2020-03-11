# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from collections.abc import Iterable
import os
import os.path as op
import tempfile
from threading import RLock

import numpy as np


def _noop():
    pass


class ProgressBar(object):
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
    disable : bool | None
        If True, disable the bar. If None, autodetect based on TTY.
    max_value : int | None
        The max value. If None, the length of ``iterable`` will be used.
    **kwargs : dict
        Additional keyword arguments for tqdm.
    """

    def __init__(self, iterable=None, initial_value=0, mesg=None,
                 max_total_width='auto', disable=None, max_value=None,
                 **kwargs):  # noqa: D102
        from ..externals.tqdm import tqdm  # currently 4.40.2
        defaults = dict(
            leave=True, mininterval=0.016, miniters=0, smoothing=0.9,
            bar_format='{percentage:3.0f}%|{bar}| {desc} : {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt:>11}{postfix}]',  # noqa: E501
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
        if max_total_width == 'auto':
            max_total_width = None  # tqdm's auto
        with tempfile.NamedTemporaryFile('wb', prefix='tmp_mne_prog') as tf:
            self._mmap_fname = tf.name
        del tf  # should remove the file
        self._mmap = None
        self._tqdm = tqdm(
            iterable=self.iterable, desc=mesg, total=self.max_value,
            initial=initial_value, disable=disable, ncols=max_total_width,
            **kwargs)
        self._tqdm.set_lock(RLock())

    def update(self, cur_value, mesg=None):
        """Update progressbar with current value of process.

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            ``(cur_value / max_value) * 100``.
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        with self._tqdm.get_lock():
            self.update_with_increment_value(cur_value - self._tqdm.n, mesg)

    def update_with_increment_value(self, increment_value, mesg=None):
        """Update progressbar with an increment.

        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            ``(self.cur_value + increment_value / max_value) * 100``.
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        self._tqdm.update(increment_value)
        if mesg is not None:
            self._tqdm.set_description(mesg)

    def __iter__(self):
        """Iterate to auto-increment the pbar with 1."""
        return iter(self._tqdm)

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

    def __setitem__(self, idx, val):
        """Use alternative, mmap-based incrementing (max_value must be int)."""
        assert val is True
        self._mmap[idx] = True
        self.update(self._mmap.sum())

    def __enter__(self):  # noqa: D105
        # This should only be used with pb.subset and parallelization
        if op.isfile(self._mmap_fname):
            os.remove(self._mmap_fname)
        # prevent corner cases where self.max_value == 0
        self._mmap = np.memmap(self._mmap_fname, bool, 'w+',
                               shape=max(self.max_value, 1))
        self.update(0)  # must be zero as we just created the memmap

        # We need to control how the pickled bars exit: remove print statements
        self._tqdm_close = self._tqdm.close
        self._tqdm.close = _noop
        return self

    def __exit__(self, type_, value, traceback):  # noqa: D105
        # Restore exit behavior for our one from the main thread
        self._tqdm.close = self._tqdm_close
        self.update(self._mmap.sum())
        self._tqdm.close()
        self._mmap = None
        if op.isfile(self._mmap_fname):
            os.remove(self._mmap_fname)


class _PBSubsetUpdater(object):

    def __init__(self, pb, idx):
        self.pb = pb
        self.idx = idx

    def update(self, ii):
        self.pb[self.idx[:ii]] = True
