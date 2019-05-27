# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

from collections.abc import Iterable
import time
import logging
import tempfile
import sys
import os.path as op
import os
import shutil

import numpy as np

from ._logging import logger
from .misc import sizeof_fmt


class ProgressBar(object):
    """Generate a command-line progressbar.

    Parameters
    ----------
    max_value : int | iterable
        Maximum value of process (e.g. number of samples to process, bytes to
        download, etc.). If an iterable is given, then `max_value` will be set
        to the length of this iterable.
    initial_value : int
        Initial value of process, useful when resuming process from a specific
        value, defaults to 0.
    mesg : str
        Message to include at end of progress bar.
    max_chars : int | str
        Number of characters to use for progress bar itself.
        This does not include characters used for the message or percent
        complete. Can be "auto" (default) to try to set a sane value based
        on the terminal width.
    progress_character : char
        Character in the progress bar that indicates the portion completed.
    spinner : bool
        Show a spinner.  Useful for long-running processes that may not
        increment the progress bar very often.  This provides the user with
        feedback that the progress has not stalled.
    max_total_width : int | str
        Maximum total message width. Can use "auto" (default) to try to set
        a sane value based on the current terminal width.
    verbose_bool : bool | 'auto'
        If True, show progress. 'auto' will use the current MNE verbose level.

    Example
    -------
    >>> progress = ProgressBar(13000)
    >>> progress.update(3000) # doctest: +SKIP
    [.........                               ] 23.07692 |
    >>> progress.update(6000) # doctest: +SKIP
    [..................                      ] 46.15385 |

    >>> progress = ProgressBar(13000, spinner=True)
    >>> progress.update(3000) # doctest: +SKIP
    [.........                               ] 23.07692 |
    >>> progress.update(6000) # doctest: +SKIP
    [..................                      ] 46.15385 /
    """

    spinner_symbols = ['|', '/', '-', '\\']
    template = '\r[{0}{1}] {2:6.02f}% {4} {3}   '

    def __init__(self, max_value, initial_value=0, mesg='', max_chars='auto',
                 progress_character='.', spinner=False,
                 max_total_width='auto', verbose_bool=True):  # noqa: D102
        self.cur_value = initial_value
        if isinstance(max_value, Iterable):
            self.max_value = len(max_value)
            self.iterable = max_value
        else:
            self.max_value = max_value
            self.iterable = None
        self.mesg = mesg
        self.progress_character = progress_character
        self.spinner = spinner
        self.spinner_index = 0
        self.n_spinner = len(self.spinner_symbols)
        if verbose_bool == 'auto':
            verbose_bool = True if logger.level <= logging.INFO else False
        self._do_print = verbose_bool
        self.cur_time = time.time()
        if max_total_width == 'auto':
            max_total_width = _get_terminal_width()
        self.max_total_width = int(max_total_width)
        if max_chars == 'auto':
            max_chars = min(max(max_total_width - 40, 10), 60)
        self.max_chars = int(max_chars)
        self.cur_rate = 0
        with tempfile.NamedTemporaryFile('wb', prefix='tmp_mne_prog') as tf:
            self._mmap_fname = tf.name
        del tf  # should remove the file
        self._mmap = None

    def update(self, cur_value, mesg=None):
        """Update progressbar with current value of process.

        Parameters
        ----------
        cur_value : number
            Current value of process.  Should be <= max_value (but this is not
            enforced).  The percent of the progressbar will be computed as
            (cur_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        cur_time = time.time()
        cur_rate = ((cur_value - self.cur_value) /
                    max(float(cur_time - self.cur_time), 1e-6))
        # Smooth the estimate a bit
        cur_rate = 0.1 * cur_rate + 0.9 * self.cur_rate
        # Ensure floating-point division so we can get fractions of a percent
        # for the progressbar.
        self.cur_time = cur_time
        self.cur_value = cur_value
        self.cur_rate = cur_rate
        max_value = float(self.max_value) if self.max_value else 1.
        progress = np.clip(self.cur_value / max_value, 0, 1)
        num_chars = int(progress * self.max_chars)
        num_left = self.max_chars - num_chars

        # Update the message
        if mesg is not None:
            if mesg == 'file_sizes':
                mesg = '(%s, %s/s)' % (
                    sizeof_fmt(self.cur_value).rjust(8),
                    sizeof_fmt(cur_rate).rjust(8))
            self.mesg = mesg

        # The \r tells the cursor to return to the beginning of the line rather
        # than starting a new line.  This allows us to have a progressbar-style
        # display in the console window.
        bar = self.template.format(self.progress_character * num_chars,
                                   ' ' * num_left,
                                   progress * 100,
                                   self.spinner_symbols[self.spinner_index],
                                   self.mesg)
        bar = bar[:self.max_total_width]
        # Force a flush because sometimes when using bash scripts and pipes,
        # the output is not printed until after the program exits.
        if self._do_print:
            sys.stdout.write(bar)
            sys.stdout.flush()
        # Increment the spinner
        if self.spinner:
            self.spinner_index = (self.spinner_index + 1) % self.n_spinner

    def update_with_increment_value(self, increment_value, mesg=None):
        """Update progressbar with an increment.

        Parameters
        ----------
        increment_value : int
            Value of the increment of process.  The percent of the progressbar
            will be computed as
            (self.cur_value + increment_value / max_value) * 100
        mesg : str
            Message to display to the right of the progressbar.  If None, the
            last message provided will be used.  To clear the current message,
            pass a null string, ''.
        """
        self.update(self.cur_value + increment_value, mesg)

    def __iter__(self):
        """Iterate to auto-increment the pbar with 1."""
        if self.iterable is None:
            raise ValueError("Must give an iterable to be used in a loop.")
        self.update(self.cur_value)
        for obj in self.iterable:
            yield obj
            self.update_with_increment_value(1)

    def __call__(self, seq):
        """Call the ProgressBar in a joblib-friendly way."""
        while True:
            try:
                yield next(seq)
            except StopIteration:
                return
            else:
                self.update_with_increment_value(1)

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
        if not self._do_print:
            return
        assert val is True
        self._mmap[idx] = True
        self.update(self._mmap.sum())

    def __enter__(self):  # noqa: D105
        if op.isfile(self._mmap_fname):
            os.remove(self._mmap_fname)
        # prevent corner cases where self.max_value == 0
        self._mmap = np.memmap(self._mmap_fname, bool, 'w+',
                               shape=max(self.max_value, 1))
        self.update(0)  # must be zero as we just created the memmap
        return self

    def __exit__(self, type, value, traceback):  # noqa: D105
        """Clean up memmapped file."""
        # we can't put this in __del__ b/c then each worker will delete the
        # file, which is not so good
        self._mmap = None
        if op.isfile(self._mmap_fname):
            os.remove(self._mmap_fname)
        self.done()

    def done(self):
        """Print a newline."""
        if self._do_print:
            sys.stdout.write('\n')
            sys.stdout.flush()


class _PBSubsetUpdater(object):

    def __init__(self, pb, idx):
        self.pb = pb
        self.idx = idx

    def update(self, ii):
        self.pb[self.idx[:ii]] = True


def _get_terminal_width():
    """Get the terminal width."""
    if sys.version[0] == '2':
        return 80
    else:
        return shutil.get_terminal_size((80, 20)).columns
