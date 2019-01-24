# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import inspect
import re
import sys
import logging
import os.path as op
import warnings


logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


class WrapStdOut(object):
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name):  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux)
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError("'file' object has not attribute '%s'" % name)


def warn(message, category=RuntimeWarning, module='mne'):
    """Emit a warning with trace outside the mne namespace.

    This function takes arguments like warnings.warn, and sends messages
    using both ``warnings.warn`` and ``logger.warn``. Warnings can be
    generated deep within nested function calls. In order to provide a
    more helpful warning, this function traverses the stack until it
    reaches a frame outside the ``mne`` namespace that caused the error.

    Parameters
    ----------
    message : str
        Warning message.
    category : instance of Warning
        The warning class. Defaults to ``RuntimeWarning``.
    module : str
        The name of the module emitting the warning.
    """
    root_dir = op.dirname(__file__)
    frame = None
    if logger.level <= logging.WARN:
        last_fname = ''
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_code.co_filename
            lineno = frame.f_lineno
            # in verbose dec
            if fname == '<string>' and last_fname == 'utils.py':
                last_fname = fname
                frame = frame.f_back
                continue
            # treat tests as scripts
            # and don't capture unittest/case.py (assert_raises)
            if not (fname.startswith(root_dir) or
                    ('unittest' in fname and 'case' in fname)) or \
                    op.basename(op.dirname(fname)) == 'tests':
                break
            last_fname = op.basename(fname)
            frame = frame.f_back
        del frame
        # We need to use this instead of warn(message, category, stacklevel)
        # because we move out of the MNE stack, so warnings won't properly
        # recognize the module name (and our warnings.simplefilter will fail)
        warnings.warn_explicit(
            message, category, fname, lineno, module,
            globals().get('__warningregistry__', {}))
    # To avoid a duplicate warning print, we only emit the logger.warning if
    # one of the handlers is a FileHandler. See gh-5592
    if any(isinstance(h, logging.FileHandler) or getattr(h, '_mne_file_like',
                                                         False)
           for h in logger.handlers):
        logger.warning(message)


def filter_out_warnings(warn_record, category=None, match=None):
    r"""Remove particular records from ``warn_record``.

    This helper takes a list of :class:`warnings.WarningMessage` objects,
    and remove those matching category and/or text.

    Parameters
    ----------
    category: WarningMessage type | None
       class of the message to filter out

    match : str | None
        text or regex that matches the error message to filter out

    Examples
    --------
    This can be used as::

        >>> import pytest
        >>> import warnings
        >>> from mne.utils import filter_out_warnings
        >>> with pytest.warns(None) as recwarn:
        ...     warnings.warn("value must be 0 or None", UserWarning)
        >>> filter_out_warnings(recwarn, match=".* 0 or None")
        >>> assert len(recwarn.list) == 0

        >>> with pytest.warns(None) as recwarn:
        ...     warnings.warn("value must be 42", UserWarning)
        >>> filter_out_warnings(recwarn, match=r'.* must be \d+$')
        >>> assert len(recwarn.list) == 0

        >>> with pytest.warns(None) as recwarn:
        ...     warnings.warn("this is not here", UserWarning)
        >>> filter_out_warnings(recwarn, match=r'.* must be \d+$')
        >>> assert len(recwarn.list) == 1
    """
    regexp = re.compile('.*' if match is None else match)
    is_category = [w.category == category if category is not None else True
                   for w in warn_record._list]
    is_match = [regexp.match(w.message.args[0]) is not None
                for w in warn_record._list]
    ind = [ind for ind, (c, m) in enumerate(zip(is_category, is_match))
           if c and m]

    for i in reversed(ind):
        warn_record._list.pop(i)
