"""Parallel util function."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import logging
import multiprocessing
import os

from . import get_config
from .utils import (logger, verbose, warn, ProgressBar, _validate_type,
                    _check_option, _ensure_int)
from .fixes import _get_args


@verbose
def parallel_func(func, n_jobs, max_nbytes='auto', pre_dispatch='n_jobs',
                  total=None, prefer=None, *, max_jobs=None, verbose=None):
    """Return parallel instance with delayed function.

    Util function to use joblib only if available

    Parameters
    ----------
    func : callable
        A function.
    %(n_jobs)s
    max_nbytes : int, str, or None
        Threshold on the minimum size of arrays passed to the workers that
        triggers automated memory mapping. Can be an int in Bytes,
        or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmaping of large arrays. Use 'auto' to
        use the value set using mne.set_memmap_min_size.
    pre_dispatch : int, or str, optional
        See :class:`joblib.Parallel`.
    total : int | None
        If int, use a progress bar to display the progress of dispatched
        jobs. This should only be used when directly iterating, not when
        using ``split_list`` or :func:`np.array_split`.
        If None (default), do not add a progress bar.
    prefer : str | None
        If str, can be "processes" or "threads". See :class:`joblib.Parallel`.
        Ignored if the joblib version is too old to support this.

        .. versionadded:: 0.18
    max_jobs : int | None
        The maximum number of jobs to use. This is useful when you know ahead
        of a the maximum number of calls into :class:`joblib.Parallel` that
        you will need, and the returned ``n_jobs`` should not exceed this
        value.
    %(verbose)s INFO or DEBUG
        will print parallel status, others will not.

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object.
    my_func: callable
        ``func`` if not parallel or delayed(func).
    n_jobs: int
        Number of jobs >= 0.
    """
    should_print = (logger.level <= logging.INFO)
    # for a single job, we don't need joblib
    if n_jobs is None:
        n_jobs = 1
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warn('joblib not installed. Cannot run in parallel.')
            n_jobs = 1
    if n_jobs == 1:
        n_jobs = 1
        my_func = func
        parallel = list
    else:
        # check if joblib is recent enough to support memmaping
        p_args = _get_args(Parallel.__init__)
        joblib_mmap = ('temp_folder' in p_args and 'max_nbytes' in p_args)

        cache_dir = get_config('MNE_CACHE_DIR', None)
        if isinstance(max_nbytes, str) and max_nbytes == 'auto':
            max_nbytes = get_config('MNE_MEMMAP_MIN_SIZE', None)

        if max_nbytes is not None:
            if not joblib_mmap and cache_dir is not None:
                warn('"MNE_CACHE_DIR" is set but a newer version of joblib is '
                     'needed to use the memmapping pool.')
            if joblib_mmap and cache_dir is None:
                logger.info(
                    'joblib supports memapping pool but "MNE_CACHE_DIR" '
                    'is not set in MNE-Python config. To enable it, use, '
                    'e.g., mne.set_cache_dir(\'/tmp/shm\'). This will '
                    'store temporary files under /dev/shm and can result '
                    'in large memory savings.')

        # create keyword arguments for Parallel
        kwargs = {'verbose': 5 if should_print and total is None else 0}
        kwargs['pre_dispatch'] = pre_dispatch
        if 'prefer' in p_args:
            kwargs['prefer'] = prefer

        if joblib_mmap:
            if cache_dir is None:
                max_nbytes = None  # disable memmaping
            kwargs['temp_folder'] = cache_dir
            kwargs['max_nbytes'] = max_nbytes

        n_jobs = check_n_jobs(n_jobs, max_jobs=max_jobs)
        parallel = _check_wrapper(Parallel(n_jobs, **kwargs))
        my_func = delayed(func)

    if total is not None:
        def parallel_progress(op_iter):
            return parallel(ProgressBar(iterable=op_iter, max_value=total))
        parallel_out = parallel_progress
    else:
        parallel_out = parallel
    return parallel_out, my_func, n_jobs


def _check_wrapper(fun):
    def run(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except RuntimeError as err:
            msg = str(err.args[0]) if err.args else ''
            if msg.startswith('The task could not be sent to the workers'):
                raise RuntimeError(
                    msg + ' Consider using joblib memmap caching to get '
                    'around this problem. See mne.set_mmap_min_size, '
                    'mne.set_cache_dir, and buffer_size parallel function '
                    'arguments (if applicable).')
            raise
    return run


def check_n_jobs(n_jobs, allow_cuda=False, *, max_jobs=None):
    """Check n_jobs in particular for negative values.

    Parameters
    ----------
    n_jobs : int
        The number of jobs.
    allow_cuda : bool
        Allow n_jobs to be 'cuda'. Default: False.
    max_jobs : int
        The maximum number of jobs to allow.

    Returns
    -------
    n_jobs : int
        The checked number of jobs. Always positive (or 'cuda' if
        applicable).
    """
    types = ('int-like', None)
    if allow_cuda:
        types = types + ('str',)
    _validate_type(n_jobs, types, 'n_jobs')
    if isinstance(n_jobs, str):
        # We can only be in this path if allow_cuda
        _check_option('n_jobs', n_jobs, ('cuda',), extra='when str')
        return 'cuda'  # return 'cuda'
    # From now on we're int | None
    if n_jobs is None:
        n_jobs = 1  # TODO actually do something better here
    n_jobs = _ensure_int(n_jobs, 'n_jobs')
    if os.getenv('MNE_FORCE_SERIAL', '').lower() in ('true', '1') and \
            n_jobs != 1:
        n_jobs = 1
        logger.info('... MNE_FORCE_SERIAL set. Processing in forced '
                    'serial mode.')
    elif n_jobs <= 0:
        n_cores = multiprocessing.cpu_count()
        n_jobs_orig = n_jobs
        n_jobs = min(n_cores + n_jobs + 1, n_cores)
        if n_jobs <= 0:
            raise ValueError(
                f'If n_jobs has a non-positive value ({n_jobs_orig}) it must '
                f'not be less than the number of CPUs present ({n_cores})')
    if max_jobs is not None:
        n_jobs = min(n_jobs, max(_ensure_int(max_jobs, 'max_jobs'), 1))
    # n_jobs is now an int and > 0
    return n_jobs
