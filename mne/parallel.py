"""Parallel util function."""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import logging
import os

from . import get_config
from .utils import logger, verbose, warn, ProgressBar
from .fixes import _get_args

if 'MNE_FORCE_SERIAL' in os.environ:
    _force_serial = True
else:
    _force_serial = None


@verbose
def parallel_func(func, n_jobs, max_nbytes='auto', pre_dispatch='n_jobs',
                  total=None, prefer=None, verbose=None):
    """Return parallel instance with delayed function.

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel
    max_nbytes : int, str, or None
        Threshold on the minimum size of arrays passed to the workers that
        triggers automated memory mapping. Can be an int in Bytes,
        or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmaping of large arrays. Use 'auto' to
        use the value set using mne.set_memmap_min_size.
    pre_dispatch : int, or string, optional
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
    %(verbose)s INFO or DEBUG
        will print parallel status, others will not.

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object
    my_func: callable
        func if not parallel or delayed(func)
    n_jobs: int
        Number of jobs >= 0
    """
    should_print = (logger.level <= logging.INFO)
    # for a single job, we don't need joblib
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            try:
                from sklearn.externals.joblib import Parallel, delayed
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

        n_jobs = check_n_jobs(n_jobs)
        parallel = _check_wrapper(Parallel(n_jobs, **kwargs))
        my_func = delayed(func)

    if total is not None:
        def parallel_progress(op_iter):
            pb = ProgressBar(total, verbose_bool=should_print)
            return parallel(pb(op_iter))
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


def check_n_jobs(n_jobs, allow_cuda=False):
    """Check n_jobs in particular for negative values.

    Parameters
    ----------
    n_jobs : int
        The number of jobs.
    allow_cuda : bool
        Allow n_jobs to be 'cuda'. Default: False.

    Returns
    -------
    n_jobs : int
        The checked number of jobs. Always positive (or 'cuda' if
        applicable.)
    """
    if not isinstance(n_jobs, int):
        if not allow_cuda:
            raise ValueError('n_jobs must be an integer')
        elif not isinstance(n_jobs, str) or n_jobs != 'cuda':
            raise ValueError('n_jobs must be an integer, or "cuda"')
        # else, we have n_jobs='cuda' and this is okay, so do nothing
    elif _force_serial:
        n_jobs = 1
        logger.info('... MNE_FORCE_SERIAL set. Processing in forced '
                    'serial mode.')
    elif n_jobs <= 0:
        try:
            import multiprocessing
            n_cores = multiprocessing.cpu_count()
            n_jobs = min(n_cores + n_jobs + 1, n_cores)
            if n_jobs <= 0:
                raise ValueError('If n_jobs has a negative value it must not '
                                 'be less than the number of CPUs present. '
                                 'You\'ve got %s CPUs' % n_cores)
        except ImportError:
            # only warn if they tried to use something other than 1 job
            if n_jobs != 1:
                warn('multiprocessing not installed. Cannot run in parallel.')
                n_jobs = 1

    return n_jobs
