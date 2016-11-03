"""Parallel util function."""

# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

from .externals.six import string_types
import logging
import os

from . import get_config
from .utils import logger, verbose, warn
from .fixes import _get_args

if 'MNE_FORCE_SERIAL' in os.environ:
    _force_serial = True
else:
    _force_serial = None


@verbose
def parallel_func(func, n_jobs, verbose=None, max_nbytes='auto'):
    """Return parallel instance with delayed function.

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). INFO or DEBUG
        will print parallel status, others will not.
    max_nbytes : int, str, or None
        Threshold on the minimum size of arrays passed to the workers that
        triggers automated memmory mapping. Can be an int in Bytes,
        or a human-readable string, e.g., '1M' for 1 megabyte.
        Use None to disable memmaping of large arrays. Use 'auto' to
        use the value set using mne.set_memmap_min_size.

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object
    my_func: callable
        func if not parallel or delayed(func)
    n_jobs: int
        Number of jobs >= 0
    """
    # for a single job, we don't need joblib
    if n_jobs == 1:
        n_jobs = 1
        my_func = func
        parallel = list
        return parallel, my_func, n_jobs

    try:
        from joblib import Parallel, delayed
    except ImportError:
        try:
            from sklearn.externals.joblib import Parallel, delayed
        except ImportError:
            warn('joblib not installed. Cannot run in parallel.')
            n_jobs = 1
            my_func = func
            parallel = list
            return parallel, my_func, n_jobs

    # check if joblib is recent enough to support memmaping
    p_args = _get_args(Parallel.__init__)
    joblib_mmap = ('temp_folder' in p_args and 'max_nbytes' in p_args)

    cache_dir = get_config('MNE_CACHE_DIR', None)
    if isinstance(max_nbytes, string_types) and max_nbytes == 'auto':
        max_nbytes = get_config('MNE_MEMMAP_MIN_SIZE', None)

    if max_nbytes is not None:
        if not joblib_mmap and cache_dir is not None:
            warn('"MNE_CACHE_DIR" is set but a newer version of joblib is '
                 'needed to use the memmapping pool.')
        if joblib_mmap and cache_dir is None:
            logger.info('joblib supports memapping pool but "MNE_CACHE_DIR" '
                        'is not set in MNE-Python config. To enable it, use, '
                        'e.g., mne.set_cache_dir(\'/tmp/shm\'). This will '
                        'store temporary files under /dev/shm and can result '
                        'in large memory savings.')

    # create keyword arguments for Parallel
    kwargs = {'verbose': 5 if logger.level <= logging.INFO else 0}

    if joblib_mmap:
        if cache_dir is None:
            max_nbytes = None  # disable memmaping
        kwargs['temp_folder'] = cache_dir
        kwargs['max_nbytes'] = max_nbytes

    n_jobs = check_n_jobs(n_jobs)
    parallel = Parallel(n_jobs, **kwargs)
    my_func = delayed(func)
    return parallel, my_func, n_jobs


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
        elif not isinstance(n_jobs, string_types) or n_jobs != 'cuda':
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
