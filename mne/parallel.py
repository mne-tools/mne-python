"""Parallel util function
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

import logging
logger = logging.getLogger('mne')

from . import verbose


@verbose
def parallel_func(func, n_jobs, verbose=None):
    """Return parallel instance with delayed function

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        INFO or DEBUG will print parallel status, others will not.

    Returns
    -------
    parallel: instance of joblib.Parallel or list
        The parallel object
    my_func: callable
        func if not parallel or delayed(func)
    n_jobs: int
        Number of jobs >= 0
    """
    try:
        from sklearn.externals.joblib import Parallel, delayed
    except ImportError:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warn("joblib not installed. Cannot run in parallel.")
            n_jobs = 1
            my_func = func
            parallel = list
            return parallel, my_func, n_jobs

    parallel_verbose = 5 if logger.level <= logging.INFO else 0
    parallel = Parallel(n_jobs, verbose=parallel_verbose)
    my_func = delayed(func)
    n_jobs = check_n_jobs(n_jobs)
    return parallel, my_func, n_jobs


def check_n_jobs(n_jobs):
    """Check n_jobs in particular for negative values

    Parameters
    ----------
    n_jobs : int
        The number of jobs

    Returns
    -------
    n_jobs : int
        The checked number of jobs. Always positive.
    """
    try:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        if n_cores + n_jobs <= 0:
            raise ValueError('If n_jobs has a negative value it must not be less '
                             'than the number of CPUs present. You\'ve got '
                             '%s CPUs' % n_cores)
        n_jobs = n_cores + n_jobs
    except ImportError:
        logger.warn('multiprocessing not installed. Cannot run in '
                     'parallel.')
        n_jobs = 1
    return n_jobs
