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

    if n_jobs == -1:
        try:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        except ImportError:
            logger.warn('multiprocessing not installed. Cannot run in '
                         'parallel.')
            n_jobs = 1

    return parallel, my_func, n_jobs
