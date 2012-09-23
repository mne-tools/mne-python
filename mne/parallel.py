"""Parallel util function
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

from warnings import warn

from . import get_cache_dir


def parallel_func(func, n_jobs, max_nbytes=1e6, verbose=5):
    """Return parallel instance with delayed function

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel
    max_nbytes int or None, optional, 1e6 (1MB) by default
        Threshold on the size of arrays passed to the workers that
        triggers automated memmory mapping. If None, memory sharing
        is diabled.
    verbose: int
        Verbosity level

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

    # check if joblib is new enough to support memmapping pool
    import inspect
    aspec = inspect.getargspec(Parallel.__init__)
    joblib_mmap = ('temp_folder' in aspec.args and 'max_nbytes' in aspec.args)

    # create keyword arguments for Parallel
    kwargs = {'verbose': verbose}

    # get the cache directory
    cache_dir = get_cache_dir()

    if joblib_mmap:
        kwargs['temp_folder'] = cache_dir
        if cache_dir is None:
            # we only use memmapping if temp_dir has been set
            kwargs['max_nbytes'] = None
        else:
            kwargs['max_nbytes'] = max_nbytes
    else:
        if cache_dir is not None:
            warn('joblib is not new enough to support memmapping pool. '
                 'Update joblib to use this feature.')

    parallel = Parallel(n_jobs, **kwargs)

    my_func = delayed(func)

    if n_jobs == -1:
        try:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        except ImportError:
            print "multiprocessing not installed. Cannot run in parallel."
            n_jobs = 1

    return parallel, my_func, n_jobs
