"""Parallel util function
"""

# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: Simplified BSD

from warnings import warn


def parallel_func(func, n_jobs, temp_folder=None, max_nbytes=1e6, verbose=5):
    """Return parallel instance with delayed function

    Util function to use joblib only if available

    Parameters
    ----------
    func: callable
        A function
    n_jobs: int
        Number of jobs to run in parallel
    temp_folder: str, optional
        Folder to be used by the pool for memmaping large numpy
        arrays for sharing memory with worker processes. If
        None, memory sharing is disabled.
    max_nbytes int or None, optional, 1e6 (1MB) by default
        Threshold on the size of arrays passed to the workers that
        triggers automated memmory mapping in temp_folder.
        If None, memory sharing is diabled
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

    if joblib_mmap:
        kwargs['temp_folder'] = temp_folder
        if temp_folder is None:
            # we only use memmapping if temp_folder has been specified
            kwargs['max_nbytes'] = None
        else:
            kwargs['max_nbytes'] = max_nbytes
    else:
        if temp_folder is not None:
            warn('joblib is not new enough to support memmapping pool. '
                 'Update joblib or use temp_folder=None')

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
