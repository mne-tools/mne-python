# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2014 SCoT Development Team


def parallel_loop(func, n_jobs=1, verbose=1):
    """run loops in parallel, if joblib is available.

    Parameters
    ----------
    func : function
        function to be executed in parallel
    n_jobs : int
        number of jobs
    verbose : int
        verbosity level
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        n_jobs = None

    if n_jobs is None:
        if verbose >= 10:
            print('running ', func, ' serially')
        par = lambda x: list(x)
    else:
        if verbose >= 10:
            print('running ', func, ' in parallel')
        func = delayed(func)
        par = Parallel(n_jobs=n_jobs, verbose=verbose)

    return par, func
