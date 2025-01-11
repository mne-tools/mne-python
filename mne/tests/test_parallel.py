# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import multiprocessing
import os
from contextlib import nullcontext

import pytest

from mne.parallel import parallel_func


@pytest.mark.parametrize(
    "n_jobs",
    [
        None,
        1,
        -1,
        "loky 2",
        "threading 3",
        "multiprocessing 4",
    ],
)
def test_parallel_func(n_jobs):
    """Test Parallel wrapping."""
    joblib = pytest.importorskip("joblib")
    if os.getenv("MNE_FORCE_SERIAL", "").lower() in ("true", "1"):
        pytest.skip("MNE_FORCE_SERIAL is set")

    def fun(x):
        return x * 2

    if isinstance(n_jobs, str):
        backend, n_jobs = n_jobs.split()
        n_jobs = want_jobs = int(n_jobs)
        try:
            func = joblib.parallel_config
        except AttributeError:
            # joblib < 1.3
            func = joblib.parallel_backend
        ctx = func(backend, n_jobs=n_jobs)
        n_jobs = None
    else:
        ctx = nullcontext()
        if n_jobs is not None and n_jobs < 0:
            want_jobs = multiprocessing.cpu_count() + 1 + n_jobs
        else:
            want_jobs = 1
    with ctx:
        parallel, p_fun, got_jobs = parallel_func(fun, n_jobs, verbose="debug")
    assert got_jobs == want_jobs
