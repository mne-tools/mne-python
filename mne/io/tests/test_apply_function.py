# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray
from mne.utils import logger, catch_logging, run_tests_if_main


def bad_1(x):
    """Fail."""
    return  # bad return type


def bad_2(x):
    """Fail."""
    return x[:-1]  # bad shape


def printer(x):
    """Print."""
    logger.info('exec')
    return x


@pytest.mark.slowtest
def test_apply_function_verbose():
    """Test apply function verbosity."""
    n_chan = 2
    n_times = 3
    ch_names = [str(ii) for ii in range(n_chan)]
    raw = RawArray(np.zeros((n_chan, n_times)),
                   create_info(ch_names, 1., 'mag'))
    # test return types in both code paths (parallel / 1 job)
    pytest.raises(TypeError, raw.apply_function, bad_1)
    pytest.raises(ValueError, raw.apply_function, bad_2)
    pytest.raises(TypeError, raw.apply_function, bad_1, n_jobs=2)
    pytest.raises(ValueError, raw.apply_function, bad_2, n_jobs=2)

    # check our arguments
    with catch_logging() as sio:
        out = raw.apply_function(printer, verbose=False)
        assert len(sio.getvalue()) == 0
        assert out is raw
        raw.apply_function(printer, verbose=True)
        assert sio.getvalue().count('\n') == n_chan


run_tests_if_main()
