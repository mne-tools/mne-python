import os.path as op
import numpy as np
from nose.tools import assert_true, assert_equal
from numpy.testing import assert_allclose
import warnings

from mne import read_dip, read_dipoles, write_dipoles
from mne.datasets import testing
from mne.utils import run_tests_if_main, _TempDir

warnings.simplefilter('always')
data_path = testing.data_path(download=False)
fname_dip = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')


def _compare_dipoles(orig, new):
    for key, value in orig.items():
        assert_true(key in new)
        if isinstance(value, np.ndarray):
            assert_allclose(value, new[key], err_msg='Mismatch for %s' % key)
        else:
            assert_equal(value, new[key])
    assert_equal(set(orig.keys()), set(new.keys()))


@testing.requires_testing_data
def test_io_dipoles():
    """Test IO for .dip files
    """
    tempdir = _TempDir()
    out_fname = op.join(tempdir, 'temp.dip')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        time, pos, amplitude, ori, gof = read_dip(fname_dip)
    assert_true(len(w) >= 1)

    assert_true(pos.shape[1] == 3)
    assert_true(ori.shape[1] == 3)
    assert_true(len(time) == len(pos))
    assert_true(len(time) == gof.size)
    assert_true(len(time) == amplitude.size)

    dipoles = dict(time=time, pos=pos, amplitude=amplitude, ori=ori, gof=gof,
                   name='ALL')
    write_dipoles(out_fname, dipoles)
    dipoles_new = read_dipoles(out_fname)
    _compare_dipoles(dipoles, dipoles_new)

run_tests_if_main(False)
