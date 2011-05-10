import os.path as op

from numpy.testing import assert_array_almost_equal, assert_equal

from .. import read_evoked, write_evoked

fname = op.join(op.dirname(__file__), 'data', 'test-ave.fif')

def test_io_evoked():
    """Test IO for evoked data
    """
    ave = read_evoked(fname)

    ave.crop(tmin=0)

    write_evoked('evoked.fif', ave)
    ave2 = read_evoked('evoked.fif')

    assert_array_almost_equal(ave.data, ave2.data)
    assert_array_almost_equal(ave.times, ave2.times)
    assert_equal(ave.nave, ave2.nave)
    assert_equal(ave.aspect_kind, ave2.aspect_kind)
    assert_equal(ave.last, ave2.last)
    assert_equal(ave.first, ave2.first)
