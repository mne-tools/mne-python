from nose.tools import assert_equal, assert_true
from copy import deepcopy

from mne.defaults import _handle_default


def test_handle_default():
    """Test mutable default
    """
    x = deepcopy(_handle_default('scalings'))
    y = _handle_default('scalings')
    z = _handle_default('scalings', dict(mag=1, grad=2))
    w = _handle_default('scalings', {})
    assert_equal(set(x.keys()), set(y.keys()))
    assert_equal(set(x.keys()), set(z.keys()))
    for key in x.keys():
        assert_equal(x[key], y[key])
        assert_equal(x[key], w[key])
        if key in ('mag', 'grad'):
            assert_true(x[key] != z[key])
        else:
            assert_equal(x[key], z[key])
