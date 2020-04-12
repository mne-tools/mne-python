from copy import deepcopy

from mne.defaults import _handle_default


def test_handle_default():
    """Test mutable default."""
    x = deepcopy(_handle_default('scalings'))
    y = _handle_default('scalings')
    z = _handle_default('scalings', dict(mag=1, grad=2))
    w = _handle_default('scalings', {})
    assert set(x.keys()) == set(y.keys())
    assert set(x.keys()) == set(z.keys())
    for key in x.keys():
        assert x[key] == y[key]
        assert x[key] == w[key]
        if key in ('mag', 'grad'):
            assert x[key] != z[key]
        else:
            assert x[key] == z[key]
