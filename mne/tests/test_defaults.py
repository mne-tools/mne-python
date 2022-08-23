from copy import deepcopy

import pytest
from numpy.testing import assert_allclose
from mne.defaults import _handle_default
from mne.io.base import _get_scaling


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


def test_si_units():
    """Test that our scalings actually produce SI units."""
    scalings = _handle_default('scalings', None)
    units = _handle_default('units', None)
    # Add a bad one to test that we actually detect it
    assert 'csd_bad' not in scalings
    scalings['csd_bad'] = 1e5
    units['csd_bad'] = 'V/m²'
    assert set(scalings) == set(units)

    for key, scale in scalings.items():
        if key == 'csd_bad':
            with pytest.raises(KeyError, match='is not a channel type'):
                want_scale = _get_scaling(key, units[key])
        else:
            want_scale = _get_scaling(key, units[key])
            assert_allclose(scale, want_scale, rtol=1e-12)


@pytest.mark.parametrize('key', ('si_units', 'color', 'scalings',
                                 'scalings_plot_raw'))
def test_consistency(key):
    """Test defaults consistency."""
    units = set(_handle_default('units'))
    other = set(_handle_default(key))
    au_keys = set('stim exci syst resp ias chpi'.split())
    assert au_keys.intersection(units) == set()
    if key in ('color', 'scalings_plot_raw'):
        assert au_keys.issubset(other)
        other = other.difference(au_keys)
    else:
        assert au_keys.intersection(other) == set()
    assert units == other, key
