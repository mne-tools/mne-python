from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose
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


def test_si_units():
    """Test that our scalings actually produce SI units."""
    scalings = _handle_default('scalings', None)
    units = _handle_default('units', None)
    # Add a bad one to test that we actually detect it
    assert 'csd_bad' not in scalings
    scalings['csd_bad'] = 1e5
    units['csd_bad'] = 'V/m²'
    assert set(scalings) == set(units)
    known_prefixes = {
        '': 1,
        'm': 1e-3,
        'c': 1e-2,
        'µ': 1e-6,
        'n': 1e-9,
        'f': 1e-15,
    }
    known_SI = {'V', 'T', 'Am', 'm', 'M', 'rad',
                'AU', 'GOF'}  # not really SI but we tolerate them
    powers = '²'

    def _split_si(x):
        if x == 'nAm':
            prefix, si = 'n', 'Am'
        elif x == 'GOF':
            prefix, si = '', 'GOF'
        elif x == 'AU':
            prefix, si = '', 'AU'
        elif x == 'rad':
            prefix, si = '', 'rad'
        elif len(x) == 2:
            if x[1] in powers:
                prefix, si = '', x
            else:
                prefix, si = x
        else:
            assert len(x) in (0, 1), x
            prefix, si = '', x
        return prefix, si

    for key, scale in scalings.items():
        unit = units[key]
        try:
            num, denom = unit.split('/')
        except ValueError:  # not enough to unpack
            num, denom = unit, ''
        # check the numerator and denominator
        num_prefix, num_SI = _split_si(num)
        assert num_prefix in known_prefixes
        assert num_SI in known_SI
        den_prefix, den_SI = _split_si(denom)
        assert den_prefix in known_prefixes
        if not (den_SI == den_prefix == ''):
            assert den_SI.strip(powers) in known_SI
        # reconstruct the scale factor
        want_scale = known_prefixes[den_prefix] / known_prefixes[num_prefix]
        if key == 'csd_bad':
            assert not np.isclose(scale, want_scale, rtol=10)
        else:
            assert_allclose(scale, want_scale, rtol=1e-12)
