import numpy as np
from numpy.testing import assert_allclose
import pytest

from mne._ola import _COLA, _Interp2, _Storer


def test_interp_2pt():
    """Test our two-point interpolator."""
    n_pts = 200
    assert n_pts % 50 == 0
    feeds = [  # test a bunch of feeds to make sure they don't break things
        [n_pts],
        [50] * (n_pts // 50),
        [10] * (n_pts // 10),
        [5] * (n_pts // 5),
        [2] * (n_pts // 2),
        [1] * n_pts,
    ]

    # ZOH
    values = np.array([10, -10])
    expected = np.full(n_pts, 10)
    for feed in feeds:
        expected[-1] = 10
        interp = _Interp2([0, n_pts], values, 'zero')
        out = np.concatenate([interp.feed(f)[0] for f in feed])
        assert_allclose(out, expected)
        interp = _Interp2([0, n_pts - 1], values, 'zero')
        expected[-1] = -10
        out = np.concatenate([interp.feed(f)[0] for f in feed])
        assert_allclose(out, expected)

    # linear and inputs of different sizes
    values = [np.arange(2)[:, np.newaxis, np.newaxis], np.array([20, 10])]
    expected = [
        np.linspace(0, 1, n_pts, endpoint=False)[np.newaxis, np.newaxis, :],
        np.linspace(20, 10, n_pts, endpoint=False)]
    for feed in feeds:
        interp = _Interp2([0, n_pts], values, 'linear')
        outs = [interp.feed(f) for f in feed]
        outs = [np.concatenate([o[0] for o in outs], axis=-1),
                np.concatenate([o[1] for o in outs], axis=-1)]
        assert_allclose(outs[0], expected[0], atol=1e-7)
        assert_allclose(outs[1], expected[1], atol=1e-7)

    # cos**2 and more interesting bounds
    values = np.array([10, -10])
    expected = np.full(n_pts, 10.)
    expected[-5:] = -10
    cos = np.cos(np.linspace(0, np.pi / 2., n_pts - 9,
                             endpoint=False))
    expected[4:-5] = cos ** 2 * 20 - 10
    for feed in feeds:
        interp = _Interp2([4, n_pts - 5], values, 'cos2')
        out = np.concatenate([interp.feed(f)[0] for f in feed])
        assert_allclose(out, expected, atol=1e-7)
    out = interp.feed(10)[0]
    assert_allclose(out, [values[-1]] * 10, atol=1e-7)

    # hann and broadcasting
    n_hann = n_pts - 9
    expected[4:-5] = np.hanning(2 * n_hann + 1)[n_hann:-1] * 20 - 10
    expected = np.array([expected, expected[::-1] * 0.5])
    values = np.array([values, values[::-1] * 0.5]).T
    for feed in feeds:
        interp = _Interp2([4, n_pts - 5], values, 'hann')
        out = np.concatenate([interp.feed(f)[0] for f in feed], axis=-1)
        assert_allclose(out, expected, atol=1e-7)

    # one control point and None support
    values = [np.array([10]), None]
    for start in [0, 50, 99, 100, 1000]:
        interp = _Interp2([start], values, 'zero')
        out, none = interp.feed(n_pts)
        assert none is None
        expected = np.full(n_pts, 10.)
        assert_allclose(out, expected)


@pytest.mark.parametrize('ndim', (1, 2, 3))
def test_cola(ndim):
    """Test COLA processing."""
    sfreq = 1000.
    rng = np.random.RandomState(0)

    def processor(x):
        return (x / 2.,)  # halve the signal

    for n_total in (999, 1000, 1001):
        signal = rng.randn(n_total)
        out = rng.randn(n_total)  # shouldn't matter
        for _ in range(ndim - 1):
            signal = signal[np.newaxis]
            out = out[np.newaxis]
        for n_samples in (99, 100, 101, 102,
                          n_total - n_total // 2 + 1, n_total):
            for window in ('hann', 'bartlett', 'boxcar', 'triang'):
                # A few example COLA possibilities
                n_overlaps = ()
                if window in ('hann', 'bartlett') or n_samples % 2 == 0:
                    n_overlaps += ((n_samples + 1) // 2,)
                if window == 'boxcar':
                    n_overlaps += (0,)
                for n_overlap in n_overlaps:
                    # can pass callable or ndarray
                    for storer in (out, _Storer(out)):
                        cola = _COLA(processor, storer, n_total, n_samples,
                                     n_overlap, sfreq, window)
                        n_input = 0
                        # feed data in an annoying way
                        while n_input < n_total:
                            next_len = min(rng.randint(1, 30),
                                           n_total - n_input)
                            cola.feed(signal[..., n_input:n_input + next_len])
                            n_input += next_len
                        assert_allclose(out, signal / 2., atol=1e-7)
