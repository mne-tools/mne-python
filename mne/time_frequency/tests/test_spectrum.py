import numpy as np
import pytest


def test_spectrum_errors(raw):
    """Test for expected errors in the .compute_psd() method."""
    with pytest.raises(ValueError, match='must not exceed ½ the sampling'):
        raw.compute_psd(fmax=raw.info['sfreq'] * 0.51)
    with pytest.raises(TypeError, match='unexpected keyword argument foo for'):
        raw.compute_psd(foo=None)
    with pytest.raises(TypeError, match='keyword arguments foo, bar for'):
        raw.compute_psd(foo=None, bar=None)


@pytest.mark.parametrize('method', ('welch', 'multitaper'))
@pytest.mark.parametrize(
    ('fmin, fmax, tmin, tmax, picks, proj, n_fft, n_overlap, n_per_seg, '
     'average, window, bandwidth, adaptive, low_bias, normalization'),
    [[0, np.inf, None, None, None, False, 256, 0, None,
      'mean', 'hamming', None, False, True, 'length'],  # defaults
     [5, 50, 1, 6, 'grad', True, 128, 8, 32,
      'median', 'triang', 10, True, False, 'full']  # non-defaults
     ]
)
def test_spectrum_params(method, fmin, fmax, tmin, tmax, picks, proj, n_fft,
                         n_overlap, n_per_seg, average, window, bandwidth,
                         adaptive, low_bias, normalization, raw, epochs):
    """Test valid parameter combinations in the .compute_psd() method."""
    kwargs = dict(method=method, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                  picks=picks, proj=proj)
    if method == 'welch':
        kwargs.update(n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                      average=average, window=window)
    else:
        kwargs.update(bandwidth=bandwidth, adaptive=adaptive,
                      low_bias=low_bias, normalization=normalization)
    # test with Raw
    raw.compute_psd(**kwargs)
