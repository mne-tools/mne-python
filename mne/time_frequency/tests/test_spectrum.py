import numpy as np
from numpy.testing import assert_array_equal
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
    # TODO test with Epochs
    # epochs.compute_psd(**kwargs)


def test_spectrum_to_data_frame(raw):
    """Test the to_data_frame method for Spectrum."""
    spectrum = raw.compute_psd()
    n_chan, n_freq = spectrum.get_data().shape
    # test wide format
    df_wide = spectrum.to_data_frame()
    n_row, n_col = df_wide.shape
    assert n_row == n_freq
    assert n_col == n_chan + 1  # freq column
    # test long format
    df_long = spectrum.to_data_frame(long_format=True)
    n_row, n_col = df_long.shape
    assert n_row == n_freq * n_chan
    assert n_col == 4  # freq, ch_name, ch_type, value
    # test index
    _ = spectrum.to_data_frame(index='freq')
    with pytest.raises(ValueError, match='"time" is not a valid option'):
        spectrum.to_data_frame(index='time')
    # test picks
    picks = [0, 1]
    _pick_first = spectrum.pick(picks).to_data_frame()
    _pick_last = spectrum.to_data_frame(picks=picks)
    assert_array_equal(_pick_first, _pick_last)


def test_epoch_spectrum_to_data_frame(epochs):
    """Test the to_data_frame method for Spectrum."""
    spectrum = epochs.compute_psd()
    n_epo, n_chan, n_freq = spectrum.get_data().shape
    # test wide format
    df_wide = spectrum.to_data_frame()
    n_row, n_col = df_wide.shape
    assert n_row == n_freq * n_epo
    assert n_col == n_chan + 3  # freq, condition, epoch
    # test long format
    df_long = spectrum.to_data_frame(long_format=True)
    n_row, n_col = df_long.shape
    assert n_row == n_freq * n_epo * n_chan
    assert n_col == 6  # freq, cond, epo, ch_name, ch_type, value
    # test index
    df_idx = spectrum.to_data_frame(index=['epoch', 'condition'])
    subset = df_idx.loc[(0, 'auditory/right')]
    assert subset.shape == (n_freq, n_chan + 1)  # the + 1 is the freq column
    # test picks
    picks = [0, 1]
    _pick_first = spectrum.pick(picks).to_data_frame()
    _pick_last = spectrum.to_data_frame(picks=picks)
    assert_array_equal(_pick_first, _pick_last)
