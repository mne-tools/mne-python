from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pandas import Series
from pandas.testing import assert_frame_equal

from mne.time_frequency.multitaper import _psd_from_mt


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
                         adaptive, low_bias, normalization, raw):
    """Test valid parameter combinations in the .compute_psd() method."""
    kwargs = dict(method=method, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                  picks=picks, proj=proj)
    if method == 'welch':
        kwargs.update(n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                      average=average, window=window)
    else:
        kwargs.update(bandwidth=bandwidth, adaptive=adaptive,
                      low_bias=low_bias, normalization=normalization)
    raw.compute_psd(**kwargs)


@pytest.mark.parametrize('long_format', (False, True))
def test_unaggregated_welch_spectrum_to_data_frame(raw, long_format):
    """Test converting unaggregated welch spectra to data frame."""
    # aggregated welch
    orig_df = raw.compute_psd().to_data_frame(long_format=long_format)
    # unaggregated welch → agg w/ pandas (make sure we did reshaping right)
    df = raw.compute_psd(average=False).to_data_frame(long_format=long_format)
    drop_cols = 'ch_type' if long_format else 'segment'
    group_by = ['freq', 'channel'] if long_format else 'freq'
    agg_df = (df.drop(columns=drop_cols)
                .groupby(group_by)
                .aggregate(np.nanmean)  # this is the psd_array_welch() default
                .reset_index())
    if long_format:
        agg_df.sort_values(by=group_by, inplace=True)
        orig_df.sort_values(by=group_by, inplace=True, ignore_index=True)
        orig_df.drop(columns=drop_cols, inplace=True)
    assert_frame_equal(agg_df, orig_df)


def _agg_helper(df, weights, group_cols):
    unagged_columns = df[group_cols].iloc[0].values.tolist()
    x_mt = df.drop(columns=group_cols).values[np.newaxis].T
    psd = _psd_from_mt(x_mt, weights)
    psd = np.atleast_1d(np.squeeze(psd)).tolist()
    _df = dict(zip(df.columns, unagged_columns + psd))
    return Series(_df)


@pytest.mark.parametrize('long_format', (False, True))
def test_unaggregated_multitaper_spectrum_to_data_frame(raw, long_format):
    """Test converting complex multitaper spectra to data frame."""
    # aggregated multitaper
    orig_df = (raw.compute_psd(method='multitaper')
                  .to_data_frame(long_format=long_format))
    # complex multitaper → aggr. w/ pandas (make sure we did reshaping right)
    spectrum = raw.compute_psd(method='multitaper', output='complex')
    df = spectrum.to_data_frame(long_format=long_format)
    group_by = ['freq']
    drop_cols = ['taper']
    if long_format:
        group_by.append('channel')
        drop_cols.append('ch_type')
        orig_df.drop(columns='ch_type', inplace=True)
    # only do a couple freq bins, otherwise test takes forever
    subset = partial(np.isin, test_elements=spectrum.freqs[:2])
    df = df.loc[subset(df['freq'])]
    orig_df = orig_df.loc[subset(orig_df['freq'])]
    # aggregate
    agg_df = (df.drop(columns=drop_cols)
                .groupby(group_by, sort=False, as_index=False)
                .apply(_agg_helper, spectrum._mt_weights, group_by))
    assert_frame_equal(agg_df, orig_df, check_categorical=False)


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
