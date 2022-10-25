from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne.time_frequency import read_spectrum
from mne.time_frequency.multitaper import _psd_from_mt
from mne.utils import requires_h5py, requires_pandas


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


def test_n_welch_windows(raw):
    """Test computation of welch windows https://mne.discourse.group/t/5734."""
    with raw.info._unlock():
        raw.info['sfreq'] = 999.412109375
    raw.compute_psd(
        method='welch', n_fft=999, n_per_seg=999, n_overlap=250, average=None)


def _get_inst(inst, request, evoked):
    # ↓ XXX workaround:
    # ↓ parametrized fixtures are not accessible via request.getfixturevalue
    # ↓ https://github.com/pytest-dev/pytest/issues/4666#issuecomment-456593913
    return evoked if inst == 'evoked' else request.getfixturevalue(inst)


@requires_h5py
@pytest.mark.parametrize('inst', ('raw', 'epochs', 'evoked'))
def test_spectrum_io(inst, tmp_path, request, evoked):
    """Test save/load of spectrum objects."""
    fname = tmp_path / f'{inst}-spectrum.h5'
    inst = _get_inst(inst, request, evoked)
    orig = inst.compute_psd()
    orig.save(fname)
    loaded = read_spectrum(fname)
    assert orig == loaded


def test_spectrum_copy(raw):
    """Test copying Spectrum objects."""
    spect = raw.compute_psd()
    spect_copy = spect.copy()
    assert spect == spect_copy
    assert id(spect) != id(spect_copy)
    spect_copy._freqs = None
    assert spect.freqs is not None


def test_spectrum_getitem_raw(raw):
    """Test Spectrum.__getitem__ for Raw-derived spectra."""
    spect = raw.compute_psd()
    want = spect.get_data(slice(1, 3), fmax=7)
    freq_idx = np.searchsorted(spect.freqs, 7)
    got = spect[1:3, :freq_idx]
    assert_array_equal(want, got)


def test_spectrum_getitem_epochs(epochs):
    """Test Spectrum.__getitem__ for Epochs-derived spectra."""
    spect = epochs.compute_psd()
    # testing data has just one epoch, its event_id label is "1"
    want = spect.get_data()
    got = spect['1'].get_data()
    assert_array_equal(want, got)


@pytest.mark.parametrize('method', ('mean', partial(np.std, axis=0)))
def test_epochs_spectrum_average(epochs, method):
    """Test EpochsSpectrum.average()."""
    spect = epochs.compute_psd()
    avg_spect = spect.average(method=method)
    assert avg_spect.shape == spect.shape[1:]
    assert avg_spect._dims == ('channel', 'freq')  # no 'epoch'


def _agg_helper(df, weights, group_cols):
    """Aggregate complex multitaper spectrum after conversion to DataFrame."""
    from pandas import Series

    unagged_columns = df[group_cols].iloc[0].values.tolist()
    x_mt = df.drop(columns=group_cols).values[np.newaxis].T
    psd = _psd_from_mt(x_mt, weights)
    psd = np.atleast_1d(np.squeeze(psd)).tolist()
    _df = dict(zip(df.columns, unagged_columns + psd))
    return Series(_df)


@requires_pandas
@pytest.mark.parametrize('long_format', (False, True))
@pytest.mark.parametrize('method', ('welch', 'multitaper'))
def test_unaggregated_spectrum_to_data_frame(raw, long_format, method):
    """Test converting complex multitaper spectra to data frame."""
    from pandas.testing import assert_frame_equal

    from mne.utils.dataframe import _inplace

    # aggregated spectrum → dataframe
    orig_df = (raw.compute_psd(method=method)
                  .to_data_frame(long_format=long_format))
    # unaggregated welch or complex multitaper →
    #   aggregate w/ pandas (to make sure we did reshaping right)
    kwargs = {'average': False} if method == 'welch' else {'output': 'complex'}
    spectrum = raw.compute_psd(method=method, **kwargs)
    df = spectrum.to_data_frame(long_format=long_format)
    grouping_cols = ['freq']
    drop_cols = ['segment'] if method == 'welch' else ['taper']
    if long_format:
        grouping_cols.append('channel')
        drop_cols.append('ch_type')
        orig_df.drop(columns='ch_type', inplace=True)
    # only do a couple freq bins, otherwise test takes forever for multitaper
    subset = partial(np.isin, test_elements=spectrum.freqs[:2])
    df = df.loc[subset(df['freq'])]
    orig_df = orig_df.loc[subset(orig_df['freq'])]
    # sort orig_df, because at present we can't actually prevent pandas from
    # sorting at the agg step *sigh*
    _inplace(orig_df, 'sort_values', by=grouping_cols, ignore_index=True)
    # aggregate
    gb = df.drop(columns=drop_cols).groupby(grouping_cols, as_index=False)
    if method == 'welch':
        agg_df = gb.aggregate(np.nanmean)
    else:
        agg_df = gb.apply(_agg_helper, spectrum._mt_weights, grouping_cols)
    # even with check_categorical=False, we know that the *data* matches;
    # what may differ is the order of the "levels" in the *metadata* for the
    # channel name column
    assert_frame_equal(agg_df, orig_df, check_categorical=False)


@requires_pandas
@pytest.mark.parametrize('inst', ('raw', 'epochs', 'evoked'))
def test_spectrum_to_data_frame(inst, request, evoked):
    """Test the to_data_frame method for Spectrum."""
    from pandas.testing import assert_frame_equal

    # setup
    is_epochs = inst == 'epochs'
    inst = _get_inst(inst, request, evoked)
    extra_dim = () if is_epochs else (1,)
    extra_cols = ['freq', 'condition', 'epoch'] if is_epochs else ['freq']
    # compute PSD
    spectrum = inst.compute_psd()
    n_epo, n_chan, n_freq = extra_dim + spectrum.get_data().shape
    # test wide format
    df_wide = spectrum.to_data_frame()
    n_row, n_col = df_wide.shape
    assert n_row == n_freq
    assert n_col == n_chan + len(extra_cols)
    assert set(spectrum.ch_names + extra_cols) == set(df_wide.columns)
    # test long format
    df_long = spectrum.to_data_frame(long_format=True)
    n_row, n_col = df_long.shape
    assert n_row == n_epo * n_freq * n_chan
    base_cols = ['channel', 'ch_type', 'value']
    assert n_col == len(base_cols + extra_cols)
    assert set(base_cols + extra_cols) == set(df_long.columns)
    # test index
    index = extra_cols[-2:]  # ['freq'] or ['condition', 'epoch']
    df = spectrum.to_data_frame(index=index)
    if is_epochs:
        index_tuple = (list(spectrum.event_id)[0],  # condition
                       spectrum.selection[0])       # epoch number
        subset = df.loc[index_tuple]
        assert subset.shape == (n_freq, n_chan + 1)  # + 1 is the freq column
    with pytest.raises(ValueError, match='"time" is not a valid option'):
        spectrum.to_data_frame(index='time')
    # test picks
    picks = [0, 1]
    _pick_first = spectrum.pick(picks).to_data_frame()
    _pick_last = spectrum.to_data_frame(picks=picks)
    assert_frame_equal(_pick_first, _pick_last)


# not testing with Evoked because it already has projs applied
@pytest.mark.parametrize('inst', ('raw', 'epochs'))
def test_spectrum_proj(inst, request):
    """Test that proj is applied correctly (gh 11177)."""
    inst = request.getfixturevalue(inst)
    has_proj = inst.compute_psd(proj=True)
    no_proj = inst.compute_psd(proj=False)
    assert not np.array_equal(has_proj.get_data(), no_proj.get_data())
    # make sure only the data (and the projs) were different
    has_proj._data = no_proj._data
    with has_proj.info._unlock():
        has_proj.info['projs'] = no_proj.info['projs']
    assert has_proj == no_proj
