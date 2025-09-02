# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
from functools import partial

import numpy as np
import pytest
from matplotlib.colors import same_color
from numpy.testing import assert_allclose, assert_array_equal

from mne import Annotations, BaseEpochs, create_info, make_fixed_length_epochs
from mne.io import RawArray
from mne.time_frequency import read_spectrum
from mne.time_frequency.multitaper import _psd_from_mt
from mne.time_frequency.spectrum import (
    EpochsSpectrumArray,
    SpectrumArray,
    combine_spectrum,
)
from mne.utils import _record_warnings


def test_compute_psd_errors(raw):
    """Test for expected errors in the .compute_psd() method."""
    with pytest.raises(ValueError, match="must not exceed ½ the sampling"):
        raw.compute_psd(fmax=raw.info["sfreq"] * 0.51)
    with pytest.raises(TypeError, match="unexpected keyword argument foo for"):
        raw.compute_psd(foo=None)
    with pytest.raises(TypeError, match="keyword arguments foo, bar for"):
        raw.compute_psd(foo=None, bar=None)
    raw.set_annotations(Annotations(onset=0.01, duration=0.01, description="bad_foo"))
    with pytest.raises(NotImplementedError, match='Cannot use method="multitaper"'):
        raw.compute_psd(method="multitaper", reject_by_annotation=True)


@pytest.mark.parametrize("method", ("welch", "multitaper"))
@pytest.mark.parametrize(
    (
        "fmin, fmax, tmin, tmax, picks, proj, n_fft, n_overlap, n_per_seg, "
        "average, window, bandwidth, adaptive, low_bias, normalization, output"
    ),
    [
        [
            0,
            np.inf,
            None,
            None,
            None,
            False,
            256,
            0,
            None,
            "mean",
            "hamming",
            None,
            False,
            True,
            "length",
            "power",
        ],  # defaults
        [
            5,
            50,
            1,
            6,
            "grad",
            True,
            128,
            8,
            32,
            "median",
            "triang",
            10,
            True,
            False,
            "full",
            "power",
        ],  # non-defaults (excluding output)
        [
            0,
            np.inf,
            None,
            None,
            None,
            False,
            256,
            0,
            None,
            "mean",
            "hamming",
            None,
            False,
            True,
            "length",
            "complex",
        ],  # complex (testing with non-defaults doesn't increase coverage)
    ],
)
def test_spectrum_params(
    method,
    fmin,
    fmax,
    tmin,
    tmax,
    picks,
    proj,
    n_fft,
    n_overlap,
    n_per_seg,
    average,
    window,
    bandwidth,
    adaptive,
    low_bias,
    normalization,
    output,
    raw,
):
    """Test valid parameter combinations in the .compute_psd() method."""
    kwargs = dict(
        method=method,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        proj=proj,
        output=output,
    )
    if method == "welch":
        kwargs.update(
            n_fft=n_fft,
            n_overlap=n_overlap,
            n_per_seg=n_per_seg,
            average=average,
            window=window,
        )
    else:
        kwargs.update(
            bandwidth=bandwidth,
            adaptive=adaptive,
            low_bias=low_bias,
            normalization=normalization,
        )
    raw.compute_psd(**kwargs)


def test_n_welch_windows(raw):
    """Test computation of welch windows https://mne.discourse.group/t/5734."""
    with raw.info._unlock():
        raw.info["sfreq"] = 999.412109375
    raw.compute_psd(
        method="welch", n_fft=999, n_per_seg=999, n_overlap=250, average=None
    )


def _get_inst(inst, request, *, evoked=None, average_tfr=None):
    # ↓ XXX workaround:
    # ↓ parametrized fixtures are not accessible via request.getfixturevalue
    # ↓ https://github.com/pytest-dev/pytest/issues/4666#issuecomment-456593913
    if inst == "evoked":
        return evoked
    elif inst == "average_tfr":
        return average_tfr
    return request.getfixturevalue(inst)


@pytest.mark.parametrize("inst", ("raw", "epochs_full", "evoked"))
def test_spectrum_io(inst, tmp_path, request, evoked):
    """Test save/load of spectrum objects."""
    pytest.importorskip("h5io")
    fname = tmp_path / f"{inst}-spectrum.h5"
    inst = _get_inst(inst, request, evoked=evoked)
    if isinstance(inst, BaseEpochs):
        # fake HED-like tags (https://mne.discourse.group/t/10634)
        inst.events[-2:, -1] = 2
        inst.event_id = {"foo/bar": 1, "foo/qux": 2}
    orig = inst.compute_psd()
    if isinstance(inst, BaseEpochs):
        orig = orig["foo"]
    orig.save(fname)
    loaded = read_spectrum(fname)
    assert orig == loaded


def test_spectrum_copy(raw_spectrum):
    """Test copying Spectrum objects."""
    spect_copy = raw_spectrum.copy()
    assert raw_spectrum == spect_copy
    assert id(raw_spectrum) != id(spect_copy)
    spect_copy._freqs = None
    assert raw_spectrum.freqs is not None


@pytest.mark.parametrize("weights", ["nave", "equal", [1, -1]])
def test_combine_spectrum(raw_spectrum, weights):
    """Test `combine_spectrum()` works."""
    spectrum1 = raw_spectrum.copy()
    spectrum2 = raw_spectrum.copy()
    if weights == "nave":
        spectrum1._nave = 1
        spectrum2._nave = 2
        spectrum2._data *= 2
        new_spectrum = combine_spectrum([spectrum1, spectrum2], weights=weights)
        assert_allclose(new_spectrum.data, spectrum1.data * (5 / 3))
    elif weights == "equal":
        spectrum2._data *= 2
        new_spectrum = combine_spectrum([spectrum1, spectrum2], weights=weights)
        assert_allclose(new_spectrum.data, spectrum1.data * 1.5)
    else:
        new_spectrum = combine_spectrum([spectrum1, spectrum2], weights=weights)
        assert_allclose(new_spectrum.data, 0)


def test_combine_spectrum_error_catch(raw_spectrum):
    """Test `combine_spectrum()` catches errors."""
    # Test bad weights
    with pytest.raises(
        ValueError, match='Weights must be a list of float, or "nave" or "equal"'
    ):
        combine_spectrum([raw_spectrum, raw_spectrum], weights="foo")
    with pytest.raises(
        ValueError, match="Weights must be the same size as all_spectrum"
    ):
        combine_spectrum([raw_spectrum, raw_spectrum], weights=[1, 1, 1])

    # Test bad nave
    with pytest.raises(ValueError, match="The 'nave' attribute is not specified"):
        combine_spectrum([raw_spectrum, raw_spectrum], weights="nave")

    # Test inconsistent channels
    raw_spectrum2 = raw_spectrum.copy()
    raw_spectrum2.drop_channels(raw_spectrum2.ch_names[0])
    with pytest.raises(AssertionError, match=".* do not contain the same channels"):
        combine_spectrum([raw_spectrum, raw_spectrum2], weights="equal")

    # Test inconsistent frequencies
    raw_spectrum2 = raw_spectrum.copy()
    raw_spectrum2._freqs = raw_spectrum2._freqs + 1
    with pytest.raises(AssertionError, match=".* do not contain the same frequencies"):
        combine_spectrum([raw_spectrum, raw_spectrum2], weights="equal")


def test_spectrum_reject_by_annot(raw):
    """Test rejecting by annotation.

    Cannot use raw_spectrum fixture here because we're testing reject_by_annotation in
    .compute_psd() method.
    """
    kw = dict(n_per_seg=512)  # smaller than shortest good span, to avoid warning
    spect_no_annot = raw.compute_psd(**kw)
    raw.set_annotations(Annotations([1, 5], [3, 3], ["test", "test"]))
    spect_benign_annot = raw.compute_psd(**kw)
    raw.annotations.description = np.array(["bad_test", "bad_test"])
    spect_reject_annot = raw.compute_psd(**kw)
    spect_ignored_annot = raw.compute_psd(**kw, reject_by_annotation=False)
    # the only one that should be different is `spect_reject_annot`
    assert spect_no_annot == spect_benign_annot
    assert spect_no_annot == spect_ignored_annot
    assert spect_no_annot != spect_reject_annot


def test_spectrum_bads_exclude(raw):
    """Test bads are not removed unless exclude="bads"."""
    raw.pick("mag")  # get rid of IAS channel
    spect_no_excld = raw.compute_psd()
    spect_with_excld = raw.compute_psd(exclude="bads")
    assert raw.info["bads"] == spect_no_excld.info["bads"]
    assert spect_with_excld.info["bads"] == []
    assert set(raw.ch_names) - set(spect_with_excld.ch_names) == set(raw.info["bads"])


def test_spectrum_getitem_raw(raw_spectrum):
    """Test Spectrum.__getitem__ for Raw-derived spectra."""
    want = raw_spectrum.get_data(slice(1, 3), fmax=7)
    freq_idx = np.searchsorted(raw_spectrum.freqs, 7)
    got = raw_spectrum[1:3, :freq_idx]
    assert_array_equal(want, got)


def test_spectrum_getitem_epochs(epochs_spectrum):
    """Test Spectrum.__getitem__ for Epochs-derived spectra."""
    # testing data has just one epoch, its event_id label is "1"
    want = epochs_spectrum.get_data()
    got = epochs_spectrum["1"].get_data()
    assert_array_equal(want, got)


@pytest.mark.parametrize("method", ("mean", partial(np.std, axis=0)))
def test_epochs_spectrum_average(epochs_spectrum, method):
    """Test EpochsSpectrum.average()."""
    avg_spect = epochs_spectrum.average(method=method)
    assert avg_spect.shape == epochs_spectrum.shape[1:]
    assert avg_spect._dims == ("channel", "freq")  # no 'epoch'


@pytest.mark.parametrize("inst", ("raw_spectrum", "epochs_spectrum", "evoked"))
def test_spectrum_to_data_frame(inst, request, evoked):
    """Test the to_data_frame method for Spectrum."""
    pytest.importorskip("pandas")
    from pandas.testing import assert_frame_equal

    # setup
    is_already_psd = inst in ("raw_spectrum", "epochs_spectrum")
    is_epochs = inst == "epochs_spectrum"
    inst = _get_inst(inst, request, evoked=evoked)
    extra_dim = () if is_epochs else (1,)
    extra_cols = ["freq", "condition", "epoch"] if is_epochs else ["freq"]
    # compute PSD
    spectrum = inst if is_already_psd else inst.compute_psd(exclude="bads")
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
    base_cols = ["channel", "ch_type", "value"]
    assert n_col == len(base_cols + extra_cols)
    assert set(base_cols + extra_cols) == set(df_long.columns)
    # test index
    index = extra_cols[-2:]  # ['freq'] or ['condition', 'epoch']
    df = spectrum.to_data_frame(index=index)
    if is_epochs:
        index_tuple = (
            list(spectrum.event_id)[0],  # condition
            spectrum.selection[0],
        )  # epoch number
        subset = df.loc[index_tuple]
        assert subset.shape == (n_freq, n_chan + 1)  # + 1 is the freq column
    with pytest.raises(ValueError, match='"time" is not a valid option'):
        spectrum.to_data_frame(index="time")
    # test picks
    picks = [0, 1]
    _pick_first = spectrum.pick(picks).to_data_frame()
    _pick_last = spectrum.to_data_frame(picks=picks)
    assert_frame_equal(_pick_first, _pick_last)


def _agg_helper(df, weights, group_cols):
    """Aggregate complex multitaper spectrum after conversion to DataFrame."""
    from pandas import Series

    unagged_columns = df[group_cols].iloc[0].values.tolist()
    x_mt = df.drop(columns=group_cols).values[np.newaxis].T
    psd = _psd_from_mt(x_mt, weights)
    psd = np.atleast_1d(np.squeeze(psd)).tolist()
    _df = dict(zip(df.columns, unagged_columns + psd))
    return Series(_df)


@pytest.mark.parametrize("long_format", (False, True))
@pytest.mark.parametrize(
    "method, output",
    [("welch", "complex"), ("welch", "power"), ("multitaper", "complex")],
)
def test_unaggregated_spectrum_to_data_frame(raw, long_format, method, output):
    """Test converting unaggregated spectra (multiple segments/tapers) to data frame."""
    pytest.importorskip("pandas")
    from pandas.testing import assert_frame_equal

    from mne.utils.dataframe import _inplace

    # aggregated spectrum → dataframe
    orig_df = raw.compute_psd(method=method).to_data_frame(long_format=long_format)
    # unaggregated welch or complex multitaper →
    #   aggregate w/ pandas (to make sure we did reshaping right)
    kwargs = dict()
    if method == "welch":
        kwargs.update(average=False)
    spectrum = raw.compute_psd(method=method, output=output, **kwargs)
    df = spectrum.to_data_frame(long_format=long_format)
    grouping_cols = ["freq"]
    drop_cols = ["segment"] if method == "welch" else ["taper"]
    if long_format:
        grouping_cols.append("channel")
        drop_cols.append("ch_type")
        orig_df.drop(columns="ch_type", inplace=True)
    # only do a couple freq bins, otherwise test takes forever for multitaper
    subset = partial(np.isin, test_elements=spectrum.freqs[:2])
    df = df.loc[subset(df["freq"])]
    orig_df = orig_df.loc[subset(orig_df["freq"])]
    # sort orig_df, because at present we can't actually prevent pandas from
    # sorting at the agg step *sigh*
    _inplace(orig_df, "sort_values", by=grouping_cols, ignore_index=True)
    # aggregate
    df = df.drop(columns=drop_cols)
    gb = df.groupby(grouping_cols, as_index=False, observed=False)
    if method == "welch":
        if output == "complex":

            def _fun(x):
                return np.mean(np.real(x * np.conj(x)))  # use mean to aggregate

            agg_df = gb.agg(_fun)
        else:
            agg_df = gb.mean()  # excludes missing values itself
    else:
        gb = gb[df.columns]  # XXX: try removing when minimum pandas >= 2.1 is required
        agg_df = gb.apply(_agg_helper, spectrum.weights, grouping_cols)
    # even with check_categorical=False, we know that the *data* matches;
    # what may differ is the order of the "levels" in the *metadata* for the
    # channel name column
    agg_df.sort_values(by=grouping_cols, ignore_index=True, inplace=True)
    orig_df.sort_values(by=grouping_cols, ignore_index=True, inplace=True)
    # One can have categorical dtype and the other plain object, so don't check that
    assert_frame_equal(agg_df, orig_df, check_categorical=False, check_dtype=False)


# not testing with Evoked because it already has projs applied
@pytest.mark.parametrize("inst", ("raw", "epochs"))
def test_spectrum_proj(inst, request):
    """Test that proj is applied correctly (gh 11177)."""
    inst = request.getfixturevalue(inst)
    has_proj = inst.compute_psd(proj=True)
    no_proj = inst.compute_psd(proj=False)
    assert not np.array_equal(has_proj.get_data(), no_proj.get_data())
    # make sure only the data (and the projs) were different
    has_proj._data = no_proj._data
    with has_proj.info._unlock():
        has_proj.info["projs"] = no_proj.info["projs"]
    assert has_proj == no_proj


@pytest.mark.parametrize(
    "method, average", [("welch", False), ("welch", "mean"), ("multitaper", None)]
)
def test_spectrum_complex(method, average):
    """Test output='complex' support."""
    sfreq = 100
    n = 10 * sfreq
    freq = 3.0
    phase = np.pi / 4  # should be recoverable
    data = np.cos(2 * np.pi * freq * np.arange(n) / sfreq + phase)[np.newaxis]
    raw = RawArray(data, create_info(1, sfreq, "eeg"))
    epochs = make_fixed_length_epochs(raw, duration=2.0, preload=True)
    assert len(epochs) == 5
    assert len(epochs.times) == 2 * sfreq
    kwargs = dict(output="complex", method=method)
    if method == "welch":
        kwargs["n_fft"] = sfreq
        want_dims = ("epoch", "channel", "freq")
        want_shape = (5, 1, sfreq // 2 + 1)
        if not average:
            want_dims = want_dims + ("segment",)
            want_shape = want_shape + (2,)
            kwargs["average"] = average
    else:
        assert method == "multitaper"
        assert not average
        want_dims = ("epoch", "channel", "taper", "freq")
        want_shape = (5, 1, 7, sfreq + 1)
    spectrum = epochs.compute_psd(**kwargs)
    idx = np.argmin(np.abs(spectrum.freqs - freq))
    assert spectrum.freqs[idx] == freq
    assert spectrum._dims == want_dims
    assert spectrum.shape == want_shape
    data = spectrum.get_data()
    assert data.dtype == np.complex128
    coef = spectrum.get_data(fmin=freq, fmax=freq).mean(0)
    if method == "multitaper":
        coef = coef[..., 0, :]  # first taper
    elif not average:
        coef = coef.mean(-1)  # over segments
    coef = coef.item()
    # Test phase matches what was simulated
    assert_allclose(np.angle(coef), phase, rtol=1e-4)
    # Now test that it warns appropriately
    epochs._data[0, 0, :] = 0  # actually zero for one epoch and ch
    with pytest.warns(UserWarning, match="Zero value.*channel 0"):
        epochs.compute_psd(**kwargs)
    # But not if we mark that channel as bad
    epochs.info["bads"] = epochs.ch_names[:1]
    epochs.compute_psd(**kwargs)


def test_spectrum_kwarg_triaging(raw):
    """Test kwarg triaging in legacy plot_psd() method."""
    import matplotlib.pyplot as plt

    regex = r"legacy plot_psd\(\) method.*unexpected keyword.*'axes'.*Try rewriting"
    _, axes = plt.subplots(1, 2)
    # `axes` is the new param name: technically only valid for Spectrum.plot()
    with _record_warnings(), pytest.warns(RuntimeWarning, match=regex):
        raw.plot_psd(axes=axes)
    # `ax` is the correct legacy param name
    raw.plot_psd(ax=axes)


def _check_spectrum_equivalent(spect1, spect2, tmp_path):
    data1 = spect1.get_data()
    data2 = spect2.get_data()
    assert_array_equal(data1, data2)
    assert_array_equal(spect1.freqs, spect2.freqs)


def test_spectrum_array_errors():
    """Test (Epochs)SpectrumArray constructor errors."""
    n_epochs = 10
    n_chans = 5
    n_freqs = 50
    freqs = np.arange(n_freqs)
    sfreq = 100
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_freqs))
    dim_names = ("epoch", "channel", "freq")
    info = create_info(n_chans, sfreq, "eeg")
    # test incorrect ndims (for SpectrumArray; allows 2-3D data)
    with pytest.raises(ValueError, match="Invalid value for the 'data.ndim' parameter"):
        SpectrumArray(data[0, 0, :], info, freqs, dim_names=dim_names)
    with pytest.raises(ValueError, match="Invalid value for the 'data.ndim' parameter"):
        SpectrumArray(np.expand_dims(data, axis=3), info, freqs, dim_names=dim_names)
    # test incorrect ndims (for EpochsSpectrumArray; allows 3-4D data)
    with pytest.raises(ValueError, match="Invalid value for the 'data.ndim' parameter"):
        EpochsSpectrumArray(data[0, :, :], info, freqs, dim_names=dim_names)
    with pytest.raises(ValueError, match="Invalid value for the 'data.ndim' parameter"):
        EpochsSpectrumArray(
            np.expand_dims(data, axis=(3, 4)), info, freqs, dim_names=dim_names
        )
    # test incorrect epochs location
    with pytest.raises(ValueError, match="'epoch' must be the first dimension"):
        EpochsSpectrumArray(data, info, freqs, dim_names=("channel", "epoch", "freq"))
    # test mismatching events shape
    events = np.vstack(
        (
            np.arange(n_epochs + 1),
            np.zeros(n_epochs + 1, dtype=int),
            np.ones(n_epochs + 1, dtype=int),
        )
    ).T
    with pytest.raises(ValueError, match=r"first dimension.*dimension of `events`"):
        EpochsSpectrumArray(data, info, freqs, events, dim_names=dim_names)
    # test data-dimname mismatch
    with pytest.raises(ValueError, match=r"Expected data to have.*dimensions, got.*"):
        EpochsSpectrumArray(data, info, freqs, dim_names=dim_names[:-1])
    # test unrecognised dim_names (for SpectrumArray; epoch not allowed)
    with pytest.raises(ValueError, match="Invalid value for the 'dim_names' parameter"):
        SpectrumArray(data[0, :, :], info, freqs, dim_names=("epoch", "channel"))
    # test unrecognised dim_names (for EpochsSpectrumArray)
    with pytest.raises(ValueError, match="Invalid value for the 'dim_names' parameter"):
        EpochsSpectrumArray(
            data, info, freqs, dim_names=("epoch", "channel", "notfreq")
        )
    # test missing dim_names
    with pytest.raises(ValueError, match="Both 'channel' and 'freq' must be present"):
        EpochsSpectrumArray(
            data, info, freqs, dim_names=("epoch", "channel", "channel")
        )
    with pytest.raises(ValueError, match="Both 'channel' and 'freq' must be present"):
        EpochsSpectrumArray(data, info, freqs, dim_names=("epoch", "freq", "freq"))
    with pytest.raises(ValueError, match="Both 'channel' and 'freq' must be present"):
        EpochsSpectrumArray(data, info, freqs, dim_names=("epoch", "epoch", "epoch"))
    # test incorrect channel location (for SpectrumArray; must be 1st dim)
    with pytest.raises(ValueError, match="'channel' must be the first dimension"):
        SpectrumArray(data[0, :, :], info, freqs, dim_names=("freq", "channel"))
    # test incorrect channel location (for EpochsSpectrumArray; must be 2nd dim)
    with pytest.raises(ValueError, match="'channel' must be the second dimension"):
        EpochsSpectrumArray(data, info, freqs, dim_names=("epoch", "freq", "channel"))
    # test mismatching number of channels
    with pytest.raises(
        ValueError, match=re.escape("number of good + bad data channels")
    ):
        EpochsSpectrumArray(data[:, :-1, :], info, freqs, dim_names=dim_names)
    # test incorrect taper position
    with pytest.raises(ValueError, match="'taper' must be the second to last dim"):
        EpochsSpectrumArray(
            np.expand_dims(data, axis=3), info, freqs, dim_names=dim_names + ("taper",)
        )
    # test incorrect weight size
    with pytest.raises(ValueError, match=r"Expected size of `weights` to be.*, got.*"):
        EpochsSpectrumArray(
            np.expand_dims(data, axis=2),
            info,
            freqs,
            dim_names=("epoch", "channel", "taper", "freq"),
            weights=None,
        )
    with pytest.raises(ValueError, match=r"Expected size of `weights` to be.*, got.*"):
        EpochsSpectrumArray(
            np.expand_dims(data, axis=2),
            info,
            freqs,
            dim_names=("epoch", "channel", "taper", "freq"),
            weights=np.ones((1, 2, 1)),
        )
    # test incorrect segment position
    with pytest.raises(ValueError, match="'segment' must be the last dim"):
        EpochsSpectrumArray(
            np.expand_dims(data, axis=2),
            info,
            freqs,
            dim_names=("epoch", "channel", "segment", "freq"),
        )
    # test mismatching number of frequencies
    with pytest.raises(ValueError, match=r"number of frequencies.*number of elements"):
        EpochsSpectrumArray(data[:, :, :-1], info, freqs, dim_names=dim_names)


@pytest.mark.parametrize(
    "kind, method, output, average",
    [
        ("raw", "welch", "power", "mean"),  # test with precomputed spectrum
        ("epochs", "welch", "power", False),  # test with segments
        ("epochs", "multitaper", "complex", None),  # test with tapers
    ],  # additional variants don't improve coverage
)
def test_spectrum_array(kind, method, output, average, tmp_path, request):
    """Test EpochsSpectrumArray and SpectrumArray constructors."""
    dim_names = ("epoch", "channel") if kind == "epochs" else ("channel",)
    if method == "welch":
        dim_names += ("freq",) if average else ("freq", "segment")
    else:  # i.e. multitaper
        dim_names += ("freq",) if output == "power" else ("taper", "freq")
    if method == "welch" and output == "power" and average:
        spectrum = request.getfixturevalue(f"{kind}_spectrum")
    else:
        data = request.getfixturevalue(kind)
        kwargs = dict()
        if method == "welch":
            kwargs.update(average=average)
        spectrum = data.compute_psd(method=method, output=output, **kwargs)
    data, freqs = spectrum.get_data(return_freqs=True)
    Klass = SpectrumArray if kind == "raw" else EpochsSpectrumArray
    spect_arr = Klass(
        data=data,
        info=spectrum.info,
        freqs=freqs,
        dim_names=dim_names,
        weights=spectrum.weights,
    )
    _check_spectrum_equivalent(spectrum, spect_arr, tmp_path)


@pytest.mark.parametrize(
    "method, output, average",
    [
        ("welch", "power", "mean"),  # test with precomputed spectrum
        ("welch", "complex", False),  # test aggr over segments & conversion to power
        ("multitaper", "complex", None),  # test aggr over tapers & conversion to power
    ],  # additional variants don't improve coverage
)
def test_plot_spectrum(method, output, average, request):
    """Test plotting EpochsSpectrum(Array).

    Testing Spectrum(Array) with raw data doesn't improve coverage.
    """
    if method == "welch" and output == "power" and average:
        spectrum = request.getfixturevalue("epochs_spectrum")
    else:
        data = request.getfixturevalue("epochs")
        kwargs = dict()
        if method == "welch":
            kwargs.update(average=average)
        spectrum = data.compute_psd(method=method, output=output, **kwargs)
    spectrum.info["bads"] = spectrum.ch_names[:1]  # one grad channel
    spectrum.plot(average=True, amplitude=True, spatial_colors=True)
    spectrum.plot(average=True, amplitude=False, spatial_colors=False)
    n_grad = sum(ch_type == "grad" for ch_type in spectrum.get_channel_types())
    for amp, sc in ((True, True), (False, False)):
        fig = spectrum.plot(average=False, amplitude=amp, spatial_colors=sc, exclude=())
        lines = fig.axes[0].lines[2:]  # grads, ignore two vlines
        assert len(lines) == n_grad
        bad_color = "0.5" if sc else "r"
        n_bad = sum(same_color(line.get_color(), bad_color) for line in lines)
        assert n_bad == 1
    spectrum.plot_topo()
    spectrum.plot_topomap()


def test_plot_spectrum_array_with_bads():
    """Test plotting a spectrum array with bads."""
    raw = RawArray(np.random.randn(3, 1000), create_info(3, 1000, "eeg"))
    raw.info["bads"] = [raw.ch_names[1]]
    spectrum = raw.compute_psd()
    with pytest.raises(
        ValueError, match=re.escape("number of good + bad data channels")
    ):
        SpectrumArray(spectrum.get_data(), spectrum.info, spectrum.freqs)
    spectrum2 = SpectrumArray(
        spectrum.get_data(exclude=()), spectrum.info, spectrum.freqs
    )
    spectrum2.plot(spatial_colors=False)


@pytest.mark.parametrize("dB", (False, True))
@pytest.mark.parametrize("amplitude", (False, True))
def test_plot_spectrum_dB(raw_spectrum, dB, amplitude):
    """Test that we properly handle amplitude/power and dB."""
    idx = 7
    power = 3
    freqs = np.linspace(1, 100, 100)
    data = np.full((1, freqs.size), np.finfo(float).tiny)
    data[0, idx] = power
    info = create_info(ch_names=["delta"], sfreq=1000, ch_types="eeg")
    psd = SpectrumArray(data=data, info=info, freqs=freqs)
    with pytest.warns(RuntimeWarning, match="Channel locations not available"):
        fig = psd.plot(dB=dB, amplitude=amplitude)
    trace = list(
        filter(lambda x: len(x.get_data()[0]) == len(freqs), fig.axes[0].lines)
    )[0]
    got = trace.get_data()[1][idx]
    want = power * 1e12  # scaling for EEG (V → μV), squared
    if amplitude:
        want = np.sqrt(want)
    if dB:
        want = (20 if amplitude else 10) * np.log10(want)

    assert want == got, f"expected {want}, got {got}"
