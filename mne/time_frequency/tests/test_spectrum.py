# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
from functools import partial

import numpy as np
import pytest
from matplotlib.colors import same_color
from numpy.testing import assert_array_equal

from mne import Annotations
from mne.time_frequency import read_spectrum
from mne.time_frequency.multitaper import _psd_from_mt
from mne.time_frequency.spectrum import EpochsSpectrumArray, SpectrumArray


def test_compute_psd_errors(raw):
    """Test for expected errors in the .compute_psd() method."""
    with pytest.raises(ValueError, match="must not exceed ½ the sampling"):
        raw.compute_psd(fmax=raw.info["sfreq"] * 0.51)
    with pytest.raises(TypeError, match="unexpected keyword argument foo for"):
        raw.compute_psd(foo=None)
    with pytest.raises(TypeError, match="keyword arguments foo, bar for"):
        raw.compute_psd(foo=None, bar=None)
    # TODO: More code to remove here?
    with pytest.raises(RuntimeError, match="Complex output support in.*not supported"):
        raw.compute_psd(output="complex")


@pytest.mark.parametrize("method", ("welch", "multitaper"))
@pytest.mark.parametrize(
    (
        "fmin, fmax, tmin, tmax, picks, proj, n_fft, n_overlap, n_per_seg, "
        "average, window, bandwidth, adaptive, low_bias, normalization"
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
        ],  # non-defaults
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


def _get_inst(inst, request, evoked):
    # ↓ XXX workaround:
    # ↓ parametrized fixtures are not accessible via request.getfixturevalue
    # ↓ https://github.com/pytest-dev/pytest/issues/4666#issuecomment-456593913
    return evoked if inst == "evoked" else request.getfixturevalue(inst)


@pytest.mark.parametrize("inst", ("raw", "epochs", "evoked"))
def test_spectrum_io(inst, tmp_path, request, evoked):
    """Test save/load of spectrum objects."""
    pytest.importorskip("h5io")
    fname = tmp_path / f"{inst}-spectrum.h5"
    inst = _get_inst(inst, request, evoked)
    orig = inst.compute_psd()
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


def test_spectrum_reject_by_annot(raw):
    """Test rejecting by annotation.

    Cannot use raw_spectrum fixture here because we're testing reject_by_annotation in
    .compute_psd() method.
    """
    spect_no_annot = raw.compute_psd()
    raw.set_annotations(Annotations([1, 5], [3, 3], ["test", "test"]))
    spect_benign_annot = raw.compute_psd()
    raw.annotations.description = np.array(["bad_test", "bad_test"])
    spect_reject_annot = raw.compute_psd()
    spect_ignored_annot = raw.compute_psd(reject_by_annotation=False)
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
    [
        ("welch", "power"),
    ],
)
def test_unaggregated_spectrum_to_data_frame(raw, long_format, method, output):
    """Test converting complex multitaper spectra to data frame."""
    pytest.importorskip("pandas")
    from pandas.testing import assert_frame_equal

    from mne.utils.dataframe import _inplace

    # aggregated spectrum → dataframe
    orig_df = raw.compute_psd(method=method).to_data_frame(long_format=long_format)
    # unaggregated welch or complex multitaper →
    #   aggregate w/ pandas (to make sure we did reshaping right)
    kwargs = dict()
    if method == "welch":
        kwargs.update(average=False, verbose="error")
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
                return np.nanmean(np.abs(x))

            agg_df = gb.agg(_fun)
        else:
            agg_df = gb.mean()  # excludes missing values itself
    else:
        gb = gb[df.columns]  # https://github.com/pandas-dev/pandas/pull/52477
        agg_df = gb.apply(_agg_helper, spectrum._mt_weights, grouping_cols)
    # even with check_categorical=False, we know that the *data* matches;
    # what may differ is the order of the "levels" in the *metadata* for the
    # channel name column
    assert_frame_equal(agg_df, orig_df, check_categorical=False)


@pytest.mark.parametrize("inst", ("raw_spectrum", "epochs_spectrum", "evoked"))
def test_spectrum_to_data_frame(inst, request, evoked):
    """Test the to_data_frame method for Spectrum."""
    pytest.importorskip("pandas")
    from pandas.testing import assert_frame_equal

    # setup
    is_already_psd = inst in ("raw_spectrum", "epochs_spectrum")
    is_epochs = inst == "epochs_spectrum"
    inst = _get_inst(inst, request, evoked)
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


def test_spectrum_kwarg_triaging(raw):
    """Test kwarg triaging in legacy plot_psd() method."""
    import matplotlib.pyplot as plt

    regex = r"legacy plot_psd\(\) method.*unexpected keyword.*'axes'.*Try rewriting"
    fig, axes = plt.subplots(1, 2)
    # `axes` is the new param name: technically only valid for Spectrum.plot()
    with pytest.warns(RuntimeWarning, match=regex):
        raw.plot_psd(axes=axes)
    # `ax` is the correct legacy param name
    raw.plot_psd(ax=axes)


def _check_spectrum_equivalent(spect1, spect2, tmp_path):
    data1 = spect1.get_data()
    data2 = spect2.get_data()
    assert_array_equal(data1, data2)
    assert_array_equal(spect1.freqs, spect2.freqs)


def test_spectrum_array_errors(epochs_spectrum):
    """Test EpochsSpectrumArray constructor errors."""
    data, freqs = epochs_spectrum.get_data(return_freqs=True)
    info = epochs_spectrum.info
    with pytest.raises(ValueError, match="Data must be a 3D array"):
        EpochsSpectrumArray(np.empty((2, 3, 4, 5)), info, freqs)
    with pytest.raises(ValueError, match=r"number of channels.*good data channels"):
        EpochsSpectrumArray(data[:, :-1], info, freqs)
    with pytest.raises(ValueError, match=r"last dimension.*same number of elements"):
        EpochsSpectrumArray(data[..., :-1], info, freqs)
    # test mismatching events shape
    n_epo = data.shape[0] + 1  # +1 so they purposely don't match
    events = np.vstack(
        (np.arange(n_epo), np.zeros(n_epo, dtype=int), np.ones(n_epo, dtype=int))
    ).T
    with pytest.raises(ValueError, match=r"first dimension.*dimension of `events`"):
        EpochsSpectrumArray(data, info, freqs, events)


@pytest.mark.parametrize("kind", ("raw", "epochs"))
def test_spectrum_array(kind, tmp_path, request):
    """Test EpochsSpectrumArray and SpectrumArray constructors."""
    spectrum = request.getfixturevalue(f"{kind}_spectrum")
    data, freqs = spectrum.get_data(return_freqs=True)
    Klass = SpectrumArray if kind == "raw" else EpochsSpectrumArray
    spect_arr = Klass(data=data, info=spectrum.info, freqs=freqs)
    _check_spectrum_equivalent(spectrum, spect_arr, tmp_path)


@pytest.mark.parametrize("kind", ("raw", "epochs"))
@pytest.mark.parametrize("array", (False, True))
def test_plot_spectrum(kind, array, request):
    """Test plotting (Epochs)Spectrum(Array)."""
    spectrum = request.getfixturevalue(f"{kind}_spectrum")
    if array:
        data, freqs = spectrum.get_data(return_freqs=True)
        Klass = SpectrumArray if kind == "raw" else EpochsSpectrumArray
        spectrum = Klass(data=data, info=spectrum.info, freqs=freqs)
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
