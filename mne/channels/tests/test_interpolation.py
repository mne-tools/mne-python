# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from itertools import compress
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import Epochs, pick_channels, pick_types, read_events
from mne._fiff.constants import FIFF
from mne._fiff.proj import _has_eeg_average_ref_proj
from mne.channels import make_dig_montage, make_standard_montage
from mne.channels.interpolation import _make_interpolation_matrix
from mne.datasets import testing
from mne.io import RawArray, read_raw_ctf, read_raw_fif, read_raw_nirx
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
)
from mne.utils import _record_warnings

base_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = base_dir / "test_raw.fif"
event_name = base_dir / "test-eve.fif"
raw_fname_ctf = base_dir / "test_ctf_raw.fif"
testing_path = testing.data_path(download=False)
event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2


def _load_data(kind):
    """Load data."""
    # It is more memory efficient to load data in a separate
    # function so it's loaded on-demand
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    # subselect channels for speed
    if kind == "eeg":
        picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])[:15]
        epochs = Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            picks=picks,
            preload=True,
            reject=dict(eeg=80e-6),
        )
    else:
        picks = pick_types(raw.info, meg=True, eeg=False, exclude=[])[1:200:2]
        assert kind == "meg"
        with pytest.warns(RuntimeWarning, match="projection"):
            epochs = Epochs(
                raw,
                events,
                event_id,
                tmin,
                tmax,
                picks=picks,
                preload=True,
                reject=dict(grad=1000e-12, mag=4e-12),
            )
    return raw, epochs


@pytest.mark.parametrize("offset", (0.0, 0.1))
@pytest.mark.parametrize(
    "avg_proj, ctol",
    [
        (True, (0.86, 0.93)),
        (False, (0.97, 0.99)),
    ],
)
@pytest.mark.parametrize(
    "method, atol",
    [
        pytest.param(None, 3e-6, marks=pytest.mark.slowtest),  # slow on Azure
        (dict(eeg="MNE"), 4e-6),
    ],
)
@pytest.mark.filterwarnings("ignore:.*than 20 mm from head frame origin.*")
def test_interpolation_eeg(offset, avg_proj, ctol, atol, method):
    """Test interpolation of EEG channels."""
    raw, epochs_eeg = _load_data("eeg")
    epochs_eeg = epochs_eeg.copy()
    assert not _has_eeg_average_ref_proj(epochs_eeg.info)
    # Offsetting the coordinate frame should have no effect on the output
    for inst in (raw, epochs_eeg):
        for ch in inst.info["chs"]:
            if ch["kind"] == FIFF.FIFFV_EEG_CH:
                ch["loc"][:3] += offset
                ch["loc"][3:6] += offset
        for d in inst.info["dig"]:
            d["r"] += offset

    # check that interpolation does nothing if no bads are marked
    epochs_eeg.info["bads"] = []
    evoked_eeg = epochs_eeg.average()
    kw = dict(method=method)
    with pytest.warns(RuntimeWarning, match="Doing nothing"):
        evoked_eeg.interpolate_bads(**kw)

    # create good and bad channels for EEG
    epochs_eeg.info["bads"] = []
    goods_idx = np.ones(len(epochs_eeg.ch_names), dtype=bool)
    goods_idx[epochs_eeg.ch_names.index("EEG 012")] = False
    bads_idx = ~goods_idx
    pos = epochs_eeg._get_channel_positions()

    evoked_eeg = epochs_eeg.average()
    if avg_proj:
        evoked_eeg.set_eeg_reference(projection=True).apply_proj()
        assert_allclose(evoked_eeg.data.mean(0), 0.0, atol=1e-20)
    ave_before = evoked_eeg.data[bads_idx]

    # interpolate bad channels for EEG
    epochs_eeg.info["bads"] = ["EEG 012"]
    evoked_eeg = epochs_eeg.average()
    if avg_proj:
        evoked_eeg.set_eeg_reference(projection=True).apply_proj()
        good_picks = pick_types(evoked_eeg.info, meg=False, eeg=True)
        assert_allclose(evoked_eeg.data[good_picks].mean(0), 0.0, atol=1e-20)
    evoked_eeg_bad = evoked_eeg.copy()
    bads_picks = pick_channels(
        epochs_eeg.ch_names, include=epochs_eeg.info["bads"], ordered=True
    )
    evoked_eeg_bad.data[bads_picks, :] = 1e10

    # Test first the exclude parameter
    evoked_eeg_2_bads = evoked_eeg_bad.copy()
    evoked_eeg_2_bads.info["bads"] = ["EEG 004", "EEG 012"]
    evoked_eeg_2_bads.data[
        pick_channels(evoked_eeg_bad.ch_names, ["EEG 004", "EEG 012"])
    ] = 1e10
    evoked_eeg_interp = evoked_eeg_2_bads.interpolate_bads(
        origin=(0.0, 0.0, 0.0), exclude=["EEG 004"], **kw
    )
    assert evoked_eeg_interp.info["bads"] == ["EEG 004"]
    assert np.all(evoked_eeg_interp.get_data("EEG 004") == 1e10)
    assert np.all(evoked_eeg_interp.get_data("EEG 012") != 1e10)

    # Now test without exclude parameter
    evoked_eeg_bad.info["bads"] = ["EEG 012"]
    evoked_eeg_interp = evoked_eeg_bad.copy().interpolate_bads(
        origin=(0.0, 0.0, 0.0), **kw
    )
    if avg_proj:
        assert_allclose(evoked_eeg_interp.data.mean(0), 0.0, atol=1e-6)
    interp_zero = evoked_eeg_interp.data[bads_idx]
    if method is None:  # using
        pos_good = pos[goods_idx]
        pos_bad = pos[bads_idx]
        interpolation = _make_interpolation_matrix(pos_good, pos_bad)
        assert interpolation.shape == (1, len(epochs_eeg.ch_names) - 1)
        interp_manual = np.dot(interpolation, evoked_eeg_bad.data[goods_idx])
        assert_array_equal(interp_manual, interp_zero)
        del interp_manual, interpolation, pos, pos_good, pos_bad
    assert_allclose(ave_before, interp_zero, atol=atol)
    assert ctol[0] < np.corrcoef(ave_before, interp_zero)[0, 1] < ctol[1]
    interp_fit = evoked_eeg_bad.copy().interpolate_bads(**kw).data[bads_idx]
    assert_allclose(ave_before, interp_fit, atol=2.5e-6)
    assert ctol[1] < np.corrcoef(ave_before, interp_fit)[0, 1]  # better

    # check that interpolation fails when preload is False
    epochs_eeg.preload = False
    with pytest.raises(RuntimeError, match="requires epochs data to be load"):
        epochs_eeg.interpolate_bads(**kw)
    epochs_eeg.preload = True

    # check that interpolation changes the data in raw
    raw_eeg = RawArray(data=epochs_eeg._data[0], info=epochs_eeg.info)
    raw_before = raw_eeg._data[bads_idx]
    raw_after = raw_eeg.interpolate_bads(**kw)._data[bads_idx]
    assert not np.all(raw_before == raw_after)

    # check that interpolation fails when preload is False
    for inst in [raw, epochs_eeg]:
        assert hasattr(inst, "preload")
        inst.preload = False
        inst.info["bads"] = [inst.ch_names[1]]
        with pytest.raises(RuntimeError, match="requires.*data to be loaded"):
            inst.interpolate_bads(**kw)

    # check that interpolation works with few channels
    raw_few = raw.copy().crop(0, 0.1).load_data()
    raw_few.pick(raw_few.ch_names[:1] + raw_few.ch_names[3:4])
    assert len(raw_few.ch_names) == 2
    raw_few.del_proj()
    raw_few.info["bads"] = [raw_few.ch_names[-1]]
    orig_data = raw_few[1][0]
    with _record_warnings() as w:
        raw_few.interpolate_bads(reset_bads=False, **kw)
    assert len([ww for ww in w if "more than" not in str(ww.message)]) == 0
    new_data = raw_few[1][0]
    assert (new_data == 0).mean() < 0.5
    assert np.corrcoef(new_data, orig_data)[0, 1] > 0.2


@pytest.mark.slowtest
def test_interpolation_meg():
    """Test interpolation of MEG channels."""
    # speed accuracy tradeoff: channel subselection is faster but the
    # correlation drops
    thresh = 0.68

    raw, epochs_meg = _load_data("meg")

    # check that interpolation works when non M/EEG channels are present
    # before MEG channels
    raw.crop(0, 0.1).load_data().pick(epochs_meg.ch_names)
    raw.info.normalize_proj()
    raw.set_channel_types({raw.ch_names[0]: "stim"}, on_unit_change="ignore")
    raw.info["bads"] = [raw.ch_names[1]]
    raw.load_data()
    raw.interpolate_bads(mode="fast")
    del raw

    # check that interpolation works for MEG
    epochs_meg.info["bads"] = ["MEG 0141"]
    evoked = epochs_meg.average()
    pick = pick_channels(epochs_meg.info["ch_names"], epochs_meg.info["bads"])

    # MEG -- raw
    raw_meg = RawArray(data=epochs_meg._data[0], info=epochs_meg.info)
    raw_meg.info["bads"] = ["MEG 0141"]
    data1 = raw_meg[pick, :][0][0]

    raw_meg.info.normalize_proj()
    data2 = raw_meg.interpolate_bads(reset_bads=False, mode="fast")[pick, :][0][0]
    assert np.corrcoef(data1, data2)[0, 1] > thresh
    # the same number of bads as before
    assert len(raw_meg.info["bads"]) == len(raw_meg.info["bads"])

    # MEG -- epochs
    data1 = epochs_meg.get_data(pick).ravel()
    epochs_meg.info.normalize_proj()
    epochs_meg.interpolate_bads(mode="fast")
    data2 = epochs_meg.get_data(pick).ravel()
    assert np.corrcoef(data1, data2)[0, 1] > thresh
    assert len(epochs_meg.info["bads"]) == 0

    # MEG -- evoked (plus auto origin)
    data1 = evoked.data[pick]
    evoked.info.normalize_proj()
    data2 = evoked.interpolate_bads(origin="auto").data[pick]
    assert np.corrcoef(data1, data2)[0, 1] > thresh

    # MEG -- with exclude
    evoked.info["bads"] = ["MEG 0141", "MEG 0121"]
    pick = pick_channels(evoked.ch_names, evoked.info["bads"], ordered=True)
    evoked.data[pick[-1]] = 1e10
    data1 = evoked.data[pick]
    evoked.info.normalize_proj()
    data2 = evoked.interpolate_bads(origin="auto", exclude=["MEG 0121"]).data[pick]
    assert np.corrcoef(data1[0], data2[0])[0, 1] > thresh
    assert np.all(data2[1] == 1e10)


def _this_interpol(inst, ref_meg=False):
    from mne.channels.interpolation import _interpolate_bads_meg

    _interpolate_bads_meg(inst, ref_meg=ref_meg, mode="fast")
    return inst


@pytest.mark.slowtest
def test_interpolate_meg_ctf():
    """Test interpolation of MEG channels from CTF system."""
    thresh = 0.85
    tol = 0.05  # assert the new interpol correlates at least .05 "better"
    bad = "MLC22-2622"  # select a good channel to test the interpolation

    raw = read_raw_fif(raw_fname_ctf).crop(0, 1.0).load_data()  # 3 secs
    raw.apply_gradient_compensation(3)

    # Show that we have to exclude ref_meg for interpolating CTF MEG-channels
    # (fixed in #5965):
    raw.info["bads"] = [bad]
    pick_bad = pick_channels(raw.info["ch_names"], raw.info["bads"])
    data_orig = raw[pick_bad, :][0]
    # mimic old behavior (the ref_meg-arg in _interpolate_bads_meg only serves
    # this purpose):
    data_interp_refmeg = _this_interpol(raw, ref_meg=True)[pick_bad, :][0]
    # new:
    data_interp_no_refmeg = _this_interpol(raw, ref_meg=False)[pick_bad, :][0]

    R = dict()
    R["no_refmeg"] = np.corrcoef(data_orig, data_interp_no_refmeg)[0, 1]
    R["with_refmeg"] = np.corrcoef(data_orig, data_interp_refmeg)[0, 1]

    print("Corrcoef of interpolated with original channel: ", R)
    assert R["no_refmeg"] > R["with_refmeg"] + tol
    assert R["no_refmeg"] > thresh


@testing.requires_testing_data
def test_interpolation_ctf_comp():
    """Test interpolation with compensated CTF data."""
    raw_fname = testing_path / "CTF" / "somMDYO-18av.ds"
    raw = read_raw_ctf(raw_fname, preload=True)
    raw.info["bads"] = [raw.ch_names[5], raw.ch_names[-5]]
    raw.interpolate_bads(mode="fast", origin=(0.0, 0.0, 0.04))
    assert raw.info["bads"] == []


@testing.requires_testing_data
def test_interpolation_nirs():
    """Test interpolating bad nirs channels."""
    pytest.importorskip("pymatreader")
    fname = testing_path / "NIRx" / "nirscout" / "nirx_15_2_recording_w_overlap"
    raw_intensity = read_raw_nirx(fname, preload=False)
    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    bad_0 = np.where([name == raw_od.info["bads"][0] for name in raw_od.ch_names])[0][0]
    bad_0_std_pre_interp = np.std(raw_od._data[bad_0])
    bads_init = list(raw_od.info["bads"])
    raw_od.interpolate_bads(exclude=bads_init[:2])
    assert raw_od.info["bads"] == bads_init[:2]
    raw_od.interpolate_bads()
    assert raw_od.info["bads"] == []
    assert bad_0_std_pre_interp > np.std(raw_od._data[bad_0])
    raw_haemo = beer_lambert_law(raw_od, ppf=6)
    raw_haemo.info["bads"] = raw_haemo.ch_names[2:4]
    assert raw_haemo.info["bads"] == ["S1_D2 hbo", "S1_D2 hbr"]
    raw_haemo.interpolate_bads()
    assert raw_haemo.info["bads"] == []


@testing.requires_testing_data
def test_interpolation_ecog():
    """Test interpolation for ECoG."""
    raw, epochs_eeg = _load_data("eeg")
    bads = ["EEG 012"]
    bads_mask = np.isin(epochs_eeg.ch_names, bads)

    epochs_ecog = epochs_eeg.set_channel_types(
        {ch: "ecog" for ch in epochs_eeg.ch_names}
    )
    epochs_ecog.info["bads"] = bads

    # check that interpolation changes the data in raw
    raw_ecog = RawArray(data=epochs_ecog._data[0], info=epochs_ecog.info)
    raw_before = raw_ecog.copy()
    raw_after = raw_ecog.interpolate_bads(method=dict(ecog="spline"))
    assert not np.all(raw_before._data[bads_mask] == raw_after._data[bads_mask])
    assert_array_equal(raw_before._data[~bads_mask], raw_after._data[~bads_mask])


@testing.requires_testing_data
def test_interpolation_seeg():
    """Test interpolation for sEEG."""
    raw, epochs_eeg = _load_data("eeg")
    bads = ["EEG 012"]
    bads_mask = np.isin(epochs_eeg.ch_names, bads)
    epochs_seeg = epochs_eeg.set_channel_types(
        {ch: "seeg" for ch in epochs_eeg.ch_names}
    )
    epochs_seeg.info["bads"] = bads

    # check that interpolation changes the data in raw
    raw_seeg = RawArray(data=epochs_seeg._data[0], info=epochs_seeg.info)
    raw_before = raw_seeg.copy()
    montage = raw_seeg.get_montage()
    pos = montage.get_positions()
    ch_pos = pos.pop("ch_pos")
    n0 = ch_pos[epochs_seeg.ch_names[0]]
    n1 = ch_pos[epochs_seeg.ch_names[1]]
    for i, ch in enumerate(epochs_seeg.ch_names[2:]):
        ch_pos[ch] = n0 + (n1 - n0) * (i + 2)
    raw_seeg.set_montage(make_dig_montage(ch_pos, **pos))
    raw_after = raw_seeg.interpolate_bads(method=dict(seeg="spline"))
    assert not np.all(raw_before._data[bads_mask] == raw_after._data[bads_mask])
    assert_array_equal(raw_before._data[~bads_mask], raw_after._data[~bads_mask])

    # check interpolation on epochs
    epochs_seeg.set_montage(make_dig_montage(ch_pos, **pos))
    epochs_before = epochs_seeg.copy()
    epochs_after = epochs_seeg.interpolate_bads(method=dict(seeg="spline"))
    assert not np.all(
        epochs_before._data[:, bads_mask] == epochs_after._data[:, bads_mask]
    )
    assert_array_equal(
        epochs_before._data[:, ~bads_mask], epochs_after._data[:, ~bads_mask]
    )

    # test shaft all bad
    epochs_seeg.info["bads"] = epochs_seeg.ch_names
    with pytest.raises(RuntimeError, match="Not enough good channels"):
        epochs_seeg.interpolate_bads(method=dict(seeg="spline"))

    # test bad not on shaft
    ch_pos[bads[0]] = np.array([10, 10, 10])
    epochs_seeg.info["bads"] = bads
    epochs_seeg.set_montage(make_dig_montage(ch_pos, **pos))
    with pytest.raises(RuntimeError, match="No shaft found"):
        epochs_seeg.interpolate_bads(method=dict(seeg="spline"))


def test_nan_interpolation(raw):
    """Test 'nan' method for interpolating bads."""
    ch_to_interp = [raw.ch_names[1]]  # don't use channel 0 (type is IAS not MEG)
    raw.info["bads"] = ch_to_interp

    # test that warning appears for reset_bads = True
    with pytest.warns(RuntimeWarning, match="Consider setting reset_bads=False"):
        raw.interpolate_bads(method="nan", reset_bads=True)

    # despite warning, interpolation still happened, make sure the channel is NaN
    bad_chs = raw.get_data(ch_to_interp)
    assert np.isnan(bad_chs).all()

    # make sure reset_bads=False works as expected
    raw.info["bads"] = ch_to_interp
    raw.interpolate_bads(method="nan", reset_bads=False)
    assert raw.info["bads"] == ch_to_interp

    # make sure other channels are untouched
    raw.drop_channels(ch_to_interp)
    good_chs = raw.get_data()
    assert np.isfinite(good_chs).all()


@testing.requires_testing_data
def test_method_str():
    """Test method argument types."""
    raw = read_raw_fif(
        testing_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif",
        preload=False,
    )
    raw.crop(0, 1).pick(("meg", "eeg"), exclude=()).load_data()
    raw.copy().interpolate_bads(method="MNE")
    with pytest.raises(ValueError, match="Invalid value for the"):
        raw.interpolate_bads(method="spline")
    raw.pick("eeg", exclude=())
    raw.interpolate_bads(method="spline")


@pytest.mark.parametrize("montage_name", ["biosemi16", "standard_1020"])
@pytest.mark.parametrize("method", ["spline", "MNE"])
@pytest.mark.parametrize("data_type", ["raw", "epochs", "evoked"])
def test_interpolate_to_eeg(montage_name, method, data_type):
    """Test the interpolate_to method for EEG for raw, epochs, and evoked."""
    # Load EEG data
    raw, epochs_eeg = _load_data("eeg")
    epochs_eeg = epochs_eeg.copy()

    # Load data for raw
    raw.load_data()

    # Create a target montage
    montage = make_standard_montage(montage_name)

    # Prepare data to interpolate to
    if data_type == "raw":
        inst = raw.copy()
    elif data_type == "epochs":
        inst = epochs_eeg.copy()
    elif data_type == "evoked":
        inst = epochs_eeg.average()
    shape = list(inst._data.shape)
    orig_total = len(inst.info["ch_names"])
    n_eeg_orig = len(pick_types(inst.info, eeg=True))

    # Assert first and last channels are not EEG
    if data_type == "raw":
        ch_types = inst.get_channel_types()
        assert ch_types[0] != "eeg"
        assert ch_types[-1] != "eeg"

    # Record the names and data of the first and last channels.
    if data_type == "raw":
        first_name = inst.info["ch_names"][0]
        last_name = inst.info["ch_names"][-1]
        data_first = inst._data[..., 0, :].copy()
        data_last = inst._data[..., -1, :].copy()

    # Interpolate the EEG channels.
    inst_interp = inst.copy().interpolate_to(montage, method=method)

    # Check that the new channel names include the montage channels.
    assert set(montage.ch_names).issubset(set(inst_interp.info["ch_names"]))
    # Check that the overall channel order is changed.
    assert inst.info["ch_names"] != inst_interp.info["ch_names"]

    # Check that the data shape is as expected.
    new_nchan_expected = orig_total - n_eeg_orig + len(montage.ch_names)
    expected_shape = (new_nchan_expected, shape[-1])
    if len(shape) == 3:
        expected_shape = (shape[0],) + expected_shape
    assert inst_interp._data.shape == expected_shape

    # Verify that the first and last channels retain their positions.
    if data_type == "raw":
        assert inst_interp.info["ch_names"][0] == first_name
        assert inst_interp.info["ch_names"][-1] == last_name

    # Verify that the data for the first and last channels is unchanged.
    if data_type == "raw":
        np.testing.assert_allclose(
            inst_interp._data[..., 0, :],
            data_first,
            err_msg="Data for the first non-EEG channel has changed.",
        )
        np.testing.assert_allclose(
            inst_interp._data[..., -1, :],
            data_last,
            err_msg="Data for the last non-EEG channel has changed.",
        )

    # Validate that bad channels are carried over.
    # Mark the first non eeg channel as bad
    all_ch = inst_interp.info["ch_names"]
    eeg_ch = [all_ch[i] for i in pick_types(inst_interp.info, eeg=True)]
    bads = [ch for ch in all_ch if ch not in eeg_ch][:1]
    inst.info["bads"] = bads
    inst_interp = inst.copy().interpolate_to(montage, method=method)
    assert inst_interp.info["bads"] == bads
