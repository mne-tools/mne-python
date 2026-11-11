"""Test NWB voltage data and metadata using independently written fixtures."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import sys
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import io
from mne.io.tests.test_raw import _test_raw_reader

pynwb = pytest.importorskip("pynwb", minversion="3.1")
h5py = pytest.importorskip("h5py")


@pytest.fixture
def nwb_file(tmp_path, request):
    """Write a valid NWB series with calibration, metadata and scoped intervals."""
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb.testing.mock.ecephys import mock_ElectrodesTable

    kind = getattr(request, "param", "electrical")
    fname = tmp_path / "test.nwb"
    date = datetime.fromisoformat("2020-01-02T04:04:05+01:00")
    nwb = pynwb.NWBFile("session", "test", date, experimenter=["A", "B"])
    table = mock_ElectrodesTable(nwbfile=nwb)
    table.id.data[:] = [10, 20, 30, 40, 50]
    table.add_column("bad", "Bad channel", data=[True, False, False, False, False])
    n_channels = 1 if kind == "single" else 4
    data = np.random.default_rng(0).normal(size=(512, n_channels))
    values = data[:, 0] if kind == "single" else data
    offset = 0.0 if kind == "compressed" else 0.001
    kwargs = dict(starting_time=1.5, rate=100.0, conversion=2e-6, offset=offset)
    factors = np.arange(1.0, n_channels + 1)
    cls = pynwb.ecephys.ElectricalSeries
    if kind == "timestamps":
        cls, factors = pynwb.TimeSeries, np.ones(n_channels)
        kwargs = dict(
            timestamps=1.5 + np.arange(512) / 100.0,
            unit="microvolts",
            conversion=2.0,
            offset=1000.0,
        )
    else:
        kwargs.update(
            electrodes=nwb.create_electrode_table_region(
                [3, 0, 4, 1][:n_channels], "selected"
            ),
            channel_conversion=factors,
        )
    if kind == "compressed":
        values = H5DataIO(values.astype("float32"), compression="gzip")
        data = data.astype("float32")
    for idx in range(2 if kind == "ambiguous" else 1):
        series = cls(name=f"signal{idx}", data=values, **kwargs)
        nwb.add_acquisition(series)
    nwb.subject = pynwb.file.Subject(
        subject_id="a", sex="F", weight="60000 g", species="Homo sapiens"
    )
    other = pynwb.TimeSeries(
        name="other", data=data, unit="arbitrary", starting_time=1.5, rate=100.0
    )
    nwb.add_acquisition(other)
    nwb.add_invalid_time_interval(start_time=1.7, stop_time=1.9, timeseries=[series])
    nwb.add_invalid_time_interval(start_time=1.8, stop_time=2.0, timeseries=[other])
    nwb.add_trial(start_time=2.0, stop_time=2.3)
    with pynwb.NWBHDF5IO(fname, "w") as writer:
        writer.write(nwb)
    return fname, data.T * (2e-6 * factors[:, None]) + offset, kind


@pytest.mark.parametrize(
    "nwb_file",
    ["electrical", "compressed", "single", "timestamps", "ambiguous"],
    indirect=True,
)
def test_nwb_read(nwb_file, monkeypatch):
    """Check calibrated data, metadata, lazy reads and FIF round-trips."""
    fname, expected, kind = nwb_file
    monkeypatch.setitem(sys.modules, "pynwb", None)
    monkeypatch.setitem(sys.modules, "hdmf", None)
    if kind == "ambiguous":
        with pytest.raises(ValueError, match="found 2"):
            io.read_raw_nwb(fname)
        return
    raw = io.read_raw_nwb(fname)
    assert not raw.preload
    assert_allclose(raw.get_data(), expected)
    assert_allclose(raw.copy().pick([0]).get_data(start=7, stop=19), expected[:1, 7:19])
    ids = range(4) if kind == "timestamps" else (40, 10, 50, 20)
    assert raw.ch_names == [str(i) for i in ids][: len(expected)]
    assert raw.get_channel_types() == ["seeg"] * len(expected)
    unit = "µV" if kind == "timestamps" else "V"
    assert raw._orig_units == dict.fromkeys(raw.ch_names, unit)
    assert raw.info["meas_date"] == datetime(2020, 1, 2, 3, 4, 6, 500000, tzinfo=UTC)
    assert raw.info["subject_info"] == dict(his_id="a", sex=2, weight=60.0)
    assert "Homo sapiens" in raw.info["description"]
    assert list(raw.annotations.description) == ["BAD_nwb_invalid", "NWB/trial"]
    assert_allclose(raw.annotations.onset, [0.2, 0.5])
    assert_allclose(raw.annotations.duration, [0.2, 0.3])
    if kind == "electrical":
        assert raw.info["bads"] == ["10"]
        _test_raw_reader(io.read_raw_nwb, fname=fname, test_scaling=False)


def test_nwb_overrides(nwb_file):
    """Apply corrections in memory without resampling or changing the file."""
    fname, expected, _ = nwb_file
    before = fname.read_bytes()
    overrides = dict(
        sfreq=50.0, meas_date=None, ch_names=["A", "B", "C", "D"], ch_types="ecog"
    )
    with pytest.warns(RuntimeWarning, match="sampling frequency"):
        raw = io.read_raw_nwb(fname, overrides=overrides)
    assert raw.ch_names == overrides["ch_names"]
    assert raw.get_channel_types() == ["ecog"] * 4
    assert raw.info["meas_date"] is None and raw.info["bads"] == ["B"]
    assert_allclose(raw.get_data(), expected)
    assert fname.read_bytes() == before
    with h5py.File(fname, "a") as f:
        f["timestamps_reference_time"][()] = "2020-01-02T03:04:05"
    with pytest.warns(RuntimeWarning, match="date.*unset"):
        assert io.read_raw_nwb(fname).info["meas_date"] is None


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("overrides", {"unknown": 1}, "override"),
        ("overrides", {"sfreq": np.inf}, "finite"),
        ("overrides", {"ch_names": ["A"] * 4}, "unique"),
        ("data@unit", "degrees", "voltage"),
        ("starting_time@rate", 0.0, "sampling rate"),
        ("channel_conversion", [1.0], "channel_conversion"),
        ("timestamps", 1.5 + np.arange(512) ** 2, "nonuniform timestamps"),
        ("/intervals/invalid_times/timeseries_index", [1, 0], "Invalid NWB"),
        ("/general/extracellular_ephys/electrodes/bad", [False], "Invalid NWB"),
    ],
)
def test_nwb_invalid_metadata(nwb_file, path, value, match):
    """Reject miscalibration, irregular clocks and malformed table columns."""
    fname, expected, _ = nwb_file
    overrides = {}
    with h5py.File(fname, "a") as f:
        series = f["acquisition/signal0"]
        path, _, attribute = path.partition("@")
        if path == "overrides":
            overrides = value
        elif attribute:
            series[path].attrs[attribute] = value
        else:
            if path == "timestamps":
                del series["starting_time"]
            elif path in series:
                del series[path]
            series[path] = value
    with pytest.raises((TypeError, ValueError), match=match):
        io.read_raw_nwb(fname, overrides=overrides)
    if path == "timestamps":
        with pytest.warns(RuntimeWarning, match="maximum.*discrepancy"):
            assert_allclose(
                io.read_raw_nwb(fname, overrides={"sfreq": 100.0}).get_data(), expected
            )
