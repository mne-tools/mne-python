"""Test reading NWB voltage data from files written by PyNWB."""

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
    """Write an NWB file with a calibrated voltage series."""
    from pynwb.testing.mock.ecephys import mock_ElectrodesTable

    kind = getattr(request, "param", "electrical")
    fname = tmp_path / "test.nwb"
    date = datetime.fromisoformat("2020-01-02T04:04:05+01:00")
    nwb = pynwb.NWBFile("session", "test", date)
    nwb.subject = pynwb.file.Subject(subject_id="a", sex="F")
    table = mock_ElectrodesTable(nwbfile=nwb)
    table.id.data[:] = [10, 20, 30, 40, 50]
    n_channels = 1 if kind == "single" else 4
    data = np.random.default_rng(0).normal(size=(512, n_channels))
    values = data[:, 0] if kind == "single" else data
    kwargs = dict(starting_time=1.5, rate=100.0, conversion=2e-6, offset=0.001)
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
    series = [
        cls(name=f"signal{idx}", data=values, **kwargs)
        for idx in range(2 if kind == "ambiguous" else 1)
    ]
    if kind == "processing":  # voltage only in processing/ecephys/LFP
        lfp = pynwb.ecephys.LFP()
        nwb.create_processing_module("ecephys", "LFP").add(lfp)
        for one in series:
            lfp.add_electrical_series(one)
    else:
        for one in series:
            nwb.add_acquisition(one)
    # a non-voltage series that should be ignored
    other = pynwb.TimeSeries(
        name="other", data=data, unit="arbitrary", starting_time=1.5, rate=100.0
    )
    nwb.add_acquisition(other)
    for start in (2.0, 2.5):
        nwb.add_trial(start_time=start, stop_time=start + 0.25)
    # invalid intervals scoped to the voltage series, and to another series
    nwb.add_invalid_time_interval(start_time=3.0, stop_time=3.5, timeseries=series)
    nwb.add_invalid_time_interval(start_time=4.0, stop_time=4.5, timeseries=[other])
    with pynwb.NWBHDF5IO(fname, "w") as writer:
        writer.write(nwb)
    return fname, data.T * (2e-6 * factors[:, None]) + 0.001, kind


@pytest.mark.parametrize(
    "nwb_file",
    ["electrical", "single", "timestamps", "ambiguous", "processing"],
    indirect=True,
)
def test_nwb_read(nwb_file, monkeypatch):
    """Check calibrated data, metadata, lazy reads and FIF round-trips."""
    fname, expected, kind = nwb_file
    # the reader must only need h5py
    monkeypatch.setitem(sys.modules, "pynwb", None)
    monkeypatch.setitem(sys.modules, "hdmf", None)
    if kind == "ambiguous":
        with pytest.raises(ValueError, match="found 2.*series="):
            io.read_raw_nwb(fname)
    series = "acquisition/signal1" if kind == "ambiguous" else None
    raw = io.read_raw_nwb(fname, series=series)
    assert not raw.preload
    assert_allclose(raw.get_data(), expected)
    assert_allclose(raw.copy().pick([0]).get_data(start=7, stop=19), expected[:1, 7:19])
    ids = range(4) if kind == "timestamps" else (40, 10, 50, 20)
    assert raw.ch_names == [str(i) for i in ids][: len(expected)]
    assert raw.get_channel_types() == ["seeg"] * len(expected)
    unit = "µV" if kind == "timestamps" else "V"
    assert raw._orig_units == dict.fromkeys(raw.ch_names, unit)
    assert raw.info["meas_date"] == datetime(2020, 1, 2, 3, 4, 6, 500000, tzinfo=UTC)
    assert raw.info["subject_info"] == dict(his_id="a", sex=2)
    # onsets are relative to the first sample, at 1.5 s; the invalid interval
    # scoped to the other series does not apply
    assert list(raw.annotations.description) == ["NWB/trials"] * 2 + ["BAD_nwb_invalid"]
    assert_allclose(raw.annotations.onset, [0.5, 1.0, 1.5])
    assert_allclose(raw.annotations.duration, [0.25, 0.25, 0.5])
    if kind == "electrical":
        raw = io.read_raw_nwb(fname, ch_type="ecog")
        assert raw.get_channel_types() == ["ecog"] * len(expected)
        _test_raw_reader(io.read_raw_nwb, fname=fname, test_scaling=False)


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("data@unit", "degrees", "voltage"),
        ("starting_time@rate", 0.0, "sampling rate"),
        ("channel_conversion", [1.0], "channel_conversion"),
        ("timestamps", 1.5 + np.arange(512) ** 2, "nonuniform timestamps"),
        ("/timestamps_reference_time", "2020-01-02T03:04:05", None),
    ],
)
def test_nwb_invalid_metadata(nwb_file, path, value, match):
    """Reject non-voltage units, bad clocks and mismatched calibration."""
    fname, _, _ = nwb_file
    with h5py.File(fname, "a") as f:
        series = f["acquisition/signal0"]
        path, _, attribute = path.partition("@")
        if attribute:
            series[path].attrs[attribute] = value
        elif path.startswith("/"):
            f[path][()] = value
        else:
            if path == "timestamps":
                del series["starting_time"]
            elif path in series:
                del series[path]
            series[path] = value
    if match is None:  # date without a timezone
        with pytest.warns(RuntimeWarning, match="no timezone"):
            assert io.read_raw_nwb(fname).info["meas_date"] is None
    else:
        with pytest.raises(ValueError, match=match):
            io.read_raw_nwb(fname)
