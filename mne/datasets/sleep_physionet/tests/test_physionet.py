# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import pytest

from mne.datasets.sleep_physionet import age, temazepam
from mne.datasets.sleep_physionet._utils import (
    AGE_SLEEP_RECORDS,
    TEMAZEPAM_SLEEP_RECORDS,
    _update_sleep_age_records,
    _update_sleep_temazepam_records,
)
from mne.utils import requires_good_network


@pytest.fixture(scope="session")
def physionet_tmpdir(tmp_path_factory):
    """Fixture exposing a temporary directory for testing."""
    return str(tmp_path_factory.mktemp("physionet_files"))


def _keep_basename_only(paths):
    return [Path(p).name for p in paths]


def _get_expected_url(name):
    base = "https://physionet.org/physiobank/database/sleep-edfx/"
    middle = "sleep-cassette/" if name.startswith("SC") else "sleep-telemetry/"
    return base + middle + "/" + name


def _get_expected_path(base, name):
    return Path(base, name)


def _check_mocked_function_calls(mocked_func, call_fname_hash_pairs, base_path):
    # Check mocked_func has been called the right amount of times.
    assert mocked_func.call_count == len(call_fname_hash_pairs)

    # Check it has been called with the right parameters in the right
    # order.
    for idx, current in enumerate(call_fname_hash_pairs):
        _, call_kwargs = mocked_func.call_args_list[idx]
        hash_type, hash_ = call_kwargs["known_hash"].split(":")
        assert call_kwargs["url"] == _get_expected_url(current["name"]), idx
        assert Path(call_kwargs["path"], call_kwargs["fname"]) == _get_expected_path(
            base_path, current["name"]
        )
        assert hash_ == current["hash"]
        assert hash_type == "sha1"


@pytest.mark.timeout(60)
@pytest.mark.xfail(strict=False)
@requires_good_network
def test_run_update_age_records(tmp_path):
    """Test Sleep Physionet URL handling."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("xlrd", "0.9")

    fname = tmp_path / "records.csv"
    _update_sleep_age_records(fname)
    data = pd.read_csv(fname)
    pd.testing.assert_frame_equal(data, pd.read_csv(AGE_SLEEP_RECORDS))


@pytest.mark.parametrize("subject", [39, 68, 69, 78, 79, 83])
def test_sleep_physionet_age_missing_subjects(
    physionet_tmpdir, subject, download_is_error
):
    """Test handling of missing subjects in Sleep Physionet age fetcher."""
    with pytest.raises(ValueError, match="This dataset contains subjects 0 to 82"):
        age.fetch_data(
            subjects=[subject], recording=[1], on_missing="raise", path=physionet_tmpdir
        )
    with pytest.warns(RuntimeWarning, match="This dataset contains subjects 0 to 82"):
        age.fetch_data(
            subjects=[subject], recording=[1], on_missing="warn", path=physionet_tmpdir
        )
    paths = age.fetch_data(
        subjects=[subject], recording=[1], on_missing="ignore", path=physionet_tmpdir
    )
    assert paths == []


@pytest.mark.parametrize("subject,recording", [(13, 2), (36, 1), (52, 1)])
def test_sleep_physionet_age_missing_recordings(
    physionet_tmpdir, subject, recording, download_is_error
):
    """Test handling of missing recordings in Sleep Physionet age fetcher."""
    with pytest.raises(
        ValueError, match=f"Requested recording {recording} for subject"
    ):
        age.fetch_data(
            subjects=[subject],
            recording=[recording],
            on_missing="raise",
            path=physionet_tmpdir,
        )
    with pytest.warns(
        RuntimeWarning, match=f"Requested recording {recording} for subject"
    ):
        age.fetch_data(
            subjects=[subject],
            recording=[recording],
            on_missing="warn",
            path=physionet_tmpdir,
        )
    paths = age.fetch_data(
        subjects=[subject],
        recording=[recording],
        on_missing="ignore",
        path=physionet_tmpdir,
    )
    assert paths == []


def test_sleep_physionet_age(physionet_tmpdir, fake_retrieve):
    """Test Sleep Physionet URL handling."""
    paths = age.fetch_data(subjects=[0], recording=[1], path=physionet_tmpdir)
    assert _keep_basename_only(paths[0]) == [
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf",
    ]

    paths = age.fetch_data(subjects=[0, 1], recording=[1], path=physionet_tmpdir)
    assert _keep_basename_only(paths[0]) == [
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf",
    ]
    assert _keep_basename_only(paths[1]) == [
        "SC4011E0-PSG.edf",
        "SC4011EH-Hypnogram.edf",
    ]

    paths = age.fetch_data(subjects=[0], recording=[1, 2], path=physionet_tmpdir)
    assert _keep_basename_only(paths[0]) == [
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf",
    ]
    assert _keep_basename_only(paths[1]) == [
        "SC4002E0-PSG.edf",
        "SC4002EC-Hypnogram.edf",
    ]

    EXPECTED_CALLS = (
        {
            "name": "SC4001E0-PSG.edf",
            "hash": "adabd3b01fc7bb75c523a974f38ee3ae4e57b40f",
        },
        {
            "name": "SC4001EC-Hypnogram.edf",
            "hash": "21c998eadc8b1e3ea6727d3585186b8f76e7e70b",
        },
        {
            "name": "SC4011E0-PSG.edf",
            "hash": "4d17451f7847355bcab17584de05e7e1df58c660",
        },
        {
            "name": "SC4011EH-Hypnogram.edf",
            "hash": "d582a3cbe2db481a362af890bc5a2f5ca7c878dc",
        },
        {
            "name": "SC4002E0-PSG.edf",
            "hash": "c6b6d7a8605cc7e7602b6028ee77f6fbf5f7581d",
        },
        {
            "name": "SC4002EC-Hypnogram.edf",
            "hash": "386230188a3552b1fc90bba0fb7476ceaca174b6",
        },
    )
    base_path = age.data_path(path=physionet_tmpdir)
    _check_mocked_function_calls(fake_retrieve, EXPECTED_CALLS, base_path)


@pytest.mark.xfail(strict=False)
@requires_good_network
def test_run_update_temazepam_records(tmp_path):
    """Test Sleep Physionet URL handling."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("xlrd", "0.9")
    fname = tmp_path / "records.csv"
    _update_sleep_temazepam_records(fname)
    data = pd.read_csv(fname)

    pd.testing.assert_frame_equal(data, pd.read_csv(TEMAZEPAM_SLEEP_RECORDS))


def test_sleep_physionet_temazepam(physionet_tmpdir, fake_retrieve):
    """Test Sleep Physionet URL handling."""
    paths = temazepam.fetch_data(subjects=[0], path=physionet_tmpdir)
    assert _keep_basename_only(paths[0]) == [
        "ST7011J0-PSG.edf",
        "ST7011JP-Hypnogram.edf",
    ]

    EXPECTED_CALLS = (
        {
            "name": "ST7011J0-PSG.edf",
            "hash": "b9d11484126ebff1884034396d6a20c62c0ef48d",
        },
        {
            "name": "ST7011JP-Hypnogram.edf",
            "hash": "ff28e5e01296cefed49ae0c27cfb3ebc42e710bf",
        },
    )
    base_path = temazepam.data_path(path=physionet_tmpdir)
    _check_mocked_function_calls(fake_retrieve, EXPECTED_CALLS, base_path)

    with pytest.raises(ValueError, match="This dataset contains subjects 0 to 21"):
        paths = temazepam.fetch_data(subjects=[22], path=physionet_tmpdir)
