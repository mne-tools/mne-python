import pytest

from .age.sleep_physionet_age import _update_sleep_records
from .age.sleep_physionet_age import fetch_data
from ...utils import _TempDir, run_tests_if_main, requires_good_network

@requires_good_network
def test_run_update():
    """Test Sleep Physionet URL handling."""
    _update_sleep_records()


@requires_good_network
def test_sleep_physionet_age():
    """Test Sleep Physionet URL handling."""
    data_dir = _TempDir()
    paths = fetch_data(subjects=[0], path=data_dir, update_path=False)
    assert len(paths) == 1
    assert len(paths[0]) == 2
    assert paths[0][0].endswith('.edf')
    assert paths[0][1].endswith('.edf')


@pytest.mark.skip(reason="wip")
def test_xx():
    """Test Sleep Physionet URL handling."""
    _ = fetch_data(subjects=[0])
    # import pdb; pdb.set_trace()
    print('hi')


run_tests_if_main()
