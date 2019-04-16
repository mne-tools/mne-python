# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op
import numpy as np
import pytest

from distutils.version import LooseVersion
from numpy.testing import assert_array_equal

from mne.utils import run_tests_if_main, requires_good_network
from mne.utils import requires_pandas, requires_version
from mne.datasets.sleep_physionet import age, temazepam
from mne.datasets.sleep_physionet._utils import _update_sleep_temazepam_records
from mne.datasets.sleep_physionet._utils import _update_sleep_age_records
from mne.datasets.sleep_physionet._utils import AGE_SLEEP_RECORDS
from mne.datasets.sleep_physionet._utils import TEMAZEPAM_SLEEP_RECORDS


@pytest.fixture(scope='session')
def physionet_tmpdir(tmpdir_factory):
    """Fixture exposing a temporary directory for testing."""
    return str(tmpdir_factory.mktemp('physionet_files'))


def _fake_fetch_file(url, path, print_destination, hash_, hash_type):
    pass


def _keep_basename_only(path_structure):
    return np.vectorize(op.basename)(np.array(path_structure))


def _get_expected_url(name):
    base = 'https://physionet.org/physiobank/database/sleep-edfx/'
    midle = 'sleep-cassette/' if name.startswith('SC') else 'sleep-telemetry/'
    return base + midle + '/' + name


def _get_expected_path(base, name):
    return op.join(base, name)


def _check_mocked_function_calls(mocked_func, call_fname_hash_pairs,
                                 base_path):
    # Check mocked_func has been called the right amount of times.
    assert mocked_func.call_count == len(call_fname_hash_pairs)

    # Check it has been called with the right parameters in the right
    # order.
    for idx, current in enumerate(call_fname_hash_pairs):
        call_args, call_kwargs = mocked_func.call_args_list[idx]
        assert call_args[0] == _get_expected_url(current['name'])
        assert call_args[1] == _get_expected_path(base_path, current['name'])
        assert call_kwargs['hash_'] == current['hash']
        assert call_kwargs['hash_type'] == 'sha1'
        assert call_kwargs['print_destination'] is False


@pytest.mark.timeout(60)
@pytest.mark.xfail(strict=False)
@requires_good_network
@requires_pandas
@requires_version('xlrd', '0.9')
def test_run_update_age_records(tmpdir):
    """Test Sleep Physionet URL handling."""
    import pandas as pd
    fname = op.join(str(tmpdir), "records.csv")
    _update_sleep_age_records(fname)
    data = pd.read_csv(fname)

    if LooseVersion(pd.__version__) < LooseVersion('0.23.0'):
        expected = pd.read_csv(AGE_SLEEP_RECORDS)
        assert_array_equal(
            data[['subject', 'sha', 'fname']].values,
            expected[['subject', 'sha', 'fname']].values,
        )
    else:
        pd.testing.assert_frame_equal(data, pd.read_csv(AGE_SLEEP_RECORDS))


def test_sleep_physionet_age(physionet_tmpdir, mocker):
    """Test Sleep Physionet URL handling."""
    my_func = mocker.patch('mne.datasets.sleep_physionet._utils._fetch_file',
                           side_effect=_fake_fetch_file)

    params = {'path': physionet_tmpdir, 'update_path': False}

    with pytest.raises(ValueError, match='Only subjects 0 to 19 are'):
        paths = age.fetch_data(subjects=[20], recording=[1], **params)

    paths = age.fetch_data(subjects=[0], recording=[1], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf']])

    paths = age.fetch_data(subjects=[0, 1], recording=[1], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4011E0-PSG.edf', 'SC4011EH-Hypnogram.edf']])

    paths = age.fetch_data(subjects=[0], recording=[1, 2], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4002E0-PSG.edf', 'SC4002EC-Hypnogram.edf']])

    EXPECTED_CALLS = (
        {'name': 'SC4001E0-PSG.edf',
         'hash': 'adabd3b01fc7bb75c523a974f38ee3ae4e57b40f'},
        {'name': 'SC4001EC-Hypnogram.edf',
         'hash': '21c998eadc8b1e3ea6727d3585186b8f76e7e70b'},
        {'name': 'SC4001E0-PSG.edf',
         'hash': 'adabd3b01fc7bb75c523a974f38ee3ae4e57b40f'},
        {'name': 'SC4001EC-Hypnogram.edf',
         'hash': '21c998eadc8b1e3ea6727d3585186b8f76e7e70b'},
        {'name': 'SC4011E0-PSG.edf',
         'hash': '4d17451f7847355bcab17584de05e7e1df58c660'},
        {'name': 'SC4011EH-Hypnogram.edf',
         'hash': 'd582a3cbe2db481a362af890bc5a2f5ca7c878dc'},
        {'name': 'SC4001E0-PSG.edf',
         'hash': 'adabd3b01fc7bb75c523a974f38ee3ae4e57b40f'},
        {'name': 'SC4001EC-Hypnogram.edf',
         'hash': '21c998eadc8b1e3ea6727d3585186b8f76e7e70b'},
        {'name': 'SC4002E0-PSG.edf',
         'hash': 'c6b6d7a8605cc7e7602b6028ee77f6fbf5f7581d'},
        {'name': 'SC4002EC-Hypnogram.edf',
         'hash': '386230188a3552b1fc90bba0fb7476ceaca174b6'})
    base_path = age.data_path(path=physionet_tmpdir)
    _check_mocked_function_calls(mocked_func=my_func,
                                 call_fname_hash_pairs=EXPECTED_CALLS,
                                 base_path=base_path)


@pytest.mark.xfail(strict=False)
@requires_good_network
@requires_pandas
@requires_version('xlrd', '0.9')
def test_run_update_temazepam_records(tmpdir):
    """Test Sleep Physionet URL handling."""
    import pandas as pd
    fname = op.join(str(tmpdir), "records.csv")
    _update_sleep_temazepam_records(fname)
    data = pd.read_csv(fname)

    if LooseVersion(pd.__version__) < LooseVersion('0.23.0'):
        expected = pd.read_csv(TEMAZEPAM_SLEEP_RECORDS)
        assert_array_equal(
            data[['subject', 'sha_Hypnogram', 'sha_PSG']].values,
            expected[['subject', 'sha_Hypnogram', 'sha_PSG']].values,
        )
    else:
        pd.testing.assert_frame_equal(
            data, pd.read_csv(TEMAZEPAM_SLEEP_RECORDS))


def test_sleep_physionet_temazepam(physionet_tmpdir, mocker):
    """Test Sleep Physionet URL handling."""
    my_func = mocker.patch('mne.datasets.sleep_physionet._utils._fetch_file',
                           side_effect=_fake_fetch_file)

    params = {'path': physionet_tmpdir, 'update_path': False}

    paths = temazepam.fetch_data(subjects=[0], **params)
    assert_array_equal(_keep_basename_only(paths),
                       [['ST7011J0-PSG.edf', 'ST7011JP-Hypnogram.edf']])

    EXPECTED_CALLS = (
        {'name': 'ST7011J0-PSG.edf',
         'hash': 'b9d11484126ebff1884034396d6a20c62c0ef48d'},
        {'name': 'ST7011JP-Hypnogram.edf',
         'hash': 'ff28e5e01296cefed49ae0c27cfb3ebc42e710bf'})
    base_path = temazepam.data_path(path=physionet_tmpdir)
    _check_mocked_function_calls(mocked_func=my_func,
                                 call_fname_hash_pairs=EXPECTED_CALLS,
                                 base_path=base_path)

    with pytest.raises(ValueError, match='Only subjects 0 to 21 are'):
        paths = temazepam.fetch_data(subjects=[22], **params)


run_tests_if_main()
