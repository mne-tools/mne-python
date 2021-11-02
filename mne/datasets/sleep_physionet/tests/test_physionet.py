# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

import os.path as op
import numpy as np
import pytest

from numpy.testing import assert_array_equal

from mne.utils import requires_good_network
from mne.utils import requires_pandas, requires_version
from mne.datasets.sleep_physionet import age, temazepam
from mne.datasets.sleep_physionet._utils import _update_sleep_temazepam_records
from mne.datasets.sleep_physionet._utils import _update_sleep_age_records
from mne.datasets.sleep_physionet._utils import AGE_SLEEP_RECORDS
from mne.datasets.sleep_physionet._utils import TEMAZEPAM_SLEEP_RECORDS
from mne.utils.check import _soft_import


# import pooch library for handling the dataset downloading
pooch = _soft_import('pooch', 'dataset downloading', strict=True)


@pytest.fixture(scope='session')
def physionet_tmpdir(tmp_path_factory):
    """Fixture exposing a temporary directory for testing."""
    return str(tmp_path_factory.mktemp('physionet_files'))


class _FakeFetch:

    def __init__(self):
        self.call_args_list = list()

    def __call__(self, *args, **kwargs):
        self.call_args_list.append((args, kwargs))

    @property
    def call_count(self):
        return len(self.call_args_list)


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
        _, call_kwargs = mocked_func.call_args_list[idx]
        hash_type, hash = call_kwargs['known_hash'].split(':')
        assert call_kwargs['url'] == _get_expected_url(current['name'])
        assert op.join(call_kwargs['path'], call_kwargs['fname']) == \
            _get_expected_path(base_path, current['name'])
        assert hash == current['hash']
        assert hash_type == 'sha1'


@pytest.mark.timeout(60)
@pytest.mark.xfail(strict=False)
@requires_good_network
@requires_pandas
@requires_version('xlrd', '0.9')
def test_run_update_age_records(tmp_path):
    """Test Sleep Physionet URL handling."""
    import pandas as pd
    fname = op.join(str(tmp_path), "records.csv")
    _update_sleep_age_records(fname)
    data = pd.read_csv(fname)
    pd.testing.assert_frame_equal(data, pd.read_csv(AGE_SLEEP_RECORDS))


@pytest.mark.parametrize('subject', [39, 68, 69, 78, 79, 83])
def test_sleep_physionet_age_missing_subjects(physionet_tmpdir, subject,
                                              download_is_error):
    """Test handling of missing subjects in Sleep Physionet age fetcher."""
    with pytest.raises(
            ValueError, match='This dataset contains subjects 0 to 82'):
        age.fetch_data(
            subjects=[subject], recording=[1], on_missing='raise',
            path=physionet_tmpdir)
    with pytest.warns(RuntimeWarning,
                      match='This dataset contains subjects 0 to 82'):
        age.fetch_data(
            subjects=[subject], recording=[1], on_missing='warn',
            path=physionet_tmpdir)
    paths = age.fetch_data(
        subjects=[subject], recording=[1], on_missing='ignore',
        path=physionet_tmpdir)
    assert paths == []


@pytest.mark.parametrize('subject,recording', [(13, 2), (36, 1), (52, 1)])
def test_sleep_physionet_age_missing_recordings(physionet_tmpdir, subject,
                                                recording, download_is_error):
    """Test handling of missing recordings in Sleep Physionet age fetcher."""
    with pytest.raises(
            ValueError, match=f'Requested recording {recording} for subject'):
        age.fetch_data(subjects=[subject], recording=[recording],
                       on_missing='raise', path=physionet_tmpdir)
    with pytest.warns(RuntimeWarning,
                      match=f'Requested recording {recording} for subject'):
        age.fetch_data(subjects=[subject], recording=[recording],
                       on_missing='warn', path=physionet_tmpdir)
    paths = age.fetch_data(subjects=[subject], recording=[recording],
                           on_missing='ignore', path=physionet_tmpdir)
    assert paths == []


def test_sleep_physionet_age(physionet_tmpdir, monkeypatch, download_is_error):
    """Test Sleep Physionet URL handling."""
    # check download_is_error patching
    with pytest.raises(AssertionError, match='Test should not download'):
        age.fetch_data(subjects=[0], recording=[1], path=physionet_tmpdir)
    # then patch
    my_func = _FakeFetch()
    monkeypatch.setattr(pooch, 'retrieve', my_func)

    paths = age.fetch_data(subjects=[0], recording=[1], path=physionet_tmpdir)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf']])

    paths = age.fetch_data(subjects=[0, 1], recording=[1],
                           path=physionet_tmpdir)
    assert_array_equal(_keep_basename_only(paths),
                       [['SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf'],
                        ['SC4011E0-PSG.edf', 'SC4011EH-Hypnogram.edf']])

    paths = age.fetch_data(subjects=[0], recording=[1, 2],
                           path=physionet_tmpdir)
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
    _check_mocked_function_calls(my_func, EXPECTED_CALLS, base_path)


@pytest.mark.xfail(strict=False)
@requires_good_network
@requires_pandas
@requires_version('xlrd', '0.9')
def test_run_update_temazepam_records(tmp_path):
    """Test Sleep Physionet URL handling."""
    import pandas as pd
    fname = op.join(str(tmp_path), "records.csv")
    _update_sleep_temazepam_records(fname)
    data = pd.read_csv(fname)

    pd.testing.assert_frame_equal(
        data, pd.read_csv(TEMAZEPAM_SLEEP_RECORDS))


def test_sleep_physionet_temazepam(physionet_tmpdir, monkeypatch):
    """Test Sleep Physionet URL handling."""
    my_func = _FakeFetch()
    monkeypatch.setattr(pooch, 'retrieve', my_func)

    paths = temazepam.fetch_data(subjects=[0], path=physionet_tmpdir)
    assert_array_equal(_keep_basename_only(paths),
                       [['ST7011J0-PSG.edf', 'ST7011JP-Hypnogram.edf']])

    EXPECTED_CALLS = (
        {'name': 'ST7011J0-PSG.edf',
         'hash': 'b9d11484126ebff1884034396d6a20c62c0ef48d'},
        {'name': 'ST7011JP-Hypnogram.edf',
         'hash': 'ff28e5e01296cefed49ae0c27cfb3ebc42e710bf'})
    base_path = temazepam.data_path(path=physionet_tmpdir)
    _check_mocked_function_calls(my_func, EXPECTED_CALLS, base_path)

    with pytest.raises(
            ValueError, match='This dataset contains subjects 0 to 21'):
        paths = temazepam.fetch_data(subjects=[22], path=physionet_tmpdir)
