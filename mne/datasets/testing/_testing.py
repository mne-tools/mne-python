# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from functools import partial

from ...utils import verbose, get_config, _soft_import
from ..utils import (has_dataset, _data_path_doc,
                     _get_version, _version_doc, _get_path)
from ..config import testing
from ..fetch import fetch_dataset

has_testing_data = partial(has_dataset, name='testing')


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, verbose=None):  # noqa: D103
    # Make sure we don't do something stupid
    if download and \
            get_config('MNE_SKIP_TESTING_DATASET_TESTS', 'false') == 'true':
        raise RuntimeError('Cannot download data if skipping is forced')

    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = dict(testing=testing)
    config_key = testing['config_key']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='testing')

    # instantiate processor that unzips file
    processor = pooch.Untar(extract_dir=path)

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(name='testing',
                                          conf='MNE_DATASETS_TESTING_PATH')


def get_version():  # noqa: D103
    return _get_version('testing')


get_version.__doc__ = _version_doc.format(name='testing')


# Allow forcing of testing dataset skip (for Debian tests) using:
# `make test-no-testing-data`
def _skip_testing_data():
    skip_testing = (get_config('MNE_SKIP_TESTING_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_testing_data()
    return skip


def requires_testing_data(func):
    """Skip testing data test."""
    return _pytest_mark()(func)


def _pytest_param(*args, **kwargs):
    if len(args) == len(kwargs) == 0:
        args = ('testing_data',)
    import pytest
    # turn anything that uses testing data into an auto-skipper by
    # setting params=[testing._pytest_param()], or by parametrizing functions
    # with testing._pytest_param(whatever)
    return pytest.param(*args, **kwargs, marks=_pytest_mark())


def _pytest_mark():
    import pytest
    return pytest.mark.skipif(
        _skip_testing_data(), reason='Requires testing dataset')
