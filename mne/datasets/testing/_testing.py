# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from functools import partial

from ...utils import verbose, get_config
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_testing_data = partial(has_dataset, name='testing')


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, verbose=None):  # noqa: D103
    # Make sure we don't do something stupid
    if download and \
            get_config('MNE_SKIP_TESTING_DATASET_TESTS', 'false') == 'true':
        raise RuntimeError('Cannot download data if skipping is forced')
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='testing',
                      download=download)


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
    import pytest
    return pytest.mark.skipif(_skip_testing_data(),
                              reason='Requires testing dataset')(func)


def _pytest_param():
    import pytest
    # turn anything that uses testing data into an auto-skipper by
    # setting params=[testing._pytest_param()]
    return pytest.param('testing_data', marks=pytest.mark.skipif(
        _skip_testing_data(), reason='Requires testing dataset'))
