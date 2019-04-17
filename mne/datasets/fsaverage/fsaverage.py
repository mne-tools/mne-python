# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD Style.

from functools import partial

from ...utils import verbose, get_config
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_fsaverage_data = partial(has_dataset, name='fsaverage')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='fsaverage',
                      download=download)


data_path.__doc__ = _data_path_doc.format(name='fsaverage',
                                          conf='MNE_DATASETS_FSAVERAGE_PATH')


def get_version():  # noqa: D103
    return _get_version('fsaverage')


get_version.__doc__ = _version_doc.format(name='fsaverage')


# Allow forcing of fsaverage dataset skip
def _skip_fsaverage_data():
    skip_testing = (get_config('MNE_SKIP_FSAVERAGE_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_fsaverage_data()
    return skip


def requires_fsaverage_data(func):
    """Skip testing data test."""
    import pytest
    return pytest.mark.skipif(_skip_fsaverage_data(),
                              reason='Requires fsaverage dataset')(func)
