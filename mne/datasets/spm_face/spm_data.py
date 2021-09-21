# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD Style.

from functools import partial

from ...utils import verbose, get_config, _soft_import
from ..utils import (has_dataset, _get_path, _data_path_doc,
                     _get_version, _version_doc)
from ..config import spm
from ..fetch import fetch_dataset


has_spm_data = partial(has_dataset, name='spm')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'spm': spm}
    config_key = spm['config_key']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='spm')

    # instantiate processor that unzips file
    processor = pooch.Untar(extract_dir=path)

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(name='spm',
                                          conf='MNE_DATASETS_SPM_DATA_PATH')


def get_version():  # noqa: D103
    return _get_version('spm')


get_version.__doc__ = _version_doc.format(name='spm')


def _skip_spm_data():
    skip_testing = (get_config('MNE_SKIP_TESTING_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_spm_data()
    return skip


def requires_spm_data(func):
    """Skip testing data test."""
    import pytest
    return pytest.mark.skipif(_skip_spm_data(),
                              reason='Requires spm dataset')(func)
