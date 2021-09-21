# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD Style.

from functools import partial

from ...utils import verbose, _soft_import
from ..utils import (has_dataset, _get_path, _data_path_doc,
                     _get_version, _version_doc)
from ..config import phantom_4dbti
from ..fetch import fetch_dataset


has_phantom_4dbti_data = partial(has_dataset, name='phantom_4dbti')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'phantom_4dbti': phantom_4dbti}
    config_key = phantom_4dbti['config_key']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='phantom_4dbti')

    # instantiate processor that unzips file
    processor = pooch.Unzip(extract_dir=path)

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(
    name='phantom_4dbti', conf='MNE_DATASETS_PHANTOM_4DBTI_PATH')


def get_version():  # noqa: D103
    return _get_version('phantom_4dbti')


get_version.__doc__ = _version_doc.format(name='phantom_4dbti')
