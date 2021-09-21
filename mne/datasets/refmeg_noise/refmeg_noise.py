# Authors: Jeff Hanna <jeff.hanna@gmail.com>
# License: BSD Style.

from functools import partial

from ...utils import verbose, _soft_import
from ..utils import (has_dataset, _get_path, _data_path_doc,
                     _get_version, _version_doc)
from ..config import refmeg_noise
from ..fetch import fetch_dataset


has_refmeg_noise_data = partial(has_dataset, name='refmeg_noise')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = dict(name=refmeg_noise)
    config_key = dataset_params['config_key']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='refmeg_noise')

    # instantiate processor that unzips file
    processor = pooch.Unzip(extract_dir=path)

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(name='refmeg_noise',
                                          conf='MNE_DATASETS_REFMEG_NOISE_PATH')  # noqa


def get_version():  # noqa: D103
    return _get_version('refmeg_noise')


get_version.__doc__ = _version_doc.format(name='refmeg_noise')
