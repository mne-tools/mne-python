# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from functools import partial

from ...utils import verbose, _soft_import
from ..utils import (has_dataset, _get_path, _data_path_doc,
                     _get_version, _version_doc)
from ..config import sample
from ..fetch import fetch_dataset

has_sample_data = partial(has_dataset, name='sample')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'sample': sample}
    config_key = sample['config_key']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='sample')

    # instantiate processor that unzips file
    processor = pooch.Untar(extract_dir=path)

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(name='sample',
                                          conf='MNE_DATASETS_SAMPLE_PATH')


def get_version():  # noqa: D103
    return _get_version('sample')


get_version.__doc__ = _version_doc.format(name='sample')
