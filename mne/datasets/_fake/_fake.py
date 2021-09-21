# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose, _soft_import
from ..utils import (_data_path_doc, _get_path,
                     _get_version, _version_doc)
from ..config import fake
from ..fetch import fetch_dataset


@verbose
def data_path(path=None, force_update=False, update_path=False,
              download=True, verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'fake': fake}
    config_key = fake['config_key']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='fake')

    # instantiate processor that unzips file
    processor = pooch.Untar(extract_dir=path)

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(name='fake',
                                          conf='MNE_DATASETS_FAKE_PATH')


def get_version():  # noqa: D103
    return _get_version('fake')


get_version.__doc__ = _version_doc.format(name='fake')
