# Authors: Chris Holdgraf <choldgraf@berkeley.edu>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD Style.
import os.path as op
from functools import partial

from ...utils import verbose, _soft_import
from ..utils import (has_dataset, _data_path_doc,
                     _get_version, _version_doc, _get_path)
from ..config import fieldtrip_cmc
from ..fetch import fetch_dataset


data_name = "fieldtrip_cmc"
conf_name = "MNE_DATASETS_FIELDTRIP_CMC_PATH"
has_mtrf_data = partial(has_dataset, name=data_name)


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'fieldtrip_cmc': fieldtrip_cmc}
    config_key = fieldtrip_cmc['config_key']
    folder_name = fieldtrip_cmc['folder_name']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='fieldtrip_cmc')

    # instantiate processor that unzips file
    processor = pooch.Untar(extract_dir=op.join(path, folder_name))

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download)


data_path.__doc__ = _data_path_doc.format(name=data_name,
                                          conf=conf_name)


def get_version():  # noqa: D103
    return _get_version(data_name)


get_version.__doc__ = _version_doc.format(name=data_name)
