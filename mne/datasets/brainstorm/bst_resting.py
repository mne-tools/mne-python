# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD-3-Clause
import os.path as op
from functools import partial

from ...utils import verbose, _soft_import
from ..utils import (has_dataset, _get_version, _version_doc,
                     _data_path_doc_accept, _get_path)
from ..config import bst_resting
from ..fetch import fetch_dataset

has_brainstorm_data = partial(has_dataset, name='bst_resting')

_description = u"""
URL: http://neuroimage.usc.edu/brainstorm/DatasetResting
    - One subject
    - Two runs of 10 min of resting state recordings
    - Eyes open
"""


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              *, accept=False, verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'bst_resting': bst_resting}
    config_key = bst_resting['config_key']
    folder_name = bst_resting['folder_name']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='bst_resting')

    # instantiate processor that unzips file
    # this processor handles nested tar files
    processor = pooch.Untar(extract_dir=op.join(path, folder_name))

    return fetch_dataset(dataset_params=dataset_params, processor=processor,
                         path=path, force_update=force_update,
                         update_path=update_path, download=download,
                         accept=accept)


_data_path_doc = _data_path_doc_accept.format(
    name='brainstorm', conf='MNE_DATASETS_BRAINSTORM_DATA_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_resting) dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('bst_resting')


get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():
    """Get description of brainstorm (bst_resting) dataset."""
    for desc in _description.splitlines():
        print(desc)
