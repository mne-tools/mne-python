# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
import os.path as op
from functools import partial

from ...utils import verbose, _soft_import
from ..utils import (has_dataset, _get_version, _version_doc,
                     _data_path_doc_accept, _get_path)
from ..config import bst_phantom_elekta
from ..fetch import fetch_dataset

has_brainstorm_data = partial(has_dataset,
                              name='bst_phantom_elekta')


_description = u"""
URL: http://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta
"""


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              *, accept=False, verbose=None):  # noqa: D103
    # import pooch library for handling the dataset downloading
    pooch = _soft_import('pooch', 'dataset downloading', strict=True)
    dataset_params = {'bst_phantom_elekta': bst_phantom_elekta}
    config_key = bst_phantom_elekta['config_key']
    folder_name = bst_phantom_elekta['folder_name']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name='bst_phantom_elekta')

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
                                        'brainstorm (bst_phantom_elekta) '
                                        'dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('bst_phantom_elekta')


get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():
    """Get description of brainstorm (bst_phantom_elekta) dataset."""
    for desc in _description.splitlines():
        print(desc)
