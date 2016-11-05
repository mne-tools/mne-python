# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from functools import partial
import os.path as op

from ...utils import verbose
from ..utils import (has_dataset, _data_path, _get_version, _version_doc,
                     _data_path_doc)

has_brainstorm_data = partial(has_dataset, name='brainstorm')


_description = u"""
URL: http://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta
"""


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    archive_name = dict(brainstorm='bst_phantom_elekta.tar.gz')
    data_path = _data_path(path=path, force_update=force_update,
                           update_path=update_path, name='brainstorm',
                           download=download, archive_name=archive_name)
    if data_path != '':
        return op.join(data_path, 'bst_phantom_elekta')
    else:
        return data_path

_data_path_doc = _data_path_doc.format(name='brainstorm',
                                       conf='MNE_DATASETS_BRAINSTORM_DATA'
                                            '_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_phantom_elekta) '
                                        'dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('brainstorm')

get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():
    """Get description of brainstorm (bst_phantom_elekta) dataset."""
    for desc in _description.splitlines():
        print(desc)
