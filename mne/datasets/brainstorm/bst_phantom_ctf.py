# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from functools import partial

from ...utils import verbose
from ..utils import (has_dataset, _data_path, _get_version, _version_doc,
                     _data_path_doc_accept)

has_brainstorm_data = partial(has_dataset, name='brainstorm.bst_phantom_ctf')


_description = u"""
URL: http://neuroimage.usc.edu/brainstorm/Tutorials/PhantomCtf
"""


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              *, accept=False, verbose=None):  # noqa: D103
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='brainstorm',
                      download=download, archive_name='bst_phantom_ctf.tar.gz',
                      accept=accept)


_data_path_doc = _data_path_doc_accept.format(
    name='brainstorm', conf='MNE_DATASETS_BRAINSTORM_DATA_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_phantom_ctf) dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('brainstorm.bst_phantom_ctf')


get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():
    """Get description of brainstorm (bst_phantom_ctf) dataset."""
    for desc in _description.splitlines():
        print(desc)
