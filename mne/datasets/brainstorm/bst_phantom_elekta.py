# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
from ...utils import verbose
from ..utils import (_get_version, _version_doc,
                     _data_path_doc_accept, _download_mne_dataset)

_description = u"""
URL: http://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta
"""


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, accept=False, *, verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='bst_phantom_elekta', processor='nested_untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download, accept=accept)


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
