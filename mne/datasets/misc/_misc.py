# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose
from ..utils import has_dataset, _data_path_doc, _download_mne_dataset


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, *, verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='misc', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


def _pytest_mark():
    import pytest
    return pytest.mark.skipif(
        not has_dataset(name='misc'), reason='Requires misc dataset')


data_path.__doc__ = _data_path_doc.format(name='misc',
                                          conf='MNE_DATASETS_MISC_PATH')
