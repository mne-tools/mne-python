# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose
from ..utils import (_data_path_doc, _download_mne_dataset,
                     _get_version, _version_doc)


@verbose
def data_path(path=None, force_update=False, update_path=False,
              download=True, *, verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='fake', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


data_path.__doc__ = _data_path_doc.format(name='fake',
                                          conf='MNE_DATASETS_FAKE_PATH')


def get_version():  # noqa: D103
    return _get_version('fake')


get_version.__doc__ = _version_doc.format(name='fake')
