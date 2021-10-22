# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose, deprecated
from ..utils import (has_dataset, _data_path_doc, DEPRECATION_MESSAGE_TEMPLATE,
                     _get_version, _version_doc, _download_mne_dataset,
                     _HAS_DATA_DOCSTRING_TEMPLATE)


@deprecated(extra=DEPRECATION_MESSAGE_TEMPLATE.format('sample'))
def has_sample_data():  # noqa: D103
    return has_dataset(name='sample')


has_sample_data.__doc__ = _HAS_DATA_DOCSTRING_TEMPLATE.format('sample')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103

    return _download_mne_dataset(
        name='sample', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


data_path.__doc__ = _data_path_doc.format(name='sample',
                                          conf='MNE_DATASETS_SAMPLE_PATH')


def get_version():  # noqa: D103
    return _get_version('sample')


get_version.__doc__ = _version_doc.format(name='sample')
