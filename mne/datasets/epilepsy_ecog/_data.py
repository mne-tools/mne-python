# Authors: Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
# License: BSD Style.

from ...utils import verbose, deprecated
from ..utils import (has_dataset, _data_path_doc, DEPRECATION_MESSAGE_TEMPLATE,
                     _get_version, _version_doc, _download_mne_dataset,
                     _HAS_DATA_DOCSTRING_TEMPLATE)


@deprecated(extra=DEPRECATION_MESSAGE_TEMPLATE.format('epilepsy_ecog'))
def has_epilepsy_ecog_data():
    return has_dataset(name='epilepsy_ecog')


has_epilepsy_ecog_data.__doc__ = _HAS_DATA_DOCSTRING_TEMPLATE.format(
    'epilepsy_ecog')


@verbose
def data_path(
        path=None, force_update=False, update_path=True,
        download=True, verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='epilepsy_ecog', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


data_path.__doc__ = _data_path_doc.format(
    name='epilepsy_ecog', conf='MNE_DATASETS_EPILEPSY_ECOG_PATH')


def get_version():  # noqa: D103
    return _get_version('epilepsy_ecog')


get_version.__doc__ = _version_doc.format(name='epilepsy_ecog')
