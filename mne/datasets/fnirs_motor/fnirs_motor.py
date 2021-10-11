# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose, deprecated
from ..utils import (has_dataset, _data_path_doc, DEPRECATION_MESSAGE_TEMPLATE,
                     _get_version, _version_doc, _download_mne_dataset,
                     _HAS_DATA_DOCSTRING_TEMPLATE)


@deprecated(extra=DEPRECATION_MESSAGE_TEMPLATE.format('fnirs_motor'))
def has_fnirs_motor_data():
    return has_dataset(name='fnirs_motor')


has_fnirs_motor_data.__doc__ = _HAS_DATA_DOCSTRING_TEMPLATE.format(
    'fnirs_motor')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='fnirs_motor', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


data_path.__doc__ = _data_path_doc.format(name='fnirs_motor',
                                          conf='MNE_DATASETS_FNIRS_MOTOR_PATH')


def get_version():  # noqa: D103
    return _get_version('fnirs_motor')


get_version.__doc__ = _version_doc.format(name='fnirs_motor')
