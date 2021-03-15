# Authors: Dominik Welke <dominik.welke@web.de>
# License: BSD Style.

from functools import partial

from ...utils import verbose
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)

has_ssvep_data = partial(has_dataset, name='clinical')


@verbose
def data_path(
        path=None, force_update=False, update_path=True,
        download=True, verbose=None):  # noqa: D103
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='clinical',
                      download=download)


data_path.__doc__ = _data_path_doc.format(name='clinical',
                                          conf='MNE_DATASETS_CLINICAL_PATH')


def get_version():  # noqa: D103
    return _get_version('clinical')


get_version.__doc__ = _version_doc.format(name='clinical')
