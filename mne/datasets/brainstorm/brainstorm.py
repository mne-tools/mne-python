# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

from ...utils import verbose
from ...fixes import partial
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_brainstorm_data = partial(has_dataset, name='brainstorm')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='brainstorm',
                      download=download)

data_path.__doc__ = _data_path_doc.format(name='brainstorm',
                                          conf='MNE_DATASETS_BRAINSTORM_DATA_PATH')


def get_version():
    return _get_version('brainstorm')

get_version.__doc__ = _version_doc.format(name='brainstorm')
