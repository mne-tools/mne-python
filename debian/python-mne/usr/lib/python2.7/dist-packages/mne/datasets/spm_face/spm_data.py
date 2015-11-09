# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD Style.

from ...utils import verbose
from ...fixes import partial
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_spm_data = partial(has_dataset, name='spm')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='spm',
                      download=download)

data_path.__doc__ = _data_path_doc.format(name='spm',
                                          conf='MNE_DATASETS_SPM_DATA_PATH')


def get_version():
    return _get_version('spm')

get_version.__doc__ = _version_doc.format(name='spm')
