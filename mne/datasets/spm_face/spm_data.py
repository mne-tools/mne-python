# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD Style.

from functools import partial

import numpy as np

from ...utils import verbose, get_config
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_spm_data = partial(has_dataset, name='spm')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='spm',
                      download=download)

data_path.__doc__ = _data_path_doc.format(name='spm',
                                          conf='MNE_DATASETS_SPM_DATA_PATH')


def get_version():  # noqa: D103
    return _get_version('spm')

get_version.__doc__ = _version_doc.format(name='spm')


def _skip_spm_data():
    skip_testing = (get_config('MNE_SKIP_TESTING_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_spm_data()
    return skip

requires_spm_data = np.testing.dec.skipif(_skip_spm_data,
                                          'Requires spm dataset')
