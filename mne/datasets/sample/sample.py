# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from functools import partial

import numpy as np

from ...utils import verbose, get_config
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_sample_data = partial(has_dataset, name='sample')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='sample',
                      download=download)

data_path.__doc__ = _data_path_doc.format(name='sample',
                                          conf='MNE_DATASETS_SAMPLE_PATH')


def get_version():  # noqa: D103
    return _get_version('sample')

get_version.__doc__ = _version_doc.format(name='sample')


# Allow forcing of sample dataset skip
def _skip_sample_data():
    skip_testing = (get_config('MNE_SKIP_SAMPLE_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_sample_data()
    return skip

requires_sample_data = np.testing.dec.skipif(_skip_sample_data,
                                             'Requires sample dataset')
