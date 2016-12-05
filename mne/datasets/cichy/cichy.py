# License: BSD Style.

from functools import partial

import numpy as np

from ...utils import verbose, get_config
from ..utils import (has_dataset, _data_path, _data_path_doc,
                     _get_version, _version_doc)


has_cichy_data = partial(has_dataset, name='cichy')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):
    """
    Get path to local copy of Cichy dataset.

    Parameters
    ----------
    path : None | str
        Location of where to look for the Cichy data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_CICHY_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the Cichy dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MNE-cichy-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_CICHY_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.
    """
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='cichy',
                      download=download)

data_path.__doc__ = _data_path_doc.format(name='cichy',
                                          conf='MNE_DATASETS_CICHY_PATH')


def get_version():
    """Get dataset version."""
    return _get_version('cichy')

get_version.__doc__ = _version_doc.format(name='cichy')


# Allow forcing of cichy dataset skip
def _skip_cichy_data():
    skip_testing = (get_config('MNE_SKIP_CICHY_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_cichy_data()
    return skip

requires_cichy_data = np.testing.dec.skipif(_skip_cichy_data,
                                            'Requires Cichy dataset')
