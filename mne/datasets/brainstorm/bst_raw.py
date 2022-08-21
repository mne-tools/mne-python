# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD-3-Clause
from functools import partial

from ...utils import verbose, get_config
from ..utils import (has_dataset, _get_version, _version_doc,
                     _data_path_doc_accept, _download_mne_dataset)


has_brainstorm_data = partial(has_dataset, name='bst_raw')

_description = """
URL: http://neuroimage.usc.edu/brainstorm/DatasetMedianNerveCtf
    - One subject, one acquisition run of 6 minutes
    - Subject stimulated using Digitimer Constant Current Stimulator
      (model DS7A)
    - The run contains 200 electric stimulations randomly distributed between
      left and right:
        - 102 stimulations of the left hand
        - 98 stimulations of the right hand
    - Inter-stimulus interval: jittered between [1500, 2000]ms
    - Stimuli generated using PsychToolBox on Windows PC (TTL pulse generated
      with the parallel port connected to the Digitimer via the rear panel BNC)
"""


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, accept=False, *, verbose=None):    # noqa: D103
    return _download_mne_dataset(
        name='bst_raw', processor='nested_untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download, accept=accept)


_data_path_doc = _data_path_doc_accept.format(
    name='brainstorm', conf='MNE_DATASETS_BRAINSTORM_DATA_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_raw) dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('bst_raw')


get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():  # noqa: D103
    """Get description of brainstorm (bst_raw) dataset."""
    for desc in _description.splitlines():
        print(desc)


def _skip_bstraw_data():
    skip_testing = (get_config('MNE_SKIP_TESTING_DATASET_TESTS', 'false') ==
                    'true')
    skip = skip_testing or not has_brainstorm_data()
    return skip


def requires_bstraw_data(func):
    """Skip testing data test."""
    import pytest
    return pytest.mark.skipif(_skip_bstraw_data(),
                              reason='Requires brainstorm dataset')(func)
