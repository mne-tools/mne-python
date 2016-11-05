# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

from functools import partial
import os.path as op

from ...utils import verbose
from ..utils import (has_dataset, _data_path, _get_version, _version_doc,
                     _data_path_doc)

has_brainstorm_data = partial(has_dataset, name='brainstorm')

_description = u"""
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
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):    # noqa: D103
    archive_name = dict(brainstorm='bst_raw.tar.gz')
    data_path = _data_path(path=path, force_update=force_update,
                           update_path=update_path, name='brainstorm',
                           download=download, archive_name=archive_name)
    if data_path != '':
        return op.join(data_path, 'bst_raw')
    else:
        return data_path


_data_path_doc = _data_path_doc.format(name='brainstorm',
                                       conf='MNE_DATASETS_BRAINSTORM_DATA'
                                            '_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_raw) dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('brainstorm')

get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():  # noqa: D103
    """Get description of brainstorm (bst_raw) dataset."""
    for desc in _description.splitlines():
        print(desc)
