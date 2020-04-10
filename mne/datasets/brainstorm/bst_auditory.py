# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

from functools import partial

from ...utils import verbose
from ..utils import (has_dataset, _data_path, _get_version, _version_doc,
                     _data_path_doc)

has_brainstorm_data = partial(has_dataset, name='brainstorm.bst_auditory')


_description = u"""
URL: http://neuroimage.usc.edu/brainstorm/DatasetAuditory
    - One subject, two acquisition runs of 6 minutes each
    - Subject stimulated binaurally with intra-aural earphones
      (air tubes+transducers)
    - Each run contains:
        - 200 regular beeps (440Hz)
        - 40 easy deviant beeps (554.4Hz, 4 semitones higher)
    - Random inter-stimulus interval: between 0.7s and 1.7s seconds, uniformly
      distributed
    - The subject presses a button when detecting a deviant with the right
      index finger
    - Auditory stimuli generated with the Matlab Psychophysics toolbox
"""


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              verbose=None):  # noqa: D103
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name='brainstorm',
                      download=download, archive_name='bst_auditory.tar.gz')


_data_path_doc = _data_path_doc.format(name='brainstorm',
                                       conf='MNE_DATASETS_BRAINSTORM_DATA'
                                            '_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_auditory) dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('brainstorm.bst_auditory')


get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():
    """Get description of brainstorm (bst_auditory) dataset."""
    for desc in _description.splitlines():
        print(desc)
