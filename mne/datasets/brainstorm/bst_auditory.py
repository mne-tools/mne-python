# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD-3-Clause
from ...utils import verbose
from ..utils import (_get_version, _version_doc,
                     _data_path_doc_accept, _download_mne_dataset)

_description = """
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
def data_path(path=None, force_update=False, update_path=True,
              download=True, accept=False, *, verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='bst_auditory', processor='nested_untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download, accept=accept)


_data_path_doc = _data_path_doc_accept.format(
    name='brainstorm', conf='MNE_DATASETS_BRAINSTORM_DATA_PATH')
_data_path_doc = _data_path_doc.replace('brainstorm dataset',
                                        'brainstorm (bst_auditory) dataset')
data_path.__doc__ = _data_path_doc


def get_version():  # noqa: D103
    return _get_version('bst_auditory')


get_version.__doc__ = _version_doc.format(name='brainstorm')


def description():
    """Get description of brainstorm (bst_auditory) dataset."""
    for desc in _description.splitlines():
        print(desc)
