# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os.path as op
from ...utils import verbose
from ...fixes import partial
from ..utils import has_dataset, _data_path, _get_version, _version_doc


has_brainstorm_data = partial(has_dataset, name='brainstorm')


@verbose
def data_path(path=None, force_update=False, update_path=True, download=True,
              archive='DatasetAuditory', verbose=None):
    """Get path to local copy of brainstorm dataset

    Parameters
    ----------
    path : None | str
        Location of where to look for the brainstorm dataset.
        If None, the environment variable or config parameter
         MNE_DATASETS_BRAINSTORM_DATA_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the brainstorm dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MNE-brainstorm-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the brainstorm dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BRAINSTORM_DATA_PATH in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the brainstorm dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    archive : str
        The archive to fetch. Must be one of 'DatasetAuditory',
        'DatasetResting' or 'DatasetMedianNerveCtf'
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    path : str
        Path to brainstorm dataset directory.
    """
    if archive not in ['DatasetAuditory', 'DatasetResting',
                       'DatasetMedianNerveCtf']:
        raise ValueError('archive must be one of DatasetAuditory,'
                         ' DatasetResting, and DatasetMedianNerveCtf')
    archive2file = dict(DatasetAuditory='sample_auditory',
                        DatasetResting='sample_resting',
                        DatasetMedianNerveCtf='sample_raw')
    archive_name = dict(brainstorm=archive2file[archive])
    data_path = _data_path(path=path, force_update=force_update,
                           update_path=update_path, name='brainstorm',
                           download=download, archive_name=archive_name)
    return op.join(data_path, archive_name['brainstorm'])


def get_version():
    return _get_version('brainstorm')

get_version.__doc__ = _version_doc.format(name='brainstorm')
