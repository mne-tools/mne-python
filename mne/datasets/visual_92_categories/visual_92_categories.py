# License: BSD Style.

from ...utils import verbose
from ..utils import (_download_mne_dataset, _data_path_doc, _get_version,
                     _version_doc)


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, *, verbose=None):
    """
    Get path to local copy of visual_92_categories dataset.

    .. note:: The dataset contains four fif-files, the trigger files and the T1
              mri image. This dataset is rather big in size (more than 5 GB).

    Parameters
    ----------
    path : None | str
        Location of where to look for the visual_92_categories data storing
        location. If None, the environment variable or config parameter
        MNE_DATASETS_VISUAL_92_CATEGORIES_PATH is used. If it doesn't exist,
        the "mne-python/examples" directory is used. If the
        visual_92_categories dataset is not found under the given path (e.g.,
        as "mne-python/examples/MNE-visual_92_categories-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_VISUAL_92_CATEGORIES_PATH in mne-python
        config to the given path. If None, the user is prompted.
    %(verbose)s

    Returns
    -------
    path : instance of Path
        Local path to the given data file.

    Notes
    -----
    The visual_92_categories dataset is documented in the following publication
        Radoslaw M. Cichy, Dimitrios Pantazis, Aude Oliva (2014) Resolving
        human object recognition in space and time. doi: 10.1038/NN.3635
    """
    return _download_mne_dataset(
        name='visual_92_categories', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


data_path.__doc__ = _data_path_doc.format(
    name='visual_92_categories', conf='MNE_DATASETS_VISUAL_92_CATEGORIES_PATH')


def get_version():
    """Get dataset version."""
    return _get_version('visual_92_categories')


get_version.__doc__ = _version_doc.format(name='visual_92_categories')
