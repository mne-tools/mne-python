# License: BSD Style.

from ...utils import verbose
from ..utils import (_get_version, _version_doc, _download_mne_dataset)


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, *, verbose=None):
    """Get path to local copy of the kiloword dataset.

    This is the dataset from :footcite:`DufauEtAl2015`.

    Parameters
    ----------
    path : None | str
        Location of where to look for the kiloword data storing
        location. If None, the environment variable or config parameter
        MNE_DATASETS_KILOWORD_PATH is used. If it doesn't exist,
        the "mne-python/examples" directory is used. If the
        kiloword dataset is not found under the given path (e.g.,
        as "mne-python/examples/MNE-kiloword-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_KILOWORD_PATH in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the kiloword dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    %(verbose)s

    Returns
    -------
    path : list of Path
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    References
    ----------
    .. footbibliography::
    """
    return _download_mne_dataset(
        name='kiloword', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


def get_version():
    """Get dataset version."""
    return _get_version('kiloword')


get_version.__doc__ = _version_doc.format(name='kiloword')
