# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD Style.


def fetch_dataset(dataset_params, processor=None, path=None,
                  force_update=False, update_path=True, download=True,
                  check_version=False, return_version=False, accept=False,
                  auth=None):
    """Fetch an MNE-compatible dataset.

    Parameters
    ----------
    dataset_params : dict of dict
        The dataset name and corresponding parameters to download each dataset.
        The dataset parameters that contains the following keys:
        ``archive_name``, ``url``, ``folder_name``, ``hash``,
        ``config_key`` (optional). See Notes.
    processor : None | "zip" | "tar" | instance of pooch.Unzip |
            instance of pooch.Untar
        The processor to handle the downloaded file. If ``None`` (default),
        the files are left as is. If ``'zip'``, or ``'tar'`` will use
        our internally defined `pooch.Unzip` or `pooch.Untar`.
    path : None | str
        Location of where to look for the {name} dataset.
        If None, the environment variable or config parameter
        ``{conf}`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the {name} dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the {name} dataset even if a local copy exists.
        Default is False.
    update_path : bool | None
        If True (default), set the ``{conf}`` in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the {name} dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    check_version : bool
        Whether to check the version of the dataset or not. Each version
        of the dataset is stored in the root with a ``version.txt`` file.
    return_version : bool
        Whether or not to return the version of the dataset or not.
        Defaults to False.
    accept : bool
        Some MNE datasets require an acceptance of an additional license.
        Default to False.
    auth : tuple | None
        Optional authorization tuple containing the username and
        password/token. For example, ``auth=('foo', 012345)``.

    Returns
    -------
    data_path : str
        The path to the fetched dataset.
    version : str
        Only returned if ``return_version`` is True.

    Notes
    -----
    Fetching datasets uses the :mod:`pooch` module, but imposes additional
    structure for MNE-style datasets. The ``dataset_params`` argument takes in
    multiple dictionaries. Each dictionary corresponds to a dataset. This
    allows one to extract multiple zipped files corresponding to one dataset.
    One must define the following keys in the ``dataset_params`` dictionary
    for each dataset name:

    - ``archive_name``: This is the name of the archived file.
    - ``url``: This is the URL at which the files are downloaded from.
    - ``folder_name``: This is the folder name in which the uncompressed
    data will be stored.
    - ``hash``: This is the hash of the downloaded archive file.
    - ``config_key`` (optional): This is an MNE-Python environment variable
    configuration key. This is only used internally by MNE developers.

    An example would look like:

    {
        'sample': {
            'archive_name'='MNE-sample-data-processed.tar.gz',
            'hash'='md5:12b75d1cb7df9dfb4ad73ed82f61094f',
            'url'='https://osf.io/86qa2/download?version=5',
            'folder_name'='MNE-sample-data',
            'config_key'='MNE_DATASETS_SAMPLE_PATH',
        },
    }

    Fetching datasets downloads files over HTTP/HTTPS.
    """
    from mne.datasets.utils import _data_path

    return _data_path(
        dataset_params=dataset_params, processor=processor,
        path=path, force_update=force_update,
        update_path=update_path, download=download,
        check_version=check_version, return_version=return_version,
        accept=accept, auth=auth)
