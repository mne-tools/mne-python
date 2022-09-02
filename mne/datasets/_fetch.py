# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD Style.

import logging
import sys
import os
import os.path as op
from pathlib import Path
from shutil import rmtree

from .. import __version__ as mne_version
from ..utils import logger, warn, _safe_input
from .config import (
    _bst_license_text,
    RELEASES,
    TESTING_VERSIONED,
    MISC_VERSIONED,
)
from .utils import _dataset_version, _do_path_update, _get_path
from ..fixes import _compare_version


_FAKE_VERSION = None  # used for monkeypatching while testing versioning


def fetch_dataset(
    dataset_params,
    processor=None,
    path=None,
    force_update=False,
    update_path=True,
    download=True,
    check_version=False,
    return_version=False,
    accept=False,
    auth=None,
    token=None,
):
    """Fetch an MNE-compatible dataset using pooch.

    Parameters
    ----------
    dataset_params : list of dict | dict
        The dataset name(s) and corresponding parameters to download the
        dataset(s). The dataset parameters that contains the following keys:
        ``archive_name``, ``url``, ``folder_name``, ``hash``,
        ``config_key`` (optional). See Notes.
    processor : None | "unzip" | "untar" | instance of pooch.Unzip | instance of pooch.Untar
        What to do after downloading the file. ``"unzip"`` and ``"untar"`` will
        decompress the downloaded file in place; for custom extraction (e.g.,
        only extracting certain files from the archive) pass an instance of
        :class:`pooch.Unzip` or :class:`pooch.Untar`. If ``None`` (the
        default), the files are left as-is.
    path : None | str
        Directory in which to put the dataset. If ``None``, the dataset
        location is determined by first checking whether
        ``dataset_params['config_key']`` is defined, and if so, whether that
        config key exists in the MNE-Python config file. If so, the configured
        path is used; if not, the location is set to the value of the
        ``MNE_DATA`` config key (if it exists), or ``~/mne_data`` otherwise.
    force_update : bool
        Force update of the dataset even if a local copy exists.
        Default is False.
    update_path : bool | None
        If True (default), set the mne-python config to the given
        path. If None, the user is prompted.
    download : bool
        If False and the dataset has not been downloaded yet, it will not be
        downloaded and the path will be returned as ``''`` (empty string). This
        is mostly used for testing purposes and can be safely ignored by most
        users.
    check_version : bool
        Whether to check the version of the dataset or not. Each version
        of the dataset is stored in the root with a ``version.txt`` file.
    return_version : bool
        Whether or not to return the version of the dataset or not.
        Defaults to False.
    accept : bool
        Some MNE-supplied datasets require acceptance of an additional license.
        Default is ``False``.
    auth : tuple | None
        Optional authentication tuple containing the username and
        password/token, passed to :class:`pooch.HTTPDownloader` (e.g.,
        ``auth=('foo', 012345)``).
    token : str | None
        Optional authentication token passed to :class:`pooch.HTTPDownloader`.

    Returns
    -------
    data_path : instance of Path
        The path to the fetched dataset.
    version : str
        Only returned if ``return_version`` is True.

    See Also
    --------
    mne.get_config
    mne.set_config
    mne.datasets.has_dataset

    Notes
    -----
    The ``dataset_params`` argument must contain the following keys:

    - ``archive_name``: The name of the (possibly compressed) file to download
    - ``url``: URL from which the file can be downloaded
    - ``folder_name``: the subfolder within the ``MNE_DATA`` folder in which to
        save and uncompress (if needed) the file(s)
    - ``hash``: the cryptographic hash type of the file followed by a colon and
        then the hash value (examples: "sha256:19uheid...", "md5:upodh2io...")
    - ``config_key`` (optional): key passed to :func:`mne.set_config` to store
        the on-disk location of the downloaded dataset (e.g.,
        ``"MNE_DATASETS_EEGBCI_PATH"``). This will only work for the provided
        datasets listed :ref:`here <datasets>`; do not use for user-defined
        datasets.

    An example would look like::

        {'dataset_name': 'sample',
         'archive_name': 'MNE-sample-data-processed.tar.gz',
         'hash': 'md5:12b75d1cb7df9dfb4ad73ed82f61094f',
         'url': 'https://osf.io/86qa2/download?version=5',
         'folder_name': 'MNE-sample-data',
         'config_key': 'MNE_DATASETS_SAMPLE_PATH'}

    For datasets where a single (possibly compressed) file must be downloaded,
    pass a single :class:`dict` as ``dataset_params``. For datasets where
    multiple files must be downloaded and (optionally) uncompressed separately,
    pass a list of dicts.
    """  # noqa E501
    import pooch

    if auth is not None:
        if len(auth) != 2:
            raise RuntimeError(
                "auth should be a 2-tuple consisting "
                "of a username and password/token."
            )

    # processor to uncompress files
    if processor == "untar":
        processor = pooch.Untar(extract_dir=path)
    elif processor == "unzip":
        processor = pooch.Unzip(extract_dir=path)

    if isinstance(dataset_params, dict):
        dataset_params = [dataset_params]

    # extract configuration parameters
    names = [params["dataset_name"] for params in dataset_params]
    name = names[0]
    dataset_dict = dataset_params[0]
    config_key = dataset_dict.get('config_key', None)
    folder_name = dataset_dict["folder_name"]

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name=name)

    # get the actual path to each dataset folder name
    final_path = op.join(path, folder_name)

    # handle BrainStorm datasets with nested folders for datasets
    if name.startswith("bst_"):
        final_path = op.join(final_path, name)

    final_path = Path(final_path)

    # additional condition: check for version.txt and parse it
    # check if testing or misc data is outdated; if so, redownload it
    want_version = RELEASES.get(name, None)
    want_version = _FAKE_VERSION if name == "fake" else want_version

    # get the version of the dataset and then check if the version is outdated
    data_version = _dataset_version(final_path, name)
    outdated = (want_version is not None and
                _compare_version(want_version, '>', data_version))

    if outdated:
        logger.info(
            f"Dataset {name} version {data_version} out of date, "
            f"latest version is {want_version}"
        )
    empty = Path("")

    # return empty string if outdated dataset and we don't want to download
    if (not force_update) and outdated and not download:
        logger.info(
            'Dataset out of date but force_update=False and download=False, '
            'returning empty data_path')
        return (empty, data_version) if return_version else empty

    # reasons to bail early (hf_sef has separate code for this):
    if (
        (not force_update)
        and (not outdated)
        and (not name.startswith("hf_sef_"))
    ):
        # ...if target folder exists (otherwise pooch downloads every
        # time because we don't save the archive files after unpacking, so
        # pooch can't check its checksum)
        if op.isdir(final_path):
            if config_key is not None:
                _do_path_update(path, update_path, config_key, name)
            return (final_path, data_version) if return_version else final_path
        # ...if download=False (useful for debugging)
        elif not download:
            return (empty, data_version) if return_version else empty
        # ...if user didn't accept the license
        elif name.startswith("bst_"):
            if accept or "--accept-brainstorm-license" in sys.argv:
                answer = "y"
            else:
                # If they don't have stdin, just accept the license
                # https://github.com/mne-tools/mne-python/issues/8513#issuecomment-726823724  # noqa: E501
                answer = _safe_input(
                    "%sAgree (y/[n])? " % _bst_license_text, use="y")
            if answer.lower() != "y":
                raise RuntimeError(
                    "You must agree to the license to use this " "dataset"
                )
    # downloader & processors
    download_params = dict(progressbar=logger.level <= logging.INFO)
    if name == "fake":
        download_params["progressbar"] = False
    if auth is not None:
        download_params["auth"] = auth
    if token is not None:
        download_params["headers"] = {"Authorization": f"token {token}"}
    downloader = pooch.HTTPDownloader(**download_params)

    # make mappings from archive names to urls and to checksums
    urls = dict()
    registry = dict()
    for idx, this_name in enumerate(names):
        this_dataset = dataset_params[idx]
        archive_name = this_dataset["archive_name"]
        dataset_url = this_dataset["url"]
        dataset_hash = this_dataset["hash"]
        urls[archive_name] = dataset_url
        registry[archive_name] = dataset_hash

    # create the download manager
    fetcher = pooch.create(
        path=str(final_path) if processor is None else path,
        base_url="",  # Full URLs are given in the `urls` dict.
        version=None,  # Data versioning is decoupled from MNE-Python version.
        urls=urls,
        registry=registry,
        retry_if_failed=2,  # 2 retries = 3 total attempts
    )

    # use our logger level for pooch's logger too
    pooch.get_logger().setLevel(logger.getEffectiveLevel())

    for idx in range(len(names)):
        # fetch and unpack the data
        archive_name = dataset_params[idx]["archive_name"]
        fetcher.fetch(
            fname=archive_name, downloader=downloader, processor=processor
        )
        # after unpacking, remove the archive file
        if processor is not None:
            os.remove(op.join(path, archive_name))

    # remove version number from "misc" and "testing" datasets folder names
    if name == "misc":
        rmtree(final_path, ignore_errors=True)
        os.replace(op.join(path, MISC_VERSIONED), final_path)
    elif name == "testing":
        rmtree(final_path, ignore_errors=True)
        os.replace(op.join(path, TESTING_VERSIONED), final_path)

    # maybe update the config
    if config_key is not None:
        old_name = "brainstorm" if name.startswith("bst_") else name
        _do_path_update(path, update_path, config_key, old_name)

    # compare the version of the dataset and mne
    data_version = _dataset_version(path, name)
    # 0.7 < 0.7.git should be False, therefore strip
    if check_version and (
        _compare_version(data_version, '<', mne_version.strip(".git"))
    ):
        warn(
            "The {name} dataset (version {current}) is older than "
            "mne-python (version {newest}). If the examples fail, "
            "you may need to update the {name} dataset by using "
            "mne.datasets.{name}.data_path(force_update=True)".format(
                name=name, current=data_version, newest=mne_version
            )
        )
    return (final_path, data_version) if return_version else final_path
