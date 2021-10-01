# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD Style.

import sys
import os
import os.path as op
from distutils.version import LooseVersion
from shutil import rmtree

from .. import __version__ as mne_version
from ..utils import logger, warn, _safe_input, _soft_import
from .config import (
    _bst_license_text,
    RELEASES,
    TESTING_VERSIONED,
    MISC_VERSIONED,
)
from .utils import _dataset_version, _do_path_update, _get_path


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
    """Fetch an MNE-compatible dataset.

    Parameters
    ----------
    dataset_params : list of dict | dict
        The dataset name(s) and corresponding parameters to download the
        dataset(s). The dataset parameters that contains the following keys:
        ``archive_name``, ``url``, ``folder_name``, ``hash``,
        ``config_key`` (optional). See Notes.
    processor : None | "unzip" | "untar" | instance of pooch.Unzip | instance of pooch.Untar
        What to do after downloading the file. ``"unzip"`` and ``"untar"` will
        decompress the downloaded file in place; for custom extraction (e.g.,
        only extracting certain files from the archive) pass an instance of
        :class:`pooch.Unzip` or :class:`pooch.Untar`. If ``None`` (the
        default), the files are left as is.
    path : None | str
        Location of where to look for the dataset.
        If None, the environment variable or ``config_key`` parameter
        is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
        Default is False.
    update_path : bool | None
        If True (default), set the mne-python config to the given
        path. If None, the user is prompted.
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
        Is passed to `pooch.HTTPDownloader`.
    token : str | None
        Optional token to be passed to `pooch.HTTPDownloader`.

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
    multiple dictionaries if there are multiple files from one dataset. Each
    dictionary corresponds to a dataset. This allows one to extract multiple
    zipped files corresponding to one dataset. One must define the following
    keys in the ``dataset_params`` dictionary for each dataset name:

    - ``archive_name``: The name of the compressed file that is downloaded
    - ``url``: URL from which the file can be downloaded
    - ``folder_name``: the subfolder within the MNE data folder in which to
        save and uncompress (if needed) the file(s)
    - ``hash``: the cryptographic hash type of the file followed by a colon and
        then the hash value (examples: "sha256:19uheid...", "md5:upodh2io...")
    - ``config_key`` (optional): key to use with `mne.set_config` to store the
        on-disk location of the downloaded dataset (ex:
        "MNE_DATASETS_EEGBCI_PATH"). This is only used internally by MNE
        developers.

    An example would look like::

        {
            'dataset_name'='sample',
            'archive_name'='MNE-sample-data-processed.tar.gz',
            'hash'='md5:12b75d1cb7df9dfb4ad73ed82f61094f',
            'url'='https://osf.io/86qa2/download?version=5',
            'folder_name'='MNE-sample-data',
            'config_key'='MNE_DATASETS_SAMPLE_PATH',
        }

    Fetching datasets downloads files over HTTP/HTTPS. One can fetch private
    datasets by passing in authorization to the ``auth`` argument.
    """  # noqa E501
    # import pooch library for handling the dataset downloading
    pooch = _soft_import("pooch", "dataset downloading", strict=True)

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

    # additional condition: check for version.txt and parse it
    # check if testing or misc data is outdated; if so, redownload it
    want_version = RELEASES.get(name, None)
    want_version = _FAKE_VERSION if name == "fake" else want_version

    # get the version of the dataset and then check if the version is outdated
    data_version = _dataset_version(final_path, name)
    outdated = (want_version is not None and
                LooseVersion(want_version) > LooseVersion(data_version))

    if outdated:
        logger.info(
            f"Dataset {name} version {data_version} out of date, "
            f"latest version is {want_version}"
        )

    # return empty string if outdated dataset and we don't want to download
    if (not force_update) and outdated and not download:
        return ("", data_version) if return_version else ""

    # reasons to bail early (hf_sef has separate code for this):
    if (
        (not force_update)
        and (not outdated)
        and (not name.startswith("hf_sef_"))
    ):
        # if target folder exists (otherwise pooch downloads every time,
        # because we don't save the archive files after unpacking)
        if op.isdir(final_path):
            _do_path_update(path, update_path, config_key, name)
            return (final_path, data_version) if return_version else final_path
        # if download=False (useful for debugging)
        elif not download:
            return ("", data_version) if return_version else ""
        # if user didn't accept the license
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
    download_params = dict(progressbar=True)  # use tqdm
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
        path=final_path if processor is None else path,
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
        LooseVersion(data_version) < LooseVersion(mne_version.strip(".git"))
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
