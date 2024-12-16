# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import re
import time
from importlib.resources import files
from os import path as op
from pathlib import Path

from ...utils import _url_to_local_path, logger, verbose, warn
from ..utils import _do_path_update, _downloader_params, _get_path, _log_time_size

EEGMI_URL = "https://physionet.org/files/eegmmidb/1.0.0/"


@verbose
def data_path(url, path=None, force_update=False, update_path=None, *, verbose=None):
    """Get path to local copy of EEGMMI dataset URL.

    This is a low-level function useful for getting a local copy of a remote EEGBCI
    dataset :footcite:`SchalkEtAl2004`, which is also available at PhysioNet
    :footcite:`GoldbergerEtAl2000`. Metadata, such as the meaning of event markers
    may be obtained from the
    `PhysioNet documentation page <https://physionet.org/content/eegmmidb/1.0.0/>`_.

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | path-like
        Location of where to look for the EEGBCI data. If ``None``, the environment
        variable or config parameter ``MNE_DATASETS_EEGBCI_PATH`` is used. If neither
        exists, the ``~/mne_data`` directory is used. If the EEGBCI dataset is not found
        under the given path, the data will be automatically downloaded to the specified
        folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If ``True``, set ``MNE_DATASETS_EEGBCI_PATH`` in the configuration to the given
        path. If ``None``, the user is prompted.
    %(verbose)s

    Returns
    -------
    path : list of Path
        Local path to the given data file. This path is contained inside a list of
        length one for compatibility.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import eegbci
        >>> url = "http://www.physionet.org/physiobank/database/eegmmidb/"
        >>> eegbci.data_path(url, "~/datasets") # doctest:+SKIP

    This would download the given EEGBCI data file to the ``~/datasets`` folder and
    prompt the user to store this path in the config (if it does not already exist).

    References
    ----------
    .. footbibliography::
    """
    import pooch

    key = "MNE_DATASETS_EEGBCI_PATH"
    name = "EEGBCI"
    path = _get_path(path, key, name)
    fname = "MNE-eegbci-data"
    destination = _url_to_local_path(url, op.join(path, fname))
    destinations = [destination]

    # fetch the file
    downloader = pooch.HTTPDownloader(**_downloader_params())
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        pooch.retrieve(
            url=url,
            path=destination,
            downloader=downloader,
            fname=fname,
        )

    # offer to update the path
    _do_path_update(path, update_path, key, name)
    destinations = [Path(dest) for dest in destinations]
    return destinations


@verbose
def load_data(
    subjects=None,
    runs=None,
    *,
    subject=None,
    path=None,
    force_update=False,
    update_path=None,
    base_url=EEGMI_URL,
    verbose=None,
):  # noqa: D301
    """Get paths to local copies of EEGBCI dataset files.

    This will fetch data for the EEGBCI dataset :footcite:`SchalkEtAl2004`, which is
    also available at PhysioNet :footcite:`GoldbergerEtAl2000`. Metadata, such as the
    meaning of event markers may be obtained from the
    `PhysioNet documentation page <https://physionet.org/content/eegmmidb/1.0.0/>`_.

    Parameters
    ----------
    subjects : int | list of int
        The subjects to use. Can be in the range of 1-109 (inclusive).
    runs : int | list of int
        The runs to use (see Notes for details).
    subject : int
        This parameter is deprecated and will be removed in mne version 1.9.
        Please use ``subjects`` instead.
    path : None | path-like
        Location of where to look for the EEGBCI data. If ``None``, the environment
        variable or config parameter ``MNE_DATASETS_EEGBCI_PATH`` is used. If neither
        exists, the ``~/mne_data`` directory is used. If the EEGBCI dataset is not found
        under the given path, the data will be automatically downloaded to the specified
        folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If ``True``, set ``MNE_DATASETS_EEGBCI_PATH`` in the configuration to the given
        path. If ``None``, the user is prompted.
    base_url : str
        The URL root for the data.
    %(verbose)s

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    Notes
    -----
    The run numbers correspond to:

    =========  ===================================
    run        task
    =========  ===================================
    1          Baseline, eyes open
    2          Baseline, eyes closed
    3, 7, 11   Motor execution: left vs right hand
    4, 8, 12   Motor imagery: left vs right hand
    5, 9, 13   Motor execution: hands vs feet
    6, 10, 14  Motor imagery: hands vs feet
    =========  ===================================

    For example, one could do::

        >>> from mne.datasets import eegbci
        >>> eegbci.load_data([1, 2], [6, 10, 14], "~/datasets") # doctest:+SKIP

    This would download runs 6, 10, and 14 (hand/foot motor imagery) runs from subjects
    1 and 2 in the EEGBCI dataset to "~/datasets" and prompt the user to store this path
    in the config (if it does not already exist).

    References
    ----------
    .. footbibliography::
    """
    import pooch

    # XXX: Remove this with mne 1.9 ↓↓↓
    # Also remove the subject parameter at that point.
    # Also remove the `None` default for subjects and runs params at that point.
    if subject is not None:
        subjects = subject
        warn(
            "The ``subject`` parameter is deprecated and will be removed in version "
            "1.9. Use the ``subjects`` parameter (note the `s`) to suppress this "
            "warning.",
            FutureWarning,
        )
        del subject
    if subjects is None or runs is None:
        raise ValueError("You must pass the parameters ``subjects`` and ``runs``.")
    # ↑↑↑

    t0 = time.time()

    if not hasattr(subjects, "__iter__"):
        subjects = [subjects]

    if not hasattr(runs, "__iter__"):
        runs = [runs]

    # get local storage path
    config_key = "MNE_DATASETS_EEGBCI_PATH"
    folder = "MNE-eegbci-data"
    name = "EEGBCI"
    path = _get_path(path, config_key, name)

    # extract path parts
    pattern = r"(?:https?://.*)(files)/(eegmmidb)/(\d+\.\d+\.\d+)/?"
    match = re.compile(pattern).match(base_url)
    if match is None:
        raise ValueError(
            "base_url does not match the expected EEGMI folder "
            "structure. Please notify MNE-Python developers."
        )
    base_path = op.join(path, folder, *match.groups())

    # create the download manager
    fetcher = pooch.create(
        path=base_path,
        base_url=base_url,
        version=None,  # data versioning is decoupled from MNE-Python version
        registry=None,  # registry is loaded from file (below)
        retry_if_failed=2,  # 2 retries = 3 total attempts
    )

    # load the checksum registry
    registry = files("mne").joinpath("data", "eegbci_checksums.txt")
    fetcher.load_registry(registry)

    # fetch the file(s)
    data_paths = []
    sz = 0
    for subject in subjects:
        for run in runs:
            file_part = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            destination = Path(base_path, file_part)
            data_paths.append(destination)
            if destination.exists():
                if force_update:
                    destination.unlink()
                else:
                    continue
            if sz == 0:  # log once
                logger.info("Downloading EEGBCI data")
            fetcher.fetch(file_part)
            # update path in config if desired
            sz += destination.stat().st_size

    _do_path_update(path, update_path, config_key, name)
    if sz > 0:
        _log_time_size(t0, sz)
    return data_paths


def standardize(raw):
    """Standardize channel positions and names.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to standardize. Operates in-place.
    """
    rename = dict()
    for name in raw.ch_names:
        std_name = name.strip(".")
        std_name = std_name.upper()
        if std_name.endswith("Z"):
            std_name = std_name[:-1] + "z"
        if std_name.startswith("FP"):
            std_name = "Fp" + std_name[2:]
        rename[name] = std_name
    raw.rename_channels(rename)
