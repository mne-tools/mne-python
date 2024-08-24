# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import os.path as op

import numpy as np

from ...utils import _check_pandas_installed, _on_missing, _TempDir, verbose
from ..utils import _downloader_params, _get_path

AGE_SLEEP_RECORDS = op.join(op.dirname(__file__), "age_records.csv")
TEMAZEPAM_SLEEP_RECORDS = op.join(op.dirname(__file__), "temazepam_records.csv")

TEMAZEPAM_RECORDS_URL = (
    "https://physionet.org/physiobank/database/sleep-edfx/ST-subjects.xls"  # noqa: E501
)
TEMAZEPAM_RECORDS_URL_SHA1 = "f52fffe5c18826a2bd4c5d5cb375bb4a9008c885"

AGE_RECORDS_URL = "https://physionet.org/physiobank/database/sleep-edfx/SC-subjects.xls"
AGE_RECORDS_URL_SHA1 = "0ba6650892c5d33a8e2b3f62ce1cc9f30438c54f"

sha1sums_fname = op.join(op.dirname(__file__), "SHA1SUMS")


def _fetch_one(fname, hashsum, path, force_update, base_url):
    import pooch

    # Fetch the file
    url = base_url + "/" + fname
    destination = op.join(path, fname)
    if op.isfile(destination) and not force_update:
        return destination, False
    if op.isfile(destination):
        os.remove(destination)
    if not op.isdir(op.dirname(destination)):
        os.makedirs(op.dirname(destination))
    downloader = pooch.HTTPDownloader(**_downloader_params())
    pooch.retrieve(
        url=url,
        known_hash=f"sha1:{hashsum}",
        path=path,
        downloader=downloader,
        fname=fname,
    )
    return destination, True


@verbose
def _data_path(path=None, verbose=None):
    """Get path to local copy of EEG Physionet age Polysomnography dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote Polysomnography dataset :footcite:`KempEtAl2000` which is available
    at PhysioNet :footcite:`GoldbergerEtAl2000`.

    Parameters
    ----------
    path : None | str
        Location of where to look for the data storing location.
        If None, the environment variable or config parameter
        ``PHYSIONET_SLEEP_PATH`` is used. If it doesn't exist, the "~/mne_data"
        directory is used. If the dataset is not found under the given path,
        the data will be automatically downloaded to the specified folder.
    %(verbose)s

    Returns
    -------
    path : list of Path
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    key = "PHYSIONET_SLEEP_PATH"
    name = "PHYSIONET_SLEEP"
    path = _get_path(path, key, name)
    return op.join(path, "physionet-sleep-data")


def _update_sleep_temazepam_records(fname=TEMAZEPAM_SLEEP_RECORDS):
    """Help function to download Physionet's temazepam dataset records."""
    import pooch

    pd = _check_pandas_installed()
    tmp = _TempDir()

    # Download subjects info.
    subjects_fname = op.join(tmp, "ST-subjects.xls")
    downloader = pooch.HTTPDownloader(**_downloader_params())
    pooch.retrieve(
        url=TEMAZEPAM_RECORDS_URL,
        known_hash=f"sha1:{TEMAZEPAM_RECORDS_URL_SHA1}",
        path=tmp,
        downloader=downloader,
        fname=op.basename(subjects_fname),
    )

    # Load and Massage the checksums.
    sha1_df = pd.read_csv(
        sha1sums_fname, sep="  ", header=None, names=["sha", "fname"], engine="python"
    )
    select_age_records = sha1_df.fname.str.startswith(
        "ST"
    ) & sha1_df.fname.str.endswith("edf")
    sha1_df = sha1_df[select_age_records]
    sha1_df["id"] = [name[:6] for name in sha1_df.fname]

    # Load and massage the data.
    data = pd.read_excel(subjects_fname, header=[0, 1])
    data = data.set_index(("Subject - age - sex", "Nr"))
    data.index.name = "subject"
    data.columns.names = [None, None]
    data = (
        data.set_index(
            [("Subject - age - sex", "Age"), ("Subject - age - sex", "M1/F2")],
            append=True,
        )
        .stack(level=0)
        .reset_index()
    )

    data = data.rename(
        columns={
            ("Subject - age - sex", "Age"): "age",
            ("Subject - age - sex", "M1/F2"): "sex",
            "level_3": "drug",
        }
    )
    data["id"] = [f"ST7{s:02d}{n:1d}" for s, n in zip(data.subject, data["night nr"])]

    data = pd.merge(sha1_df, data, how="outer", on="id")
    data["record type"] = (
        data.fname.str.split("-", expand=True)[1]
        .str.split(".", expand=True)[0]
        .astype("category")
    )

    data = data.set_index(
        ["id", "subject", "age", "sex", "drug", "lights off", "night nr", "record type"]
    ).unstack()
    data.columns = [l1 + "_" + l2 for l1, l2 in data.columns]
    data = data.reset_index().drop(columns=["id"])

    data["sex"] = data.sex.astype("category").cat.rename_categories(
        {1: "male", 2: "female"}
    )

    data["drug"] = data["drug"].str.split(expand=True)[0]
    data["subject_orig"] = data["subject"]
    data["subject"] = data.index // 2  # to make sure index is from 0 to 21

    # Save the data.
    data.to_csv(fname, index=False)


def _update_sleep_age_records(fname=AGE_SLEEP_RECORDS):
    """Help function to download Physionet's age dataset records."""
    import pooch

    pd = _check_pandas_installed()
    tmp = _TempDir()

    # Download subjects info.
    subjects_fname = op.join(tmp, "SC-subjects.xls")
    downloader = pooch.HTTPDownloader(**_downloader_params())
    pooch.retrieve(
        url=AGE_RECORDS_URL,
        known_hash=f"sha1:{AGE_RECORDS_URL_SHA1}",
        path=tmp,
        downloader=downloader,
        fname=op.basename(subjects_fname),
    )

    # Load and Massage the checksums.
    sha1_df = pd.read_csv(
        sha1sums_fname, sep="  ", header=None, names=["sha", "fname"], engine="python"
    )
    select_age_records = sha1_df.fname.str.startswith(
        "SC"
    ) & sha1_df.fname.str.endswith("edf")
    sha1_df = sha1_df[select_age_records]
    sha1_df["id"] = [name[:6] for name in sha1_df.fname]

    # Load and massage the data.
    data = pd.read_excel(subjects_fname)
    data = data.rename(
        index=str, columns={"sex (F=1)": "sex", "LightsOff": "lights off"}
    )
    data["sex"] = data.sex.astype("category").cat.rename_categories(
        {1: "female", 2: "male"}
    )

    data["id"] = [f"SC4{s:02d}{n:1d}" for s, n in zip(data.subject, data.night)]

    data = data.set_index("id").join(sha1_df.set_index("id")).dropna()

    data["record type"] = (
        data.fname.str.split("-", expand=True)[1]
        .str.split(".", expand=True)[0]
        .astype("category")
    )

    data = data.reset_index().drop(columns=["id"])
    data = data[
        ["subject", "night", "record type", "age", "sex", "lights off", "sha", "fname"]
    ]

    # Save the data.
    data.to_csv(fname, index=False)


def _check_subjects(subjects, n_subjects, missing=None, on_missing="raise"):
    """Check whether subjects are available.

    Parameters
    ----------
    subjects : list
        Subject numbers to be checked.
    n_subjects : int
        Number of subjects available.
    missing : list | None
        Subject numbers that are missing.
    on_missing : 'raise' | 'warn' | 'ignore'
        What to do if one or several subjects are not available. Valid keys
        are 'raise' | 'warn' | 'ignore'. Default is 'error'. If on_missing
        is 'warn' it will proceed but warn, if 'ignore' it will proceed
        silently.
    """
    valid_subjects = np.arange(n_subjects)
    if missing is not None:
        valid_subjects = np.setdiff1d(valid_subjects, missing)
    unknown_subjects = np.setdiff1d(subjects, valid_subjects)
    if unknown_subjects.size > 0:
        subjects_list = ", ".join([str(s) for s in unknown_subjects])
        msg = (
            f"This dataset contains subjects 0 to {n_subjects - 1} with "
            f"missing subjects {missing}. Unknown subjects: "
            f"{subjects_list}."
        )
        _on_missing(on_missing, msg)
