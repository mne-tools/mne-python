# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import time

import numpy as np

from ...utils import verbose
from ..utils import _log_time_size
from ._utils import (
    AGE_SLEEP_RECORDS,
    _check_subjects,
    _data_path,
    _fetch_one,
    _on_missing,
)

data_path = _data_path  # expose _data_path(..) as data_path(..)

BASE_URL = "https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/"


@verbose
def fetch_data(
    subjects,
    recording=(1, 2),
    path=None,
    force_update=False,
    base_url=BASE_URL,
    on_missing="raise",
    *,
    verbose=None,
):  # noqa: D301, E501
    """Get paths to local copies of PhysioNet Polysomnography dataset files.

    This will fetch data from the publicly available subjects from PhysioNet's
    study of age effects on sleep in healthy subjects
    :footcite:`MourtazaevEtAl1995,GoldbergerEtAl2000`. This
    corresponds to a subset of 153 recordings from 37 males and 41 females that
    were 25-101 years old at the time of the recordings. There are two night
    recordings per subject except for subjects 13, 36 and 52 which have one
    record missing each due to missing recording hardware.

    See more details in
    `physionet website <https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/>`_.

    Parameters
    ----------
    subjects : list of int
        The subjects to use. Can be in the range of 0-82 (inclusive), however
        the following subjects are not available: 39, 68, 69, 78 and 79.
    recording : list of int
        The night recording indices. Valid values are : [1], [2], or [1, 2].
        The following recordings are not available: recording 1 for subject 36
        and 52, and recording 2 for subject 13.
    path : None | str
        Location of where to look for the PhysioNet data storing location.
        If None, the environment variable or config parameter
        ``PHYSIONET_SLEEP_PATH`` is used. If it doesn't exist, the "~/mne_data"
        directory is used. If the Polysomnography dataset is not found under
        the given path, the data will be automatically downloaded to the
        specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    base_url : str
        The URL root.
    on_missing : 'raise' | 'warn' | 'ignore'
        What to do if one or several recordings are not available. Valid keys
        are 'raise' | 'warn' | 'ignore'. Default is 'error'. If on_missing
        is 'warn' it will proceed but warn, if 'ignore' it will proceed
        silently.
    %(verbose)s

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    See Also
    --------
    mne.datasets.sleep_physionet.temazepam.fetch_data

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import sleep_physionet
        >>> sleep_physionet.age.fetch_data(subjects=[0])  # doctest: +SKIP

    This would download data for subject 0 if it isn't there already.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    t0 = time.time()
    records = np.loadtxt(
        AGE_SLEEP_RECORDS,
        skiprows=1,
        delimiter=",",
        usecols=(0, 1, 2, 6, 7),
        dtype={
            "names": ("subject", "record", "type", "sha", "fname"),
            "formats": ("<i2", "i1", "<S9", "S40", "<S22"),
        },
    )
    psg_records = records[np.where(records["type"] == b"PSG")]
    hyp_records = records[np.where(records["type"] == b"Hypnogram")]

    path = data_path(path=path)
    params = [path, force_update, base_url]

    _check_subjects(subjects, 83, missing=[39, 68, 69, 78, 79], on_missing=on_missing)

    # Check for missing recordings
    if set(subjects) & {36, 52} and 1 in recording:
        msg = (
            "Requested recording 1 for subject 36 and/or 52, but it is not "
            "available in corpus."
        )
        _on_missing(on_missing, msg)
    if 13 in subjects and 2 in recording:
        msg = "Requested recording 2 for subject 13, but it is not available in corpus."
        _on_missing(on_missing, msg)

    fnames = []
    sz = 0
    for subject in subjects:
        for idx in np.where(psg_records["subject"] == subject)[0]:
            if psg_records["record"][idx] in recording:
                psg_fname, pdl = _fetch_one(
                    psg_records["fname"][idx].decode(),
                    psg_records["sha"][idx].decode(),
                    *params,
                )
                hyp_fname, hdl = _fetch_one(
                    hyp_records["fname"][idx].decode(),
                    hyp_records["sha"][idx].decode(),
                    *params,
                )
                fnames.append([psg_fname, hyp_fname])
                if pdl:
                    sz += os.path.getsize(psg_fname)
                if hdl:
                    sz += os.path.getsize(hyp_fname)
    if sz > 0:
        _log_time_size(t0, sz)
    return fnames
