# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

from ...utils import _check_option, logger, verbose
from ..utils import (
    _data_path_doc,
    _download_mne_dataset,
    _downloader_params,
    _get_path,
    _get_version,
    _version_doc,
)

_N170_BASE_URL = "https://data.nemar.org/nm000132/v1.1.1/"
_N170_REGISTRY = {
    "sub-001/eeg/sub-001_task-N170_eeg.fdt": (
        "sha256:08406f3c6b4a869dc8f67c9acc233a91993bae1b04b7dee5bc0521677ed8949b"
    ),
    "sub-001/eeg/sub-001_task-N170_eeg.set": (
        "sha256:9c53dbdc3b469934a5eb6e9f01e59090dd47aeb495b8f21ceca03670991e5b11"
    ),
    "sub-001/eeg/sub-001_task-N170_events.tsv": (
        "sha256:07c87e728d097b0deb05b17d77bbdbd22ef58105111b0b56e659a767b9421e34"
    ),
}


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    return _download_mne_dataset(
        name="erp_core",
        processor="untar",
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


data_path.__doc__ = _data_path_doc.format(
    name="erp_core", conf="MNE_DATASETS_ERP_CORE_PATH"
)


def fetch_file(fname, path=None):
    """Fetch a raw ERP CORE file used by MNE-Python examples.

    Parameters
    ----------
    fname : str
        Relative path of the file within the ERP CORE dataset.
    path : path-like | None
        Parent directory for the ``ERP-CORE-N170`` folder. If ``None``, the
        ``MNE_DATASETS_ERP_CORE_PATH`` configuration value is used.

    Returns
    -------
    path : instance of Path
        Local path to the requested file.
    """
    import pooch

    _check_option("fname", fname, _N170_REGISTRY)
    path = _get_path(path, "MNE_DATASETS_ERP_CORE_PATH", "ERP CORE")
    fetcher = pooch.create(
        path=path / "ERP-CORE-N170",
        base_url=_N170_BASE_URL,
        registry=_N170_REGISTRY,
        retry_if_failed=2,
    )
    pooch.get_logger().setLevel(logger.getEffectiveLevel())
    downloader = pooch.HTTPDownloader(**_downloader_params())
    return Path(fetcher.fetch(fname, downloader=downloader, progressbar=True))


def get_version():  # noqa: D103
    return _get_version("erp_core")


get_version.__doc__ = _version_doc.format(name="erp_core")
