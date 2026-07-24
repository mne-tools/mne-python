# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Curated data subset used by the JupyterLite browser documentation.

The full MNE datasets (``sample``, ``kiloword``, ``erp_core``, ``mtrf``,
``eegbci``) ship as separate multi-GB archives, so the docs build would download
several gigabytes just to serve a handful of files to the browser notebooks.
``lite_data`` is a small curated archive holding only those files -- same data,
same checksums -- so the build fetches just what the JupyterLite notebooks need.
It extracts to ``MNE-lite-data/`` with the files under their original dataset
folders (``MNE-sample-data/``, ``MNE-kiloword-data/``, ...).
"""

from ...utils import verbose
from ..utils import _data_path_doc, _download_mne_dataset, _get_version, _version_doc


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    return _download_mne_dataset(
        name="lite_data",
        processor="untar",
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


data_path.__doc__ = _data_path_doc.format(
    name="lite_data", conf="MNE_DATASETS_LITE_DATA_PATH"
)


def get_version():  # noqa: D103
    return _get_version("lite_data")


get_version.__doc__ = _version_doc.format(name="lite_data")
