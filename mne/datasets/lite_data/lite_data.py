# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Curated data subset used by the JupyterLite browser documentation.

``lite_data`` holds the data files needed to run the tutorials and examples in
the browser, taken from ``sample``, ``kiloword``, ``erp_core``, ``mtrf`` and
``eegbci``. The files are unchanged and keep the same checksums as the full
datasets. It extracts to ``MNE-lite-data/`` with each file under its original
dataset folder (``MNE-sample-data/``, ``MNE-kiloword-data/``, ...), so paths
match.
The ``somato`` dataset is not included, so the somatosensory tutorials and
examples do not run in the browser.
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
