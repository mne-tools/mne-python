# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Curated data subset once used by the JupyterLite browser documentation.

The documentation build now serves the browser notebooks from the regular
datasets, so this archive is no longer needed.
"""

from ...utils import deprecated, verbose
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
_DEPRECATED = (
    "The documentation build no longer uses the lite_data archive, so it will be "
    "removed in MNE 1.15; use the individual dataset fetchers instead"
)
data_path = deprecated(_DEPRECATED)(data_path)


def get_version():  # noqa: D103
    return _get_version("lite_data")


get_version.__doc__ = _version_doc.format(name="lite_data")
get_version = deprecated(_DEPRECATED)(get_version)
