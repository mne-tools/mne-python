# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ...utils import verbose
from ..utils import _data_path_doc, _download_mne_dataset, _get_version, _version_doc


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    return _download_mne_dataset(
        name="eyelink",
        processor="unzip",
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


data_path.__doc__ = _data_path_doc.format(
    name="eyelink", conf="MNE_DATASETS_EYELINK_PATH"
)


def get_version():  # noqa: D103
    return _get_version("eyelink")


get_version.__doc__ = _version_doc.format(name="eyelink")
