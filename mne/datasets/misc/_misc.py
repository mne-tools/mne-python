# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ...utils import verbose
from ..utils import _data_path_doc, _download_mne_dataset, has_dataset


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    return _download_mne_dataset(
        name="misc",
        processor="untar",
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


def _pytest_mark():
    import pytest

    return pytest.mark.skipif(
        not has_dataset(name="misc"), reason="Requires misc dataset"
    )


data_path.__doc__ = _data_path_doc.format(name="misc", conf="MNE_DATASETS_MISC_PATH")
