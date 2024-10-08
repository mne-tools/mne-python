# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial

from ...utils import get_config, verbose
from ..utils import (
    _data_path_doc,
    _download_mne_dataset,
    _get_version,
    _version_doc,
    has_dataset,
)

has_spm_data = partial(has_dataset, name="spm")


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    return _download_mne_dataset(
        name="spm",
        processor="untar",
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


data_path.__doc__ = _data_path_doc.format(name="spm", conf="MNE_DATASETS_SPM_DATA_PATH")


def get_version():  # noqa: D103
    return _get_version("spm")


get_version.__doc__ = _version_doc.format(name="spm")


def _skip_spm_data():
    skip_testing = get_config("MNE_SKIP_TESTING_DATASET_TESTS", "false") == "true"
    skip = skip_testing or not has_spm_data()
    return skip


def requires_spm_data(func):
    """Skip testing data test."""
    import pytest

    return pytest.mark.skipif(_skip_spm_data(), reason="Requires spm dataset")(func)
