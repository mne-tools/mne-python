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

has_testing_data = partial(has_dataset, name="testing")


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    # Make sure we don't do something stupid
    if download and get_config("MNE_SKIP_TESTING_DATASET_TESTS", "false") == "true":
        raise RuntimeError("Cannot download data if skipping is forced")

    return _download_mne_dataset(
        name="testing",
        processor="untar",
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


data_path.__doc__ = _data_path_doc.format(
    name="testing", conf="MNE_DATASETS_TESTING_PATH"
)


def get_version():  # noqa: D103
    return _get_version("testing")


get_version.__doc__ = _version_doc.format(name="testing")


# Allow forcing of testing dataset skip (for Debian tests) using:
# `make test-no-testing-data`
def _skip_testing_data():
    skip_testing = get_config("MNE_SKIP_TESTING_DATASET_TESTS", "false") == "true"
    skip = skip_testing or not has_testing_data()
    return skip


def requires_testing_data(func):
    """Skip testing data test."""
    return _pytest_mark()(func)


def _pytest_param(*args, **kwargs):
    if len(args) == 0:
        args = ("testing_data",)
    import pytest

    # turn anything that uses testing data into an auto-skipper by
    # setting params=[testing._pytest_param()], or by parametrizing functions
    # with testing._pytest_param(whatever)
    kwargs["marks"] = kwargs.get("marks", list()) + [_pytest_mark()]
    return pytest.param(*args, **kwargs)


def _pytest_mark():
    import pytest

    return pytest.mark.skipif(_skip_testing_data(), reason="Requires testing dataset")
