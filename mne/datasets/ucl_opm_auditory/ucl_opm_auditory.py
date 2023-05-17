# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

from ...utils import verbose
from ..utils import _data_path_doc, _get_version, _version_doc, _download_mne_dataset


_NAME = "ucl_opm_auditory"
_PROCESSOR = "unzip"


@verbose
def data_path(
    path=None, force_update=False, update_path=True, download=True, *, verbose=None
):  # noqa: D103
    return _download_mne_dataset(
        name=_NAME,
        processor=_PROCESSOR,
        path=path,
        force_update=force_update,
        update_path=update_path,
        download=download,
    )


data_path.__doc__ = _data_path_doc.format(
    name=_NAME,
    conf=f"MNE_DATASETS_{_NAME.upper()}_PATH",
)


def get_version():  # noqa: D103
    return _get_version(_NAME)


get_version.__doc__ = _version_doc.format(name=_NAME)
