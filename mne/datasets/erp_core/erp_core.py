from ...utils import verbose
from ..utils import (_data_path_doc,
                     _get_version, _version_doc, _download_mne_dataset)


@verbose
def data_path(path=None, force_update=False, update_path=True,
              download=True, *, verbose=None):  # noqa: D103
    return _download_mne_dataset(
        name='erp_core', processor='untar', path=path,
        force_update=force_update, update_path=update_path,
        download=download)


data_path.__doc__ = _data_path_doc.format(name='erp_core',
                                          conf='MNE_DATASETS_ERP_CORE_PATH')


def get_version():  # noqa: D103
    return _get_version('erp_core')


get_version.__doc__ = _version_doc.format(name='erp_core')
