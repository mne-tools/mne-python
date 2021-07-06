# Author: Martin Billinger <martin.billinger@tugraz.at>
# License: BSD Style.

import os
from os import path as op

from ..utils import _get_path, _do_path_update
from ...utils import _fetch_file, _url_to_local_path, verbose


EEGMI_URL = 'https://physionet.org/files/eegmmidb/1.0.0/'


@verbose
def data_path(url, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of EEGMMI dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote EEGBCI dataset :footcite:`SchalkEtAl2004` which is available at PhysioNet :footcite:`GoldbergerEtAl2000`.

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the EEGBCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_EEGBCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the EEGBCI dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_EEGBCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    %(verbose)s

    Returns
    -------
    path : list of str
        Local path to the given data file. This path is contained inside a list
        of length one, for compatibility.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import eegbci
        >>> url = 'http://www.physionet.org/physiobank/database/eegmmidb/'
        >>> eegbci.data_path(url, os.getenv('HOME') + '/datasets') # doctest:+SKIP

    This would download the given EEGBCI data file to the 'datasets' folder,
    and prompt the user to save the 'datasets' path to the mne-python config,
    if it isn't there already.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    key = 'MNE_DATASETS_EEGBCI_PATH'
    name = 'EEGBCI'
    path = _get_path(path, key, name)
    destination = _url_to_local_path(url, op.join(path, 'MNE-eegbci-data'))
    destinations = [destination]

    # Fetch the file
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)

    # Offer to update the path
    _do_path_update(path, update_path, key, name)
    return destinations


@verbose
def load_data(subject, runs, path=None, force_update=False, update_path=None,
              base_url=EEGMI_URL, verbose=None):  # noqa: D301
    """Get paths to local copies of EEGBCI dataset files.

    This will fetch data for the EEGBCI dataset :footcite:`SchalkEtAl2004`, which is also
    available at PhysioNet :footcite:`GoldbergerEtAl2000`.

    Parameters
    ----------
    subject : int
        The subject to use. Can be in the range of 1-109 (inclusive).
    runs : int | list of int
        The runs to use. See Notes for details.
    path : None | str
        Location of where to look for the EEGBCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_EEGBCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the EEGBCI dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_EEGBCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    base_url : str
        The URL root for the data.
    %(verbose)s

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    Notes
    -----
    The run numbers correspond to:

    =========  ===================================
    run        task
    =========  ===================================
    1          Baseline, eyes open
    2          Baseline, eyes closed
    3, 7, 11   Motor execution: left vs right hand
    4, 8, 12   Motor imagery: left vs right hand
    5, 9, 13   Motor execution: hands vs feet
    6, 10, 14  Motor imagery: hands vs feet
    =========  ===================================

    For example, one could do::

        >>> from mne.datasets import eegbci
        >>> eegbci.load_data(1, [4, 10, 14], os.getenv('HOME') + '/datasets') # doctest:+SKIP

    This would download runs 4, 10, and 14 (hand/foot motor imagery) runs from
    subject 1 in the EEGBCI dataset to the 'datasets' folder, and prompt the
    user to save the 'datasets' path to the  mne-python config, if it isn't
    there already.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    if not hasattr(runs, '__iter__'):
        runs = [runs]

    data_paths = []
    for r in runs:
        url = '{u}S{s:03d}/S{s:03d}R{r:02d}.edf'.format(u=base_url,
                                                        s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    return data_paths


def standardize(raw):
    """Standardize channel positions and names.

    Parameters
    ----------
    raw : instance of Raw
        The raw data to standardize. Operates in-place.
    """
    rename = dict()
    for name in raw.ch_names:
        std_name = name.strip('.')
        std_name = std_name.upper()
        if std_name.endswith('Z'):
            std_name = std_name[:-1] + 'z'
        if std_name.startswith('FP'):
            std_name = 'Fp' + std_name[2:]
        rename[name] = std_name
    raw.rename_channels(rename)
