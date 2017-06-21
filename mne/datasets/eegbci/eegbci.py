# Author: Martin Billinger <martin.billinger@tugraz.at>
# License: BSD Style.

import os
from os import path as op

from ..utils import _get_path, _do_path_update
from ...utils import _fetch_file, _url_to_local_path, verbose


EEGMI_URL = 'http://www.physionet.org/physiobank/database/eegmmidb/'


@verbose
def data_path(url, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of EEGMMI dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote EEGBCI dataset [1]_ which is available at PhysioNet [2]_.

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

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
    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
           Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer
           Interface (BCI) System. IEEE TBME 51(6):1034-1043
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
           PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
           Research Resource for Complex Physiologic Signals.
           Circulation 101(23):e215-e220
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

    This will fetch data for the EEGBCI dataset [1]_, which is also
    available at PhysioNet [2]_.

    Parameters
    ----------
    subject : int
        The subject to use. Can be in the range of 1-109 (inclusive).
    runs : int | list of int
        The runs to use. The runs correspond to:

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import eegbci
        >>> eegbci.load_data(1, [4, 10, 14],\
                             os.getenv('HOME') + '/datasets') # doctest:+SKIP

    This would download runs 4, 10, and 14 (hand/foot motor imagery) runs from
    subject 1 in the EEGBCI dataset to the 'datasets' folder, and prompt the
    user to save the 'datasets' path to the  mne-python config, if it isn't
    there already.

    References
    ----------
    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
           Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer
           Interface (BCI) System. IEEE TBME 51(6):1034-1043
    .. [2] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh,
           Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000)
           PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
           Research Resource for Complex Physiologic Signals.
           Circulation 101(23):e215-e220
    """
    if not hasattr(runs, '__iter__'):
        runs = [runs]

    data_paths = []
    for r in runs:
        url = '{u}S{s:03d}/S{s:03d}R{r:02d}.edf'.format(u=base_url,
                                                        s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    return data_paths
