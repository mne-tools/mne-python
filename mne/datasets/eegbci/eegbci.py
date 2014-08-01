# Author: Martin Billinger <martin.billinger@tugraz.at>
# License: BSD Style.

import os
from os import path as op
from ...externals.six import string_types
from ...utils import _fetch_file, get_config, set_config, _url_to_local_path

if 'raw_input' not in __builtins__:
    raw_input = input


EEGMI_URL = 'http://www.physionet.org/physiobank/database/eegmmidb/'


def data_path(url, path=None, force_update=False, update_path=None):
    """Get path to local copy of EEGMMI dataset URL

    This is a low-level function useful for getting a local copy of a
    remote EEGBCI dataet.

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the EEGBCI data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_EEGBCI_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the EEGBCI dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MNE-eegbci-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_EEGBCI_PATH in mne-python
        config to the given path. If None, the user is prompted.

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

    The EEGBCI dataset is documented in the following publication:
        Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
        Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
        (BCI) System. IEEE TBME 51(6):1034-1043
    The data set is available at PhysioNet:
        Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
        Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
        PhysioToolkit, and PhysioNet: Components of a New Research Resource for
        Complex Physiologic Signals. Circulation 101(23):e215-e220
    """

    if path is None:
        # use an intelligent guess if it's not defined
        def_path = op.realpath(op.join(op.dirname(__file__), '..', '..',
                                       '..', 'examples'))

        key = 'MNE_DATASETS_EEGBCI_PATH'
        # backward compatibility
        if get_config(key) is None:
            key = 'MNE_DATA'

        path = get_config(key, def_path)

        # use the same for all datasets
        if not op.exists(path) or not os.access(path, os.W_OK):
            try:
                os.mkdir(path)
            except OSError:
                try:
                    logger.info("Checking for EEGBCI data in '~/mne_data'...")
                    path = op.join(op.expanduser("~"), "mne_data")
                    if not op.exists(path):
                        logger.info("Trying to create "
                                    "'~/mne_data' in home directory")
                        os.mkdir(path)
                except OSError:
                    raise OSError("User does not have write permissions "
                                  "at '%s', try giving the path as an argument "
                                  "to data_path() where user has write "
                                  "permissions, for ex:data_path"
                                  "('/home/xyz/me2/')" % (path))

    if not isinstance(path, string_types):
        raise ValueError('path must be a string or None')

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
    path = op.abspath(path)
    if update_path is None:
        if get_config(key, '') != path:
            update_path = True
            msg = ('Do you want to set the path:\n    %s\nas the default '
                   'EEGBCI dataset path in the mne-python config ([y]/n)? '
                   % path)
            answer = raw_input(msg)
            if answer.lower() == 'n':
                update_path = False
        else:
            update_path = False
    if update_path is True:
        set_config(key, path)

    return destinations


def load_data(subject, runs, path=None, force_update=False, update_path=None,
              base_url=EEGMI_URL):
    """Get paths to local copy of EEGBCI dataset files

    Parameters
    ----------
    subject : int
        The subject to use. Can be in the range of 1-109 (inclusive).
    runs : int | list of ints
        The runs to use. Can be a list or a single number. The runs correspond
        to the following tasks:
              run | task
        ----------+-----------------------------------------
                1 | Baseline, eyes open
                2 | Baseline, eyes closed
         3, 7, 11 | Motor execution: left vs right hand
         4, 8, 12 | Motor imagery: left vs right hand
         5, 9, 13 | Motor execution: hands vs feet
        6, 10, 14 | Motor imagery: hands vs feet
    path : None | str
        Location of where to look for the EEGBCI data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_EEGBCI_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the EEGBCI dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MEGSIM"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_EEGBCI_PATH in mne-python
        config to the given path. If None, the user is prompted.

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

    The EEGBCI dataset is documented in the following publication:
        Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
        Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
        (BCI) System. IEEE TBME 51(6):1034-1043
    The data set is available at PhysioNet:
        Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
        Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
        PhysioToolkit, and PhysioNet: Components of a New Research Resource for
        Complex Physiologic Signals. Circulation 101(23):e215-e220
    """
    if not hasattr(runs, '__iter__'):
        runs = [runs]

    data_paths = []
    for r in runs:
        url = '{u}S{s:03d}/S{s:03d}R{r:02d}.edf'.format(u=base_url,
                                                        s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    return data_paths
