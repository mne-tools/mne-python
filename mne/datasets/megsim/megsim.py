# Author: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
from os import path as op

import logging
logger = logging.getLogger('mne')

from ...utils import _download_status, get_config, set_config, \
                     _url_to_local_path


def data_path(url, path=None, force_update=False, update_path=None):
    """Get path to local copy of MEGSIM dataset URL

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the MEGSIM data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_MEGSIM_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the sample dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MEGSIM"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_MEGSIM_PATH in mne-python
        config to the given path. If None, the user is prompted.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import megsim
        >>> url = 'http://cobre.mrn.org/megsim/simdata/neuromag/visual/M87174545_vis_sim1A_4mm_30na_neuro_rn.fif'
        >>> megsim.data_path(url, os.getenv('HOME') + '/datasets') # doctest:+SKIP

    And this would download the given MEGSIM data file to the 'datasets'
    folder, and prompt the user to save the 'datasets' path to the mne-python
    config, if it weren't there already.
    """
    if path is None:
        # use an intelligent guess if it's not defined
        def_path = op.abspath(op.join(op.dirname(__file__), '..', '..',
                                      '..', 'examples'))
        path = get_config('MNE_DATASETS_SAMPLE_PATH', def_path)

    if not isinstance(path, basestring):
        raise ValueError('path must be a string or None')

    destination = _url_to_local_path(url, path)

    # Fetch the file
    if not op.isfile(destination) or force_update:
        logger.info('Downloading data, please wait:')
        logger.info(url)
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _download_status(url, destination, False)

    # Offer to update the path
    path = op.abspath(path)
    if update_path is None:
        if  get_config('MNE_DATASETS_MEGSIM_PATH', '') != path:
            update_path = True
            msg = ('Do you want to set the path:\n    %s\nas the default '
                   'MEGSIM dataset path in the mne-python config [y]/n:'
                   % path)
            answer = raw_input(msg)
            if answer.lower() == 'n':
                update_path = False
        else:
            update_path = False
    if update_path is True:
        set_config('MNE_DATASETS_MEGSIM_PATH', path)

    return destination
