# Author: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
from os import path as op
import zipfile
from sys import stdout

import logging
logger = logging.getLogger('mne')

from ...utils import _download_status, get_config, set_config, \
                     _url_to_local_path
from .urls import url_match, valid_data_types, valid_data_formats, \
                  valid_conditions


def data_path(url, path=None, force_update=False, update_path=None):
    """Get path to local copy of MEGSIM dataset URL

    This is a low-level function useful for getting a local copy of a
    remote MEGSIM dataet.

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the MEGSIM data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_MEGSIM_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the MEGSIM dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MEGSIM"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_MEGSIM_PATH in mne-python
        config to the given path. If None, the user is prompted.

    Returns
    -------
    path : list of str
        Local paths to the given data files. If URL was a .fif file, this
        will be a list of length 1. If it was a .zip file, it may potentially
        be many files.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import megsim
        >>> url = 'http://cobre.mrn.org/megsim/simdata/neuromag/visual/M87174545_vis_sim1A_4mm_30na_neuro_rn.fif'
        >>> megsim.data_path(url, os.getenv('HOME') + '/datasets') # doctest:+SKIP

    And this would download the given MEGSIM data file to the 'datasets'
    folder, and prompt the user to save the 'datasets' path to the mne-python
    config, if it isn't there already.

    The MEGSIM dataset is documented in the following publication:
        Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T,
        Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM
        (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using
        Realistic Simulated and Empirical Data. Neuroinform 10:141-158
    """

    if path is None:
        # use an intelligent guess if it's not defined
        def_path = op.abspath(op.join(op.dirname(__file__), '..', '..',
                                      '..', 'examples'))
        path = get_config('MNE_DATASETS_MEGSIM_PATH', None)
        if path is None:
            path = def_path
            msg = ('No path entered, defaulting to download MEGSIM data to:\n'
                   '    %s\nDo you want to continue ([y]/n)? '
                   % path)
            answer = raw_input(msg)
            if answer.lower() == 'n':
                raise ValueError('Please enter preferred path as '
                                 'megsim.data_path(url, path)')

    if not isinstance(path, basestring):
        raise ValueError('path must be a string or None')

    destination = _url_to_local_path(url, op.join(path, 'MEGSIM'))
    destinations = [destination]

    split = op.splitext(destination)
    is_zip = True if split[1].lower() == '.zip' else False
    # Fetch the file
    if not op.isfile(destination) or force_update:
        logger.info('Downloading data, please wait:')
        logger.info(url)
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _download_status(url, destination, False)

        # decompress if necessary
        if is_zip:
            z = zipfile.ZipFile(destination)
            decomp_dir, name = op.split(destination)
            files = z.namelist()
            stdout.write('Decompressing %g files from\n'
                         '"%s" ...' % (len(files), name))
            z.extractall(decomp_dir)
            z.close()
            destinations = [op.join(decomp_dir, f) for f in files]
            stdout.write(' [done]\n')

    # Offer to update the path
    path = op.abspath(path)
    if update_path is None:
        if  get_config('MNE_DATASETS_MEGSIM_PATH', '') != path:
            update_path = True
            msg = ('Do you want to set the path:\n    %s\nas the default '
                   'MEGSIM dataset path in the mne-python config ([y]/n)? '
                   % path)
            answer = raw_input(msg)
            if answer.lower() == 'n':
                update_path = False
        else:
            update_path = False
    if update_path is True:
        set_config('MNE_DATASETS_MEGSIM_PATH', path)

    return destinations


def load_data(condition='visual', data_format='raw', data_type='experimental',
              path=None, force_update=False, update_path=None):
    """Get path to local copy of MEGSIM dataset type

    Parameters
    ----------
    condition : str
        The condition to use. Either 'visual', 'auditory', or 'somatosensory'.
    data_format : str
        The data format. Either 'raw', 'evoked', or 'single-trial'.
    data_type : str
        The type of data. Either 'experimental' or 'simulation'.
    path : None | str
        Location of where to look for the MEGSIM data storing location.
        If None, the environment variable or config parameter
        MNE_DATASETS_MEGSIM_PATH is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the MEGSIM dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MEGSIM"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_MEGSIM_PATH in mne-python
        config to the given path. If None, the user is prompted.

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import megsim
        >>> megsim.load_megis('visual', 'raw', 'experimental', os.getenv('HOME') + '/datasets') # doctest:+SKIP

    And this would download the raw visual experimental MEGSIM dataset to the
    'datasets' folder, and prompt the user to save the 'datasets' path to the
    mne-python config, if it isn't there already.

    The MEGSIM dataset is documented in the following publication:
        Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T,
        Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM
        (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using
        Realistic Simulated and Empirical Data. Neuroinform 10:141-158
    """

    if not condition.lower() in valid_conditions:
        raise ValueError('Unknown condition "%s"' % condition)
    if not data_format in valid_data_formats:
        raise ValueError('Unknown data_format "%s"' % data_format)
    if not data_type in valid_data_types:
        raise ValueError('Unknown data_type "%s"' % data_type)
    urls = url_match(condition, data_format, data_type)

    data_paths = list()
    for url in urls:
        data_paths.extend(data_path(url, path, force_update, update_path))
    return data_paths
