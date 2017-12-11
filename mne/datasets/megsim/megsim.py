# Author: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
from os import path as op
import zipfile
from sys import stdout

from ...utils import _fetch_file, _url_to_local_path, verbose
from ..utils import _get_path, _do_path_update
from .urls import (url_match, valid_data_types, valid_data_formats,
                   valid_conditions)


@verbose
def data_path(url, path=None, force_update=False, update_path=None,
              verbose=None):
    """Get path to local copy of MEGSIM dataset URL.

    This is a low-level function useful for getting a local copy of a
    remote MEGSIM dataset [1]_.

    Parameters
    ----------
    url : str
        The dataset to use.
    path : None | str
        Location of where to look for the MEGSIM data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_MEGSIM_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the MEGSIM dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_MEGSIM_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

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

    References
    ----------
    .. [1] Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T,
           Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM
           (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using
           Realistic Simulated and Empirical Data. Neuroinform 10:141-158
    """  # noqa: E501
    key = 'MNE_DATASETS_MEGSIM_PATH'
    name = 'MEGSIM'
    path = _get_path(path, key, name)
    destination = _url_to_local_path(url, op.join(path, 'MEGSIM'))
    destinations = [destination]

    split = op.splitext(destination)
    is_zip = True if split[1].lower() == '.zip' else False
    # Fetch the file
    do_unzip = False
    if not op.isfile(destination) or force_update:
        if op.isfile(destination):
            os.remove(destination)
        if not op.isdir(op.dirname(destination)):
            os.makedirs(op.dirname(destination))
        _fetch_file(url, destination, print_destination=False)
        do_unzip = True

    if is_zip:
        z = zipfile.ZipFile(destination)
        decomp_dir, name = op.split(destination)
        files = z.namelist()
        # decompress if necessary (if download was re-done)
        if do_unzip:
            stdout.write('Decompressing %g files from\n'
                         '"%s" ...' % (len(files), name))
            z.extractall(decomp_dir)
            stdout.write(' [done]\n')
        z.close()
        destinations = [op.join(decomp_dir, f) for f in files]

    path = _do_path_update(path, update_path, key, name)
    return destinations


@verbose
def load_data(condition='visual', data_format='raw', data_type='experimental',
              path=None, force_update=False, update_path=None, verbose=None):
    """Get path to local copy of MEGSIM dataset type.

    The MEGSIM dataset is described in [1]_.

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
        ``MNE_DATASETS_MEGSIM_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the MEGSIM dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the ``MNE_DATASETS_MEGSIM_PATH`` in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    paths : list
        List of local data paths of the given type.

    Notes
    -----
    For example, one could do:

        >>> from mne.datasets import megsim
        >>> megsim.load_data('visual', 'raw', 'experimental', os.getenv('HOME') + '/datasets') # doctest:+SKIP

    And this would download the raw visual experimental MEGSIM dataset to the
    'datasets' folder, and prompt the user to save the 'datasets' path to the
    mne-python config, if it isn't there already.

    References
    ----------
    .. [1] Aine CJ, Sanfratello L, Ranken D, Best E, MacArthur JA, Wallace T,
           Gilliam K, Donahue CH, Montano R, Bryant JE, Scott A, Stephen JM
           (2012) MEG-SIM: A Web Portal for Testing MEG Analysis Methods using
           Realistic Simulated and Empirical Data. Neuroinform 10:141-158
    """  # noqa: E501
    if not condition.lower() in valid_conditions:
        raise ValueError('Unknown condition "%s"' % condition)
    if data_format not in valid_data_formats:
        raise ValueError('Unknown data_format "%s"' % data_format)
    if data_type not in valid_data_types:
        raise ValueError('Unknown data_type "%s"' % data_type)
    urls = url_match(condition, data_format, data_type)

    data_paths = list()
    for url in urls:
        data_paths.extend(data_path(url, path, force_update, update_path))
    return data_paths
