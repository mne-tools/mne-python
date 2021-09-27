#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Authors: Jussi Nurminen <jnu@iki.fi>
# License: BSD Style.


import os.path as op
import os
from ...utils import verbose, _check_option
from ..utils import _get_path, _do_path_update, _download_mne_dataset
from ..config import MNE_DATASETS


@verbose
def data_path(dataset='evoked', path=None, force_update=False,
              update_path=True, verbose=None):
    u"""Get path to local copy of the high frequency SEF dataset.

    Gets a local copy of the high frequency SEF MEG dataset
    :footcite:`NurminenEtAl2017`.

    Parameters
    ----------
    dataset : 'evoked' | 'raw'
        Whether to get the main dataset (evoked, structural and the rest) or
        the separate dataset containing raw MEG data only.
    path : None | str
        Where to look for the HF-SEF data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_HF_SEF_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the HF-SEF dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_HF_SEF_PATH in mne-python
        config to the given path. If None, the user is prompted.
    %(verbose)s

    Returns
    -------
    path : str
        Local path to the directory where the HF-SEF data is stored.

    References
    ----------
    .. footbibliography::
    """
    _check_option('dataset', dataset, ('evoked', 'raw'))
    if dataset == 'raw':
        data_dict = MNE_DATASETS['hf_sef_raw']
        data_dict['dataset_name'] = 'hf_sef_raw'
    else:
        data_dict = MNE_DATASETS['hf_sef_evoked']
        data_dict['dataset_name'] = 'hf_sef_evoked'
    config_key = data_dict['config_key']
    folder_name = data_dict['folder_name']

    # get download path for specific dataset
    path = _get_path(path=path, key=config_key, name=folder_name)
    final_path = op.join(path, folder_name)
    megdir = op.join(final_path, 'MEG', 'subject_a')
    has_raw = (dataset == 'raw' and op.isdir(megdir) and
               any('raw' in filename for filename in os.listdir(megdir)))
    has_evoked = (dataset == 'evoked' and
                  op.isdir(op.join(final_path, 'subjects')))
    # data not there, or force_update requested:
    if has_raw or has_evoked and not force_update:
        _do_path_update(path, update_path, config_key,
                        folder_name)
        return final_path

    # instantiate processor that unzips file
    data_path = _download_mne_dataset(name=data_dict['dataset_name'],
                                      processor='untar', path=path,
                                      force_update=force_update,
                                      update_path=update_path, download=True)
    return data_path
