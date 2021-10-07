#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Authors: Jussi Nurminen <jnu@iki.fi>
# License: BSD Style.


import os.path as op
import os
from ...utils import verbose, _check_option
from ..utils import _get_path, _do_path_update, _data_path


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
    # check if data already exists, if so bail early
    path = _get_path(path, 'MNE_DATASETS_HF_SEF_PATH', 'HF_SEF')
    final_path = op.join(path, 'HF_SEF')
    megdir = op.join(final_path, 'MEG', 'subject_a')
    has_raw = (dataset == 'raw' and op.isdir(megdir) and
               any('raw' in filename for filename in os.listdir(megdir)))
    has_evoked = (dataset == 'evoked' and
                  op.isdir(op.join(final_path, 'subjects')))
    if has_raw or has_evoked and not force_update:
        _do_path_update(path, update_path, 'MNE_DATASETS_HF_SEF_PATH',
                        'HF_SEF')
        return final_path
    # data not there, or force_update requested:
    name = f'hf_sef_{dataset}'
    return _data_path(path=path, force_update=force_update,
                      update_path=update_path, name=name)
