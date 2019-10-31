#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Authors: Jussi Nurminen <jnu@iki.fi>
# License: BSD Style.


import tarfile
import os.path as op
import os
from ...utils import _fetch_file, verbose, _check_option
from ..utils import _get_path, logger, _do_path_update


@verbose
def data_path(dataset='evoked', path=None, force_update=False,
              update_path=True, verbose=None):
    u"""Get path to local copy of the high frequency SEF dataset.

    Gets a local copy of the high frequency SEF MEG dataset [1]_.

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
    .. [1] Nurminen, J., Paananen, H., Mäkelä, J. (2017): High frequency
           somatosensory MEG dataset. https://doi.org/10.5281/zenodo.889234
    """
    key = 'MNE_DATASETS_HF_SEF_PATH'
    name = 'HF_SEF'
    path = _get_path(path, key, name)
    destdir = op.join(path, 'HF_SEF')

    urls = {'evoked':
            'https://zenodo.org/record/3523071/files/hf_sef_evoked.tar.gz',
            'raw':
            'https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz'}
    hashes = {'evoked': '13d34cb5db584e00868677d8fb0aab2b',
              'raw': '33934351e558542bafa9b262ac071168'}
    _check_option('dataset', dataset, sorted(urls.keys()))
    url = urls[dataset]
    hash_ = hashes[dataset]
    fn = url.split('/')[-1]  # pick the filename from the url
    archive = op.join(destdir, fn)

    # check for existence of evoked and raw sets
    has = dict()
    subjdir = op.join(destdir, 'subjects')
    megdir_a = op.join(destdir, 'MEG', 'subject_a')
    has['evoked'] = op.isdir(destdir) and op.isdir(subjdir)
    has['raw'] = op.isdir(megdir_a) and any(['raw' in fn_ for fn_ in
                                             os.listdir(megdir_a)])

    if not has[dataset] or force_update:
        if not op.isdir(destdir):
            os.mkdir(destdir)
        _fetch_file(url, archive, hash_=hash_)

        with tarfile.open(archive) as tar:
            logger.info('Decompressing %s' % archive)
            for member in tar.getmembers():
                # strip the leading dirname 'hf_sef/' from the archive paths
                # this should be fixed when making next version of archives
                member.name = member.name[7:]
                try:
                    tar.extract(member, destdir)
                except IOError:
                    # check whether file exists but could not be overwritten
                    fn_full = op.join(destdir, member.name)
                    if op.isfile(fn_full):
                        os.remove(fn_full)
                        tar.extract(member, destdir)
                    else:  # some more sinister cause for IOError
                        raise

        os.remove(archive)

    _do_path_update(path, update_path, key, name)
    return destdir
