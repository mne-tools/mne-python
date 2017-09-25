#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Authors: Jussi Nurminen <jnu@iki.fi>
# License: BSD Style.


import tarfile
import os.path as op
import os
from ...utils import _fetch_file, verbose
from ..utils import _get_path, logger, _do_path_update


@verbose
def data_path(set='evoked', path=None, force_update=False, update_path=None,
              verbose=None):

    key = 'MNE_DATASETS_HF_SEF_PATH'
    name = 'HF_SEF'
    path = _get_path(path, key, name)
    destdir = op.join(path, 'HF_SEF')

    urls = {'evoked':
            'https://zenodo.org/record/889235/files/hf_sef_evoked.tar.gz',
            'raw':
            'https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz'}
    if set not in urls:
        raise ValueError('Invalid set specified')
    url = urls[set]
    fn = url.split('/')[-1]  # pick the filename from the url
    archive = op.join(destdir, fn)

    if not op.isdir(destdir) or force_update:
        if op.isfile(archive):
            os.remove(archive)
        if not op.isdir(destdir):
            os.mkdir(destdir)
        _fetch_file(url, archive)

        with tarfile.open(archive) as tar:
            logger.info('Decompressing %s' % archive)
            for member in tar.getmembers():
                # strip the leading dirname 'hf_sef/' from the archive paths
                # this should be fixed when making next version of archives
                member.name = member.name[7:]
                tar.extract(member, destdir)

        os.remove(archive)

    path = _do_path_update(path, update_path, key, name)
    return destdir
