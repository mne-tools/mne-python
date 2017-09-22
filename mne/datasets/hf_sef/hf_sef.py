#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Authors: Jussi Nurminen <jnu@iki.fi>
# License: BSD Style.

import tarfile
import os.path as op
import os
from ...utils import _fetch_file
from ..utils import _get_path


def data_path(set='evoked', path=None, force_update=False, update_path=None,
              verbose=None):

    key = 'MNE_DATASETS_HF_SEF_PATH'
    name = 'HF_SEF'
    path = _get_path(path, key, name)
    destdir = op.join(path, 'HF_SEF')

    if not op.isdir(destdir):
        os.mkdir(destdir)

    urls = {'evoked':
            'https://zenodo.org/record/889235/files/hf_sef_evoked.tar.gz',
            'raw':
            'https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz'}

    if set not in urls:
        raise ValueError('Invalid set specified')

    url = urls[set]
    filename = url.split('/')[-1]  # probably politically incorrect
    destination = op.join(destdir, filename)

    _fetch_file(url, destination)

    with tarfile.open(destination) as tar:
        tar.extractall(destdir)
