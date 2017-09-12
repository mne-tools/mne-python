#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:45:03 2017

@author: jussi
"""

import tarfile
from ...utils import _fetch_file, _url_to_local_path, verbose


def data_path(set='evoked', path=None, force_update=False, update_path=None,
              verbose=None):

    key = MNE_DATASETS_HF_SEF_PATH
    name = HF_SEF
    path = _get_path(path, key, name)
    
    
    urls = {'evoked':
            'https://zenodo.org/record/889235/files/hf_sef_evoked.tar.gz',
            'raw':
            'https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz'}

    if set not in urls:
        raise ValueError('Invalid set specified')


    url = urls[set]
    _fetch_file(url)

    with tarfile.open(fn) as tar:
        tar.extractall('/tmp')

        


   