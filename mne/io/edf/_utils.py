# -*- coding: utf-8 -*-
"""Helper functions for EDF, EDF+, BDF converters to FIF."""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import re

from ...utils import hashfunc


def _load_gdf_events_lut():
    fname = op.join(op.dirname(__file__), 'gdf_encodes.txt')
    hash_ = hashfunc(fname, hash_type='md5')
    # Linux or Windows line endings
    if hash_ not in ('12134a9be7e0bfa5941e95f8bfd330f7',
                     '41696b162526f559dd7d139352eae178'):
        raise ValueError("File %s is corrupted. mdf5 hashes don't match." %
                         fname)

    # load the stuff
    with open(fname, 'r') as fh:
        elements = [line for line in fh if not line.startswith("#")]

    event_id, event_name = list(), list()
    for elem in elements:
        event_id_i, *event_name_i = elem.split('\t')
        event_id.append(int(event_id_i, 0))
        clean_name = re.sub('[ \t]+', ' ', ' '.join(event_name_i))
        clean_name = re.sub('\n', '', clean_name)
        event_name.append(clean_name)

    return dict(zip(event_id, event_name))
