# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

from collections import namedtuple


View = namedtuple('View', 'elev azim')

views_dict = {'lateral': View(elev=5, azim=0),
              'medial': View(elev=5, azim=180),
              'rostral': View(elev=5, azim=90),
              'caudal': View(elev=5, azim=-90),
              'dorsal': View(elev=90, azim=0),
              'ventral': View(elev=-90, azim=0),
              'frontal': View(elev=5, azim=110),
              'parietal': View(elev=5, azim=-110)}
# add short-size version entries into the dict
_views_dict = dict()
for k, v in views_dict.items():
    _views_dict[k[:3]] = v
views_dict.update(_views_dict)
