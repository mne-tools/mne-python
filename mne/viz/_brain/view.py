# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

from collections import namedtuple


View = namedtuple('View', 'elev azim')

views_dict = {'lateral': View(azim=180., elev=90.),
              'medial': View(azim=0., elev=90.0),
              'rostral': View(azim=90., elev=90.),
              'caudal': View(azim=270., elev=90.),
              'dorsal': View(azim=180., elev=0.),
              'ventral': View(azim=180., elev=180.),
              'frontal': View(azim=120., elev=80.),
              'parietal': View(azim=-120., elev=60.)}
# add short-size version entries into the dict
_views_dict = dict()
for k, v in views_dict.items():
    _views_dict[k[:3]] = v
views_dict.update(_views_dict)
