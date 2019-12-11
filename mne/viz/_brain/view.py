# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

from collections import namedtuple


View = namedtuple('View', 'elev azim')

lh_views_dict = {'lateral': View(azim=180., elev=90.),
                 'medial': View(azim=0., elev=90.0),
                 'rostral': View(azim=90., elev=90.),
                 'caudal': View(azim=270., elev=90.),
                 'dorsal': View(azim=180., elev=0.),
                 'ventral': View(azim=180., elev=180.),
                 'frontal': View(azim=120., elev=80.),
                 'parietal': View(azim=-120., elev=60.)}
rh_views_dict = {'lateral': View(azim=180., elev=-90.),
                 'medial': View(azim=0., elev=-90.0),
                 'rostral': View(azim=-90., elev=-90.),
                 'caudal': View(azim=90., elev=-90.),
                 'dorsal': View(azim=180., elev=0.),
                 'ventral': View(azim=180., elev=180.),
                 'frontal': View(azim=60., elev=80.),
                 'parietal': View(azim=-60., elev=60.)}
# add short-size version entries into the dict
_lh_views_dict = dict()
for k, v in lh_views_dict.items():
    _lh_views_dict[k[:3]] = v
lh_views_dict.update(_lh_views_dict)

_rh_views_dict = dict()
for k, v in rh_views_dict.items():
    _rh_views_dict[k[:3]] = v
rh_views_dict.update(_rh_views_dict)
