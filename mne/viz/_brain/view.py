# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

_lh_views_dict = {
    'lateral': dict(azimuth=180., elevation=90.),
    'medial': dict(azimuth=0., elevation=90.0),
    'rostral': dict(azimuth=90., elevation=90.),
    'caudal': dict(azimuth=270., elevation=90.),
    'dorsal': dict(azimuth=180., elevation=0.),
    'ventral': dict(azimuth=180., elevation=180.),
    'frontal': dict(azimuth=120., elevation=80.),
    'parietal': dict(azimuth=-120., elevation=60.),
    'sagittal': dict(azimuth=180., elevation=-90.),
    'coronal': dict(azimuth=90., elevation=-90.),
    'axial': dict(azimuth=180., elevation=0., roll=180),
}
_rh_views_dict = {
    'lateral': dict(azimuth=180., elevation=-90.),
    'medial': dict(azimuth=0., elevation=-90.0),
    'rostral': dict(azimuth=-90., elevation=-90.),
    'caudal': dict(azimuth=90., elevation=-90.),
    'dorsal': dict(azimuth=180., elevation=0.),
    'ventral': dict(azimuth=180., elevation=180.),
    'frontal': dict(azimuth=60., elevation=80.),
    'parietal': dict(azimuth=-60., elevation=60.),
    'sagittal': dict(azimuth=180., elevation=-90.),
    'coronal': dict(azimuth=90., elevation=-90.),
    'axial': dict(azimuth=180., elevation=0., roll=180),
}
# add short-size version entries into the dict
lh_views_dict = _lh_views_dict.copy()
for k, v in _lh_views_dict.items():
    lh_views_dict[k[:3]] = v
    lh_views_dict['flat'] = dict(azimuth=250, elevation=0)

rh_views_dict = _rh_views_dict.copy()
for k, v in _rh_views_dict.items():
    rh_views_dict[k[:3]] = v
    rh_views_dict['flat'] = dict(azimuth=-70, elevation=0)
views_dicts = dict(lh=lh_views_dict, vol=lh_views_dict, both=lh_views_dict,
                   rh=rh_views_dict)
