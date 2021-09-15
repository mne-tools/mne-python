# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

ORIGIN = 'auto'

_lh_views_dict = {
    'lateral': dict(azimuth=180., elevation=90., focalpoint=ORIGIN),
    'medial': dict(azimuth=0., elevation=90.0, focalpoint=ORIGIN),
    'rostral': dict(azimuth=90., elevation=90., focalpoint=ORIGIN),
    'caudal': dict(azimuth=270., elevation=90., focalpoint=ORIGIN),
    'dorsal': dict(azimuth=180., elevation=0., focalpoint=ORIGIN),
    'ventral': dict(azimuth=180., elevation=180., focalpoint=ORIGIN),
    'frontal': dict(azimuth=120., elevation=80., focalpoint=ORIGIN),
    'parietal': dict(azimuth=-120., elevation=60., focalpoint=ORIGIN),
    'sagittal': dict(azimuth=180., elevation=-90., focalpoint=ORIGIN),
    'coronal': dict(azimuth=90., elevation=-90., focalpoint=ORIGIN),
    'axial': dict(azimuth=180., elevation=0., focalpoint=ORIGIN, roll=0),
}
_rh_views_dict = {
    'lateral': dict(azimuth=180., elevation=-90., focalpoint=ORIGIN),
    'medial': dict(azimuth=0., elevation=-90.0, focalpoint=ORIGIN),
    'rostral': dict(azimuth=-90., elevation=-90., focalpoint=ORIGIN),
    'caudal': dict(azimuth=90., elevation=-90., focalpoint=ORIGIN),
    'dorsal': dict(azimuth=180., elevation=0., focalpoint=ORIGIN),
    'ventral': dict(azimuth=180., elevation=180., focalpoint=ORIGIN),
    'frontal': dict(azimuth=60., elevation=80., focalpoint=ORIGIN),
    'parietal': dict(azimuth=-60., elevation=60., focalpoint=ORIGIN),
    'sagittal': dict(azimuth=180., elevation=-90., focalpoint=ORIGIN),
    'coronal': dict(azimuth=90., elevation=-90., focalpoint=ORIGIN),
    'axial': dict(azimuth=180., elevation=0., focalpoint=ORIGIN, roll=0),
}
# add short-size version entries into the dict
lh_views_dict = _lh_views_dict.copy()
for k, v in _lh_views_dict.items():
    lh_views_dict[k[:3]] = v
    lh_views_dict['flat'] = dict(
        azimuth=0, elevation=0, focalpoint=ORIGIN, roll=0)

rh_views_dict = _rh_views_dict.copy()
for k, v in _rh_views_dict.items():
    rh_views_dict[k[:3]] = v
    rh_views_dict['flat'] = dict(
        azimuth=0, elevation=0, focalpoint=ORIGIN, roll=0)
views_dicts = dict(lh=lh_views_dict, vol=lh_views_dict, both=lh_views_dict,
                   rh=rh_views_dict)
