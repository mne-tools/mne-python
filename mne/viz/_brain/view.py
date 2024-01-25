# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larson.eric.d@gmail.com>
#          Oleh Kozynets <ok7mailbox@gmail.com>
#          Guillaume Favelier <guillaume.favelier@gmail.com>
#          jona-sassenhagen <jona.sassenhagen@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

ORIGIN = "auto"
DIST = "auto"

_lh_views_dict = {
    "lateral": dict(azimuth=180.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "medial": dict(azimuth=0.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "rostral": dict(azimuth=90.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "caudal": dict(azimuth=270.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "dorsal": dict(azimuth=180.0, elevation=0.0, focalpoint=ORIGIN, distance=DIST),
    "ventral": dict(azimuth=180.0, elevation=180.0, focalpoint=ORIGIN, distance=DIST),
    "frontal": dict(azimuth=120.0, elevation=80.0, focalpoint=ORIGIN, distance=DIST),
    "parietal": dict(azimuth=-120.0, elevation=60.0, focalpoint=ORIGIN, distance=DIST),
    "sagittal": dict(azimuth=180.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "coronal": dict(azimuth=90.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "axial": dict(
        azimuth=180.0, elevation=0.0, focalpoint=ORIGIN, roll=0, distance=DIST
    ),  # noqa: E501
}
_rh_views_dict = {
    "lateral": dict(azimuth=180.0, elevation=-90.0, focalpoint=ORIGIN, distance=DIST),
    "medial": dict(azimuth=0.0, elevation=-90.0, focalpoint=ORIGIN, distance=DIST),
    "rostral": dict(azimuth=-90.0, elevation=-90.0, focalpoint=ORIGIN, distance=DIST),
    "caudal": dict(azimuth=90.0, elevation=-90.0, focalpoint=ORIGIN, distance=DIST),
    "dorsal": dict(azimuth=180.0, elevation=0.0, focalpoint=ORIGIN, distance=DIST),
    "ventral": dict(azimuth=180.0, elevation=180.0, focalpoint=ORIGIN, distance=DIST),
    "frontal": dict(azimuth=60.0, elevation=80.0, focalpoint=ORIGIN, distance=DIST),
    "parietal": dict(azimuth=-60.0, elevation=60.0, focalpoint=ORIGIN, distance=DIST),
    "sagittal": dict(azimuth=180.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "coronal": dict(azimuth=90.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST),
    "axial": dict(
        azimuth=180.0, elevation=0.0, focalpoint=ORIGIN, roll=0, distance=DIST
    ),
}
# add short-size version entries into the dict
lh_views_dict = _lh_views_dict.copy()
for k, v in _lh_views_dict.items():
    lh_views_dict[k[:3]] = v
    lh_views_dict["flat"] = dict(
        azimuth=0, elevation=0, focalpoint=ORIGIN, roll=0, distance=DIST
    )

rh_views_dict = _rh_views_dict.copy()
for k, v in _rh_views_dict.items():
    rh_views_dict[k[:3]] = v
    rh_views_dict["flat"] = dict(
        azimuth=0, elevation=0, focalpoint=ORIGIN, roll=0, distance=DIST
    )
views_dicts = dict(
    lh=lh_views_dict, vol=lh_views_dict, both=lh_views_dict, rh=rh_views_dict
)
