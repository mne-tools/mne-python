# Authors: The MNE-Python contributors.
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

both_views_dict = lh_views_dict.copy()
both_views_dict["right_lateral"] = dict(
    azimuth=180.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["right_anterolateral"] = dict(
    azimuth=120.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["anterior"] = dict(
    azimuth=90.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["left_anterolateral"] = dict(
    azimuth=60.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["left_lateral"] = dict(
    azimuth=180.0, elevation=-90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["right_posterolateral"] = dict(
    azimuth=-120.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["posterior"] = dict(
    azimuth=90.0, elevation=-90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["left_posterolateral"] = dict(
    azimuth=-60.0, elevation=90.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["superior"] = dict(
    azimuth=180.0, elevation=0.0, focalpoint=ORIGIN, distance=DIST
)
both_views_dict["inferior"] = dict(
    azimuth=180.0, elevation=180.0, focalpoint=ORIGIN, distance=DIST
)


views_dicts = dict(
    lh=lh_views_dict, vol=lh_views_dict, both=both_views_dict, rh=rh_views_dict
)
