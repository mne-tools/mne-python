# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ..utils import Bunch, _check_fname, _soft_import, warn


def _read_dig_montage_egi(
    fname,
    _scaling,
    _all_data_kwargs_are_none,
):
    if not _all_data_kwargs_are_none:
        raise ValueError(
            "hsp, hpi, elp, point_names, fif must all be None if egi is not None"
        )
    _check_fname(fname, overwrite="read", must_exist=True)
    defusedxml = _soft_import("defusedxml", "reading EGI montages")
    root = defusedxml.ElementTree.parse(fname).getroot()
    ns = root.tag[root.tag.index("{") : root.tag.index("}") + 1]
    sensors = root.find(f"{ns}sensorLayout/{ns}sensors")
    fids = dict()
    dig_ch_pos = dict()

    fid_name_map = {
        "Nasion": "nasion",
        "Right periauricular point": "rpa",
        "Left periauricular point": "lpa",
    }

    for s in sensors:
        name, number, kind = s[0].text, int(s[1].text), int(s[2].text)
        coordinates = np.array([float(s[3].text), float(s[4].text), float(s[5].text)])

        coordinates *= _scaling

        # EEG Channels
        if kind == 0:
            dig_ch_pos[f"EEG {number:03d}"] = coordinates
        # Reference
        elif kind == 1:
            dig_ch_pos[f"EEG {len(dig_ch_pos) + 1:03d}"] = coordinates
        # Fiducials
        elif kind == 2:
            fid_name = fid_name_map[name]
            fids[fid_name] = coordinates
        # Unknown
        else:
            warn(
                f"Unknown sensor type {kind} detected. Skipping sensor..."
                "Proceed with caution!"
            )

    return Bunch(
        # EGI stuff
        nasion=fids["nasion"],
        lpa=fids["lpa"],
        rpa=fids["rpa"],
        ch_pos=dig_ch_pos,
        coord_frame="unknown",
    )


def _parse_brainvision_dig_montage(fname, scale):
    FID_NAME_MAP = {"Nasion": "nasion", "RPA": "rpa", "LPA": "lpa"}
    defusedxml = _soft_import("defusedxml", "reading BrainVision montages")
    root = defusedxml.ElementTree.parse(fname).getroot()
    sensors = root.find("CapTrakElectrodeList")

    fids, dig_ch_pos = dict(), dict()

    for s in sensors:
        name = s.find("Name").text

        is_fid = name in FID_NAME_MAP
        coordinates = scale * np.array(
            [float(s.find("X").text), float(s.find("Y").text), float(s.find("Z").text)]
        )

        # Fiducials
        if is_fid:
            fids[FID_NAME_MAP[name]] = coordinates
        # EEG Channels
        else:
            dig_ch_pos[name] = coordinates

    return dict(
        # BVCT stuff
        nasion=fids["nasion"],
        lpa=fids["lpa"],
        rpa=fids["rpa"],
        ch_pos=dig_ch_pos,
        coord_frame="unknown",
    )
