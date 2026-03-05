#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import re

import numpy as np

from ...utils import _pl, _soft_import


def _extract(tags, filepath=None, obj=None):
    """Extract info from XML."""
    _soft_import("defusedxml", "reading EGI MFF data")
    from defusedxml.minidom import parse

    if obj is not None:
        fileobj = obj
    elif filepath is not None:
        fileobj = parse(filepath)
    else:
        raise ValueError("There is not object or file to extract data")
    infoxml = dict()
    for tag in tags:
        value = fileobj.getElementsByTagName(tag)
        infoxml[tag] = []
        for i in range(len(value)):
            infoxml[tag].append(value[i].firstChild.data)
    return infoxml


def _get_gains(filepath):
    """Parse gains."""
    _soft_import("defusedxml", "reading EGI MFF data")
    from defusedxml.minidom import parse

    file_obj = parse(filepath)
    objects = file_obj.getElementsByTagName("calibration")
    gains = dict()
    for ob in objects:
        value = ob.getElementsByTagName("type")
        if value[0].firstChild.data == "GCAL":
            data_g = _extract(["ch"], obj=ob)["ch"]
            gains.update(gcal=np.asarray(data_g, dtype=np.float64))
        elif value[0].firstChild.data == "ICAL":
            data_g = _extract(["ch"], obj=ob)["ch"]
            gains.update(ical=np.asarray(data_g, dtype=np.float64))
    return gains


def _get_signalfname(filepath):
    """Get filenames."""
    _soft_import("defusedxml", "reading EGI MFF data")
    from defusedxml.minidom import parse

    listfiles = os.listdir(filepath)
    binfiles = list(
        f for f in listfiles if "signal" in f and f[-4:] == ".bin" and f[0] != "."
    )
    all_files = {}
    infofiles = list()
    for binfile in binfiles:
        bin_num_str = re.search(r"\d+", binfile).group()
        infofile = "info" + bin_num_str + ".xml"
        infofiles.append(infofile)
        infobjfile = os.path.join(filepath, infofile)
        infobj = parse(infobjfile)
        if len(infobj.getElementsByTagName("EEG")):
            signal_type = "EEG"
        elif len(infobj.getElementsByTagName("PNSData")):
            signal_type = "PNS"
        all_files[signal_type] = {
            "signal": f"signal{bin_num_str}.bin",
            "info": infofile,
        }
    if "EEG" not in all_files:
        infofiles_str = "\n".join(infofiles)
        raise FileNotFoundError(
            f"Could not find any EEG data in the {len(infofiles)} file{_pl(infofiles)} "
            f"found in {filepath}:\n{infofiles_str}"
        )
    return all_files
