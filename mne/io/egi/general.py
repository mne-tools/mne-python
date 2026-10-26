#
# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import re

from ...utils import _pl


def _get_ep_info(filepath):
    """Get epoch info."""
    from mffpy.xml_files import XML

    ep_obj = XML.from_file(os.path.join(filepath, "epochs.xml"))
    keys = ("first_samps", "last_samps", "first_blocks", "last_blocks")
    epoch_info = {key: list() for key in keys}
    for ep in ep_obj.epochs:
        epoch_info["first_samps"].append(ep.beginTime)
        epoch_info["last_samps"].append(ep.endTime)
        epoch_info["first_blocks"].append(ep.firstBlock)
        epoch_info["last_blocks"].append(ep.lastBlock)
    # Don't turn into ndarray here, keep native int because it can deal with
    # huge numbers (could use np.uint64 but it's more work)
    return epoch_info


def _get_signalfname(filepath):
    """Get filenames."""
    from mffpy.xml_files import XML

    listfiles = os.listdir(filepath)
    binfiles = list(
        f for f in listfiles if "signal" in f and f[-4:] == ".bin" and f[0] != "."
    )
    all_files = {}
    infofiles = list()
    for binfile in binfiles:
        match = re.search(r"\d+", binfile)
        assert match is not None
        bin_num_str = match.group()
        infofile = "info" + bin_num_str + ".xml"
        infofiles.append(infofile)
        info_obj = XML.from_file(os.path.join(filepath, infofile))
        channel_type = (
            info_obj.get_content().get("generalInformation", {}).get("channel_type", "")
        )
        if channel_type == "EEG":
            all_files["EEG"] = {"signal": f"signal{bin_num_str}.bin", "info": infofile}
        elif channel_type == "PNSData":
            all_files["PNS"] = {"signal": f"signal{bin_num_str}.bin", "info": infofile}
    if "EEG" not in all_files:
        infofiles_str = "\n".join(infofiles)
        raise FileNotFoundError(
            f"Could not find any EEG data in the {len(infofiles)} file{_pl(infofiles)} "
            f"found in {filepath}:\n{infofiles_str}"
        )
    return all_files
