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


def _get_blocks(filepath):
    """Get info from meta data blocks."""
    binfile = os.path.join(filepath)
    n_blocks = 0
    samples_block = []
    header_sizes = []
    n_channels = []
    sfreq = []
    # Meta data consists of:
    # * 1 byte of flag (1 for meta data, 0 for data)
    # * 1 byte of header size
    # * 1 byte of block size
    # * 1 byte of n_channels
    # * n_channels bytes of offsets
    # * n_channels bytes of sigfreqs?
    with open(binfile, "rb") as fid:
        fid.seek(0, 2)  # go to end of file
        file_length = fid.tell()
        block_size = file_length
        fid.seek(0)
        position = 0
        while position < file_length:
            block = _block_r(fid)
            if block is None:
                samples_block.append(samples_block[n_blocks - 1])
                n_blocks += 1
                fid.seek(block_size, 1)
                position = fid.tell()
                continue
            block_size = block["block_size"]
            header_size = block["header_size"]
            header_sizes.append(header_size)
            samples_block.append(block["nsamples"])
            n_blocks += 1
            fid.seek(block_size, 1)
            sfreq.append(block["sfreq"])
            n_channels.append(block["nc"])
            position = fid.tell()

    if any([n != n_channels[0] for n in n_channels]):
        raise RuntimeError("All the blocks don't have the same amount of channels.")
    if any([f != sfreq[0] for f in sfreq]):
        raise RuntimeError("All the blocks don't have the same sampling frequency.")
    if len(samples_block) < 1:
        raise RuntimeError("There seems to be no data")
    samples_block = np.array(samples_block)
    signal_blocks = dict(
        n_channels=n_channels[0],
        sfreq=sfreq[0],
        n_blocks=n_blocks,
        samples_block=samples_block,
        header_sizes=header_sizes,
    )
    return signal_blocks


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
        bin_num_str = re.search(r"\d+", binfile).group()
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


def _block_r(fid):
    """Read meta data."""
    if np.fromfile(fid, dtype=np.dtype("i4"), count=1).item() != 1:  # not meta
        return None
    header_size = np.fromfile(fid, dtype=np.dtype("i4"), count=1).item()
    block_size = np.fromfile(fid, dtype=np.dtype("i4"), count=1).item()
    hl = int(block_size / 4)
    nc = np.fromfile(fid, dtype=np.dtype("i4"), count=1).item()
    nsamples = int(hl / nc)
    np.fromfile(fid, dtype=np.dtype("i4"), count=nc)  # sigoffset
    sigfreq = np.fromfile(fid, dtype=np.dtype("i4"), count=nc)
    depth = sigfreq[0] & 0xFF
    if depth != 32:
        raise ValueError("I do not know how to read this MFF (depth != 32)")
    sfreq = sigfreq[0] >> 8
    count = int(header_size / 4 - (4 + 2 * nc))
    np.fromfile(fid, dtype=np.dtype("i4"), count=count)  # sigoffset
    block = dict(
        nc=nc,
        hl=hl,
        nsamples=nsamples,
        block_size=block_size,
        header_size=header_size,
        sfreq=sfreq,
    )
    return block
