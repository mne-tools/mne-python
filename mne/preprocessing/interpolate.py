"""Tools for data interpolation."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from itertools import chain

import numpy as np
from scipy.sparse.csgraph import connected_components

from .._fiff.meas_info import create_info
from ..epochs import BaseEpochs, EpochsArray
from ..evoked import Evoked, EvokedArray
from ..io import BaseRaw, RawArray
from ..transforms import _cart_to_sph, _sph_to_cart
from ..utils import _ensure_int, _validate_type


def equalize_bads(insts, interp_thresh=1.0, copy=True):
    """Interpolate or mark bads consistently for a list of instances.

    Once called on a list of instances, the instances can be concatenated
    as they will have the same list of bad channels.

    Parameters
    ----------
    insts : list
        The list of instances (Evoked, Epochs or Raw) to consider
        for interpolation. Each instance should have marked channels.
    interp_thresh : float
        A float between 0 and 1 (default) that specifies the fraction of time
        a channel should be good to be eventually interpolated for certain
        instances. For example if 0.5, a channel which is good at least half
        of the time will be interpolated in the instances where it is marked
        as bad. If 1 then channels will never be interpolated and if 0 all bad
        channels will be systematically interpolated.
    copy : bool
        If True then the returned instances will be copies.

    Returns
    -------
    insts_bads : list
        The list of instances, with the same channel(s) marked as bad in all of
        them, possibly with some formerly bad channels interpolated.
    """
    if not 0 <= interp_thresh <= 1:
        raise ValueError(f"interp_thresh must be between 0 and 1, got {interp_thresh}")

    all_bads = list(set(chain.from_iterable([inst.info["bads"] for inst in insts])))
    if isinstance(insts[0], BaseEpochs):
        durations = [len(inst) * len(inst.times) for inst in insts]
    else:
        durations = [len(inst.times) for inst in insts]

    good_times = []
    for ch_name in all_bads:
        good_times.append(
            sum(
                durations[k]
                for k, inst in enumerate(insts)
                if ch_name not in inst.info["bads"]
            )
            / np.sum(durations)
        )

    bads_keep = [ch for k, ch in enumerate(all_bads) if good_times[k] < interp_thresh]
    if copy:
        insts = [inst.copy() for inst in insts]

    for inst in insts:
        if len(set(inst.info["bads"]) - set(bads_keep)):
            inst.interpolate_bads(exclude=bads_keep)
        inst.info["bads"] = bads_keep

    return insts


def interpolate_bridged_electrodes(inst, bridged_idx, bad_limit=4):
    """Interpolate bridged electrode pairs.

    Because bridged electrodes contain brain signal, it's just that the
    signal is spatially smeared between the two electrodes, we can
    make a virtual channel midway between the bridged pairs and use
    that to aid in interpolation rather than completely discarding the
    data from the two channels.

    Parameters
    ----------
    inst : instance of Epochs, Evoked, or Raw
        The data object with channels that are to be interpolated.
    bridged_idx : list of tuple
        The indices of channels marked as bridged with each bridged
        pair stored as a tuple.
    bad_limit : int
        The maximum number of electrodes that can be bridged together
        (included) and interpolated. Above this number, an error will be
        raised.

        .. versionadded:: 1.2

    Returns
    -------
    inst : instance of Epochs, Evoked, or Raw
        The modified data object.

    See Also
    --------
    mne.preprocessing.compute_bridged_electrodes
    """
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked))
    bad_limit = _ensure_int(bad_limit, "bad_limit")
    if bad_limit <= 0:
        raise ValueError(
            "Argument 'bad_limit' should be a strictly positive "
            f"integer. Provided {bad_limit} is invalid."
        )
    montage = inst.get_montage()
    if montage is None:
        raise RuntimeError("No channel positions found in ``inst``")
    pos = montage.get_positions()
    if pos["coord_frame"] != "head":
        raise RuntimeError(
            f"Montage channel positions must be in ``head`` got {pos['coord_frame']}"
        )
    # store bads orig to put back at the end
    bads_orig = inst.info["bads"]
    inst.info["bads"] = list()

    # look for group of bad channels
    nodes = sorted(set(chain(*bridged_idx)))
    G_dense = np.zeros((len(nodes), len(nodes)))
    # fill the edges with a weight of 1
    for bridge in bridged_idx:
        idx0 = np.searchsorted(nodes, bridge[0])
        idx1 = np.searchsorted(nodes, bridge[1])
        G_dense[idx0, idx1] = 1
        G_dense[idx1, idx0] = 1
    # look for connected components
    _, labels = connected_components(G_dense, directed=False)
    groups_idx = [[nodes[j] for j in np.where(labels == k)[0]] for k in set(labels)]
    groups_names = [
        [inst.info.ch_names[k] for k in group_idx] for group_idx in groups_idx
    ]

    # warn for all bridged areas that include too many electrodes
    for group_names in groups_names:
        if len(group_names) > bad_limit:
            raise RuntimeError(
                f"The channels {', '.join(group_names)} are bridged together "
                "and form a large area of bridged electrodes. Interpolation "
                "might be inaccurate."
            )

    # make virtual channels
    virtual_chs = dict()
    bads = set()
    for k, group_idx in enumerate(groups_idx):
        group_names = [inst.info.ch_names[k] for k in group_idx]
        bads = bads.union(group_names)
        # compute centroid position in spherical "head" coordinates
        pos_virtual = _find_centroid_sphere(pos["ch_pos"], group_names)
        # create the virtual channel info and set the position
        virtual_info = create_info([f"virtual {k + 1}"], inst.info["sfreq"], "eeg")
        virtual_info["chs"][0]["loc"][:3] = pos_virtual
        # create virtual channel
        data = inst.get_data(picks=group_names)
        if isinstance(inst, BaseRaw):
            data = np.average(data, axis=0).reshape(1, -1)
            virtual_ch = RawArray(data, virtual_info, first_samp=inst.first_samp)
        elif isinstance(inst, BaseEpochs):
            data = np.average(data, axis=1).reshape(len(data), 1, -1)
            virtual_ch = EpochsArray(data, virtual_info, tmin=inst.tmin)
        else:  # evoked
            data = np.average(data, axis=0).reshape(1, -1)
            virtual_ch = EvokedArray(
                np.average(data, axis=0).reshape(1, -1),
                virtual_info,
                tmin=inst.tmin,
                nave=inst.nave,
                kind=inst.kind,
            )
        virtual_chs[f"virtual {k + 1}"] = virtual_ch

    # add the virtual channels
    inst.add_channels(list(virtual_chs.values()), force_update_info=True)

    # use the virtual channels to interpolate
    inst.info["bads"] = list(bads)
    inst.interpolate_bads()

    # drop virtual channels
    inst.drop_channels(list(virtual_chs.keys()))

    inst.info["bads"] = bads_orig
    return inst


def _find_centroid_sphere(ch_pos, group_names):
    """Compute the centroid position between N electrodes.

    The centroid should be determined in spherical "head" coordinates which is
    more accurante than cutting through the scalp by averaging in cartesian
    coordinates.

    A simple way is to average the location in cartesian coordinate, convert
    to spehrical coordinate and replace the radius with the average radius of
    the N points in spherical coordinates.

    Parameters
    ----------
    ch_pos : OrderedDict
        The position of all channels in cartesian coordinates.
    group_names : list | tuple
        The name of the N electrodes used to determine the centroid.

    Returns
    -------
    pos_centroid : array of shape (3,)
        The position of the centroid in cartesian coordinates.
    """
    cartesian_positions = np.array([ch_pos[ch_name] for ch_name in group_names])
    sphere_positions = _cart_to_sph(cartesian_positions)
    cartesian_pos_centroid = np.average(cartesian_positions, axis=0)
    sphere_pos_centroid = _cart_to_sph(cartesian_pos_centroid)
    # average the radius and overwrite it
    avg_radius = np.average(sphere_positions, axis=0)[0]
    sphere_pos_centroid[0, 0] = avg_radius
    # convert back to cartesian
    pos_centroid = _sph_to_cart(sphere_pos_centroid)[0, :]
    return pos_centroid
