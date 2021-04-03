"""Tools for data interpolation."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>

from itertools import chain
import numpy as np

from ..epochs import BaseEpochs


def interpolate_mark_bads(insts, good_fraction=1., copy=True):
    """Interpolate or mark bads consistently for a list of instances.

    Once called on a list of instances, the instances can be concatenated
    as they will have the same list of bad channels.

    Parameters
    ----------
    insts : list
        The list of instances (Evoked, Epochs or Raw) to consider
        for interpolation. Each instance should have marked channels.
    good_fraction : float
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
        The instances will consistent list of bad channels after
        potential interpolation.
    """
    all_bads = list(
        set(chain.from_iterable([inst.info['bads'] for inst in insts]))
    )
    if isinstance(insts[0], BaseEpochs):
        durations = [len(inst) * len(inst.times) for inst in insts]
    else:
        durations = [len(inst.times) for inst in insts]

    good_times = []
    for ch_name in all_bads:
        good_times.append(sum(
            durations[k] for k, inst in enumerate(insts)
            if ch_name not in inst.info['bads']
        ) / np.sum(durations))

    bads_keep = [ch for k, ch in enumerate(all_bads)
                 if good_times[k] < good_fraction]
    if copy:
        insts = [inst.copy() for inst in insts]

    for inst in insts:
        if len(set(inst.info['bads']) - set(bads_keep)):
            inst.interpolate_bads(exclude=bads_keep)
        inst.info['bads'] = bads_keep

    return insts
