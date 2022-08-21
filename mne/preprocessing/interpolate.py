"""Tools for data interpolation."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>

from itertools import chain
import numpy as np

from ..utils import _validate_type
from ..io import BaseRaw, RawArray
from ..io.meas_info import create_info
from ..epochs import BaseEpochs, EpochsArray
from ..evoked import Evoked, EvokedArray
from ..transforms import _sph_to_cart, _cart_to_sph


def equalize_bads(insts, interp_thresh=1., copy=True):
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
        raise ValueError('interp_thresh must be between 0 and 1, got %s'
                         % (interp_thresh,))

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
                 if good_times[k] < interp_thresh]
    if copy:
        insts = [inst.copy() for inst in insts]

    for inst in insts:
        if len(set(inst.info['bads']) - set(bads_keep)):
            inst.interpolate_bads(exclude=bads_keep)
        inst.info['bads'] = bads_keep

    return insts


def interpolate_bridged_electrodes(inst, bridged_idx):
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

    Returns
    -------
    inst : instance of Epochs, Evoked, or Raw
        The modified data object.

    See Also
    --------
    mne.preprocessing.compute_bridged_electrodes
    """
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked))
    montage = inst.get_montage()
    if montage is None:
        raise RuntimeError('No channel positions found in ``inst``')
    pos = montage.get_positions()
    if pos['coord_frame'] != 'head':
        raise RuntimeError('Montage channel positions must be in ``head``'
                           'got {}'.format(pos['coord_frame']))
    ch_pos = pos['ch_pos']
    # store bads orig to put back at the end
    bads_orig = inst.info['bads']
    inst.info['bads'] = list()

    # make virtual channels
    virtual_chs = dict()
    bads = set()
    data = inst.get_data()
    for idx0, idx1 in bridged_idx:
        ch0 = inst.ch_names[idx0]
        ch1 = inst.ch_names[idx1]
        bads = bads.union([ch0, ch1])
        # compute midway position in spherical coordinates in "head"
        # (more accurate than cutting though the scalp by using cartesian)
        pos0 = _cart_to_sph(ch_pos[ch0])
        pos1 = _cart_to_sph(ch_pos[ch1])
        pos_virtual = _sph_to_cart((pos0 + pos1) / 2)
        virtual_info = create_info(
            [f'{ch0}-{ch1} virtual'], inst.info['sfreq'], 'eeg')
        virtual_info['chs'][0]['loc'][:3] = pos_virtual
        if isinstance(inst, BaseRaw):
            virtual_ch = RawArray(
                (data[idx0:idx0 + 1] + data[idx1:idx1 + 1]) / 2,
                virtual_info, first_samp=inst.first_samp)
        elif isinstance(inst, BaseEpochs):
            virtual_ch = EpochsArray(
                (data[:, idx0:idx0 + 1] + data[:, idx1:idx1 + 1]) / 2,
                virtual_info, tmin=inst.tmin)
        else:  # evoked
            virtual_ch = EvokedArray(
                (data[idx0:idx0 + 1] + data[idx1:idx1 + 1]) / 2,
                virtual_info, tmin=inst.tmin, nave=inst.nave, kind=inst.kind)
        virtual_chs[f'{ch0}-{ch1} virtual'] = virtual_ch

    # add the virtual channels
    inst.add_channels(list(virtual_chs.values()), force_update_info=True)

    # use the virtual channels to interpolate
    inst.info['bads'] = list(bads)
    inst.interpolate_bads()

    # drop virtual channels
    inst.drop_channels(list(virtual_chs.keys()))

    inst.info['bads'] = bads_orig
    return inst
