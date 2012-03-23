# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from copy import deepcopy
import re

import numpy as np
from .constants import FIFF


def channel_type(info, idx):
    """Get channel type

    Parameters
    ----------
    info : dict
        Measurement info
    idx : int
        Index of channel

    Returns
    -------
    type : 'grad' | 'mag' | 'eeg' | 'stim' | 'eog' | 'emg' | 'ecg'
        Type of channel
    """

    kind = info['chs'][idx]['kind']
    if kind == FIFF.FIFFV_MEG_CH:
        if info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_T_M:
            return 'grad'
        elif info['chs'][idx]['unit'] == FIFF.FIFF_UNIT_T:
            return 'mag'
    elif kind == FIFF.FIFFV_REF_MEG_CH:
        return 'ref_meg'
    elif kind == FIFF.FIFFV_EEG_CH:
        return 'eeg'
    elif kind == FIFF.FIFFV_STIM_CH:
        return 'stim'
    elif kind == FIFF.FIFFV_EOG_CH:
        return 'eog'
    elif kind == FIFF.FIFFV_EMG_CH:
        return 'emg'
    elif kind == FIFF.FIFFV_ECG_CH:
        return 'ecg'
    elif kind == FIFF.FIFFV_MISC_CH:
        return 'misc'
    raise Exception('Unknown channel type')


def pick_channels(ch_names, include, exclude=[]):
    """Pick channels by names

    Returns the indices of the good channels in ch_names.

    Parameters
    ----------
    ch_names : list of string
        List of channels

    include : list of string
        List of channels to include. If empty include all available.

    exclude : list of string
        List of channels to exclude. If empty do not exclude any channel.

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    sel = []
    for k, name in enumerate(ch_names):
        if (len(include) == 0 or name in include) and name not in exclude:
            sel.append(k)
    sel = np.unique(sel)
    np.sort(sel)
    return sel


def pick_channels_regexp(ch_names, regexp):
    """Pick channels using regular expression

    Returns the indices of the good channels in ch_names.

    Parameters
    ----------
    ch_names : list of string
        List of channels

    regexp : string
        The regular expression. See python standard module for regular
        expressions.

    Returns
    -------
    sel : array of int
        Indices of good channels.

    Examples
    --------
    >>> pick_channels_regexp(['MEG 2331', 'MEG 2332', 'MEG 2333'], 'MEG ...1')
    [0]
    >>> pick_channels_regexp(['MEG 2331', 'MEG 2332', 'MEG 2333'], 'MEG *')
    [0, 1, 2]
    """
    r = re.compile(regexp)
    return [k for k, name in enumerate(ch_names) if r.match(name)]


def pick_types(info, meg=True, eeg=False, stim=False, eog=False, ecg=False,
               emg=False, misc=False, include=[], exclude=[]):
    """Pick channels by type and names

    Parameters
    ----------
    info : dict
        The measurement info
    meg : bool or string
        If True include all MEG channels. If False include None
        If string it can be 'mag' or 'grad' to select only gradiometers
        or magnetometers. It can also be 'ref_meg' to get CTF
        reference channels.
    eeg : bool
        If True include EEG channels
    eog : bool
        If True include EOG channels
    ecg : bool
        If True include ECG channels
    emg : bool
        If True include EMG channels
    stim : bool
        If True include stimulus channels
    misc : bool
        If True include miscellaneous analog channels
    include : list of string
        List of additional channels to include. If empty do not include any.

    exclude : list of string
        List of channels to exclude. If empty do not include any.

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    nchan = info['nchan']
    pick = np.zeros(nchan, dtype=np.bool)

    for k in range(nchan):
        kind = info['chs'][k]['kind']
        if (kind == FIFF.FIFFV_MEG_CH or kind == FIFF.FIFFV_REF_MEG_CH):
            if meg == True:
                pick[k] = True
            elif (meg == 'grad'
                    and info['chs'][k]['unit'] == FIFF.FIFF_UNIT_T_M):
                pick[k] = True
            elif (meg == 'mag'
                    and info['chs'][k]['unit'] == FIFF.FIFF_UNIT_T):
                pick[k] = True
            elif meg == 'ref_meg':
                pick[k] = True
        elif kind == FIFF.FIFFV_EEG_CH and eeg:
            pick[k] = True
        elif kind == FIFF.FIFFV_STIM_CH and stim:
            pick[k] = True
        elif kind == FIFF.FIFFV_EOG_CH and eog:
            pick[k] = True
        elif kind == FIFF.FIFFV_ECG_CH and ecg:
            pick[k] = True
        elif kind == FIFF.FIFFV_EMG_CH and emg:
            pick[k] = True
        elif kind == FIFF.FIFFV_MISC_CH and misc:
            pick[k] = True

    myinclude = [info['ch_names'][k] for k in range(nchan) if pick[k]]
    myinclude += include

    if len(myinclude) == 0:
        sel = []
    else:
        sel = pick_channels(info['ch_names'], myinclude, exclude)

    return sel


def pick_info(info, sel=[]):
    """Restrict an info structure to a selection of channels

    Parameters
    ----------
    info : dict
        Info structure from evoked or raw data
    sel : list of int
        Indices of channels to include

    Returns
    -------
    res : dict
        Info structure restricted to a selection of channels
    """

    res = deepcopy(info)
    if len(sel) == 0:
        raise ValueError('Warning : No channels match the selection.')

    res['chs'] = [res['chs'][k] for k in sel]
    res['ch_names'] = [res['ch_names'][k] for k in sel]
    res['nchan'] = len(sel)
    return res


def pick_channels_evoked(orig, include=[], exclude=[]):
    """Pick channels from evoked data

    Parameters
    ----------
    orig : Evoked object
        One evoked dataset

    include : list of string, (optional)
        List of channels to include. (if None, include all available)

    exclude : list of string, (optional)
        Channels to exclude (if None, do not exclude any)

    Returns
    -------
    res : instance of Evoked
        Evoked data restricted to selected channels. If include and
        exclude are None it returns orig without copy.
    """

    if len(include) == 0 and len(exclude) == 0:
        return orig

    sel = pick_channels(orig.info['ch_names'], include=include,
                        exclude=exclude)

    if len(sel) == 0:
        raise ValueError('Warning : No channels match the selection.')

    res = deepcopy(orig)
    #
    #   Modify the measurement info
    #
    res.info = pick_info(res.info, sel)
    #
    #   Create the reduced data set
    #
    res.data = res.data[sel, :]

    return res


def pick_types_evoked(orig, meg=True, eeg=False, stim=False, eog=False,
                      ecg=False, emg=False, misc=False, include=[],
                      exclude=[]):
    """Pick by channel type and names from evoked data

    Parameters
    ----------
    info : dict
        The measurement info
    meg : bool or string
        If True include all MEG channels. If False include None
        If string it can be 'mag' or 'grad' to select only gradiometers
        or magnetometers. It can also be 'ref_meg' to get CTF
        reference channels.
    eeg : bool
        If True include EEG channels
    eog : bool
        If True include EOG channels
    ecg : bool
        If True include ECG channels
    emg : bool
        If True include EMG channels
    stim : bool
        If True include stimulus channels
    misc : bool
        If True include miscellaneous analog channels
    include : list of string
        List of additional channels to include. If empty do not include any.

    exclude : list of string
        List of channels to exclude. If empty do not include any.

    Returns
    -------
    res : instance of Evoked
        Evoked data restricted to selected channels. If include and
        exclude are None it returns orig without copy.
    """
    sel = pick_types(orig.info, meg, eeg, stim, eog, ecg, emg, misc, include,
                     exclude)
    include_ch_names = [orig.ch_names[k] for k in sel]
    return pick_channels_evoked(orig, include_ch_names)


def pick_channels_forward(orig, include=[], exclude=[]):
    """Pick channels from forward operator

    Parameters
    ----------
    orig : dict
        A forward solution

    include : list of string, (optional)
        List of channels to include. (if None, include all available)

    exclude : list of string, (optional)
        Channels to exclude (if None, do not exclude any)

    Returns
    -------
    res : dict
        Forward solution restricted to selected channels. If include and
        exclude are None it returns orig without copy.
    """

    if len(include) == 0 and len(exclude) == 0:
        return orig

    sel = pick_channels(orig['sol']['row_names'], include=include,
                        exclude=exclude)

    fwd = deepcopy(orig)

    #   Do we have something?
    nuse = len(sel)
    if nuse == 0:
        raise ValueError('Nothing remains after picking')

    print '    %d out of %d channels remain after picking' % (nuse,
                                                            fwd['nchan'])

    #   Pick the correct rows of the forward operator
    fwd['nchan'] = nuse
    fwd['sol']['data'] = fwd['sol']['data'][sel, :]
    fwd['sol']['nrow'] = nuse
    fwd['sol']['row_names'] = [fwd['sol']['row_names'][k] for k in sel]
    fwd['chs'] = [fwd['chs'][k] for k in sel]

    if fwd['sol_grad'] is not None:
        fwd['sol_grad']['data'] = fwd['sol_grad']['data'][sel, :]
        fwd['sol_grad']['nrow'] = nuse
        fwd['sol_grad']['row_names'] = [fwd['sol_grad']['row_names'][k]
                                        for k in sel]

    return fwd


def pick_types_forward(orig, meg=True, eeg=False, include=[], exclude=[]):
    """Pick by channel type and names from a forward operator

    Parameters
    ----------
    orig : dict
        A forward solution
    meg : bool or string
        If True include all MEG channels. If False include None
        If string it can be 'mag' or 'grad' to select only gradiometers
        or magnetometers. It can also be 'ref_meg' to get CTF
        reference channels.
    eeg : bool
        If True include EEG channels
    include : list of string
        List of additional channels to include. If empty do not include any.

    exclude : list of string
        List of channels to exclude. If empty do not include any.

    Returns
    -------
    res : dict
        Forward solution restricted to selected channel types.
    """
    info = {'ch_names': orig['sol']['row_names'], 'chs': orig['chs'],
            'nchan': orig['nchan']}
    sel = pick_types(info, meg, eeg, include=include, exclude=exclude)
    include_ch_names = [info['ch_names'][k] for k in sel]

    return pick_channels_forward(orig, include_ch_names)


def channel_indices_by_type(info):
    """Get indices of channels by type
    """
    idx = dict(grad=[], mag=[], eeg=[], eog=[], ecg=[])
    for k, ch in enumerate(info['chs']):
        for key in idx.keys():
            if channel_type(info, k) == key:
                idx[key].append(k)

    return idx
