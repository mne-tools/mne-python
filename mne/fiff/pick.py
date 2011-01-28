# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from copy import copy

import numpy as np
from .constants import FIFF


def pick_channels(ch_names, include, exclude):
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
        if (include is [] or name in include) and name not in exclude:
            sel.append(k)
    sel = np.unique(sel)
    np.sort(sel)
    return sel


def pick_types(info, meg=True, eeg=False, stim=False, include=[], exclude=[]):
    """Pick channels

    Parameters
    ----------
    info : dict
        The measurement info

    meg : bool or string
        Is True include MEG channels or False include None
        If string it can be 'mag' or 'grad' to select only gradiometers
        or magnetometers.

    eeg : bool
        Is True include EEG channels

    stim : bool
        Is True include stimulus channels

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
        kind = info['chs'][k].kind
        if (kind == FIFF.FIFFV_MEG_CH or kind == FIFF.FIFFV_REF_MEG_CH):
            if meg == True:
                pick[k] = True
            elif (meg is 'grad'
                    and info['chs'][k]['unit'] == FIFF.FIFF_UNIT_T_M):
                pick[k] = True
            elif (meg is 'mag'
                    and info['chs'][k]['unit'] == FIFF.FIFF_UNIT_T):
                pick[k] = True
        elif kind == FIFF.FIFFV_EEG_CH and eeg:
            pick[k] = True
        elif kind == FIFF.FIFFV_STIM_CH and stim:
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

    res = copy(info)
    if len(sel) == 0:
        raise ValueError, 'Warning : No channels match the selection.'

    res['chs'] = [res['chs'][k] for k in sel]
    res['ch_names'] = [res['ch_names'][k] for k in sel]
    res['nchan'] = len(sel)
    return res


def pick_channels_evoked(orig, include=[], exclude=[]):
    """Pick channels from evoked data

    Parameters
    ----------
    orig : dict
        One evoked data

    include : list of string, (optional)
        List of channels to include. (if None, include all available)

    exclude : list of string, (optional)
        Channels to exclude (if None, do not exclude any)

    Returns
    -------
    res : dict
        Evoked data restricted to selected channels. If include and
        exclude are None it returns orig without copy.
    """

    if include is None and exclude is None:
        return orig

    sel = pick_channels(orig['info']['ch_names'], include=include,
                        exclude=exclude)

    if len(sel) == 0:
        raise ValueError, 'Warning : No channels match the selection.'

    res = orig.copy()
    #
    #   Modify the measurement info
    #
    res['info'] = pick_info(res['info'], sel)
    #
    #   Create the reduced data set
    #
    res['evoked']['epochs'] = res['evoked']['epochs'][sel,:]

    return res
