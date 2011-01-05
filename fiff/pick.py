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
        List of channels to include. If None include all available.

    exclude : list of string
        List of channels to exclude. If None do not exclude any channel.

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    sel = []
    for k, name in enumerate(ch_names):
        if (include is None or name in include) and name not in exclude:
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

    meg : bool
        Is True include MEG channels

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
        if (kind == FIFF.FIFFV_MEG_CH or kind == FIFF.FIFFV_REF_MEG_CH) \
                                                                    and meg:
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
