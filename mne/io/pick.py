# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from copy import deepcopy
import re

import numpy as np

from .constants import FIFF
from ..utils import logger, verbose
from ..externals.six import string_types


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
           'ref_meg' | 'resp' | 'exci' | 'ias' | 'syst' | 'misc'
           'seeg' | 'chpi'
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
    elif kind == FIFF.FIFFV_RESP_CH:
        return 'resp'
    elif kind == FIFF.FIFFV_MISC_CH:
        return 'misc'
    elif kind == FIFF.FIFFV_EXCI_CH:
        return 'exci'
    elif kind == FIFF.FIFFV_IAS_CH:
        return 'ias'
    elif kind == FIFF.FIFFV_SYST_CH:
        return 'syst'
    elif kind == FIFF.FIFFV_SEEG_CH:
        return 'seeg'
    elif kind in [FIFF.FIFFV_QUAT_0, FIFF.FIFFV_QUAT_1, FIFF.FIFFV_QUAT_2,
                  FIFF.FIFFV_QUAT_3, FIFF.FIFFV_QUAT_4, FIFF.FIFFV_QUAT_5,
                  FIFF.FIFFV_QUAT_6, FIFF.FIFFV_HPI_G, FIFF.FIFFV_HPI_ERR,
                  FIFF.FIFFV_HPI_MOV]:
        return 'chpi'  # channels relative to head position monitoring
    raise Exception('Unknown channel type')


def pick_channels(ch_names, include, exclude=[]):
    """Pick channels by names

    Returns the indices of the good channels in ch_names.

    Parameters
    ----------
    ch_names : list of string
        List of channels.
    include : list of string
        List of channels to include (if empty include all available).
    exclude : list of string
        List of channels to exclude (if empty do not exclude any channel).
        Defaults to [].

    See Also
    --------
    pick_channels_regexp, pick_types

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    if len(np.unique(ch_names)) != len(ch_names):
        raise RuntimeError('ch_names is not a unique list, picking is unsafe')
    _check_excludes_includes(include)
    _check_excludes_includes(exclude)
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

    See Also
    --------
    pick_channels

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
               emg=False, ref_meg='auto', misc=False, resp=False, chpi=False,
               exci=False, ias=False, syst=False, seeg=False,
               include=[], exclude='bads', selection=None):
    """Pick channels by type and names

    Parameters
    ----------
    info : dict
        The measurement info.
    meg : bool or string
        If True include all MEG channels. If False include None
        If string it can be 'mag', 'grad', 'planar1' or 'planar2' to select
        only magnetometers, all gradiometers, or a specific type of
        gradiometer.
    eeg : bool
        If True include EEG channels.
    stim : bool
        If True include stimulus channels.
    eog : bool
        If True include EOG channels.
    ecg : bool
        If True include ECG channels.
    emg : bool
        If True include EMG channels.
    ref_meg: bool | str
        If True include CTF / 4D reference channels. If 'auto', the reference
        channels are only included if compensations are present.
    misc : bool
        If True include miscellaneous analog channels.
    resp : bool
        If True include response-trigger channel. For some MEG systems this
        is separate from the stim channel.
    chpi : bool
        If True include continuous HPI coil channels.
    exci : bool
        Flux excitation channel used to be a stimulus channel.
    ias : bool
        Internal Active Shielding data (maybe on Triux only).
    syst : bool
        System status channel information (on Triux systems only).
    seeg : bool
        Stereotactic EEG channels
    include : list of string
        List of additional channels to include. If empty do not include any.
    exclude : list of string | str
        List of channels to exclude. If 'bads' (default), exclude channels
        in info['bads'].
    selection : list of string
        Restrict sensor channels (MEG, EEG) to this list of channel names.

    Returns
    -------
    sel : array of int
        Indices of good channels.
    """
    # NOTE: Changes to this function's signature should also be changed in
    # PickChannelsMixin
    from .meas_info import Info
    if not isinstance(info, Info):
        raise TypeError('info must be an instance of Info, not %s'
                        % type(info))
    nchan = info['nchan']
    pick = np.zeros(nchan, dtype=np.bool)

    if exclude is None:
        raise ValueError('exclude must be a list of strings or "bads"')
    elif exclude == 'bads':
        exclude = info.get('bads', [])
    elif not isinstance(exclude, (list, tuple)):
        raise ValueError('exclude must either be "bads" or a list of strings.'
                         ' If only one channel is to be excluded, use '
                         '[ch_name] instead of passing ch_name.')

    if isinstance(ref_meg, string_types):
        if ref_meg != 'auto':
            raise ValueError('ref_meg has to be either a bool or \'auto\'')

        ref_meg = ('comps' in info and info['comps'] is not None and
                   len(info['comps']) > 0)

    for k in range(nchan):
        kind = info['chs'][k]['kind']
        if kind == FIFF.FIFFV_MEG_CH:
            if meg is True:
                pick[k] = True
            elif info['chs'][k]['unit'] == FIFF.FIFF_UNIT_T_M:
                if meg == 'grad':
                    pick[k] = True
                elif meg == 'planar1' and info['ch_names'][k].endswith('2'):
                    pick[k] = True
                elif meg == 'planar2' and info['ch_names'][k].endswith('3'):
                    pick[k] = True
            elif (meg == 'mag' and
                  info['chs'][k]['unit'] == FIFF.FIFF_UNIT_T):
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
        elif kind == FIFF.FIFFV_REF_MEG_CH and ref_meg:
            pick[k] = True
        elif kind == FIFF.FIFFV_RESP_CH and resp:
            pick[k] = True
        elif kind == FIFF.FIFFV_SYST_CH and syst:
            pick[k] = True
        elif kind == FIFF.FIFFV_SEEG_CH and seeg:
            pick[k] = True
        elif kind == FIFF.FIFFV_IAS_CH and ias:
            pick[k] = True
        elif kind == FIFF.FIFFV_EXCI_CH and exci:
            pick[k] = True
        elif kind in [FIFF.FIFFV_QUAT_0, FIFF.FIFFV_QUAT_1, FIFF.FIFFV_QUAT_2,
                      FIFF.FIFFV_QUAT_3, FIFF.FIFFV_QUAT_4, FIFF.FIFFV_QUAT_5,
                      FIFF.FIFFV_QUAT_6, FIFF.FIFFV_HPI_G, FIFF.FIFFV_HPI_ERR,
                      FIFF.FIFFV_HPI_MOV] and chpi:
            pick[k] = True

    # restrict channels to selection if provided
    if selection is not None:
        # the selection only restricts these types of channels
        sel_kind = [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH,
                    FIFF.FIFFV_EEG_CH]
        for k in np.where(pick == True)[0]:  # noqa
            if (info['chs'][k]['kind'] in sel_kind and
                    info['ch_names'][k] not in selection):
                pick[k] = False

    myinclude = [info['ch_names'][k] for k in range(nchan) if pick[k]]
    myinclude += include

    if len(myinclude) == 0:
        sel = []
    else:
        sel = pick_channels(info['ch_names'], myinclude, exclude)

    return sel


def pick_info(info, sel=[], copy=True):
    """Restrict an info structure to a selection of channels

    Parameters
    ----------
    info : dict
        Info structure from evoked or raw data.
    sel : list of int
        Indices of channels to include.
    copy : bool
        If copy is False, info is modified inplace.

    Returns
    -------
    res : dict
        Info structure restricted to a selection of channels.
    """
    if copy:
        info = deepcopy(info)

    if len(sel) == 0:
        raise ValueError('No channels match the selection.')

    info['chs'] = [info['chs'][k] for k in sel]
    info['ch_names'] = [info['ch_names'][k] for k in sel]
    info['nchan'] = len(sel)

    # Check if bads_channels are included, otherwise
    # remove info['bads']
    info['bads'] = [ch for ch in info['bads'] if ch in info['ch_names']]

    return info


def _has_kit_refs(info, picks):
    """Helper to determine if KIT ref channels are chosen

    This is currently only used by make_forward_solution, which cannot
    run when KIT reference channels are included.
    """
    for p in picks:
        if info['chs'][p]['coil_type'] == FIFF.FIFFV_COIL_KIT_REF_MAG:
            return True
    return False


def pick_channels_evoked(orig, include=[], exclude='bads'):
    """Pick channels from evoked data

    Parameters
    ----------
    orig : Evoked object
        One evoked dataset.
    include : list of string, (optional)
        List of channels to include (if empty, include all available).
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in orig.info['bads']. Defaults to 'bads'.

    Returns
    -------
    res : instance of Evoked
        Evoked data restricted to selected channels. If include and
        exclude are empty it returns orig without copy.
    """
    if len(include) == 0 and len(exclude) == 0:
        return orig

    exclude = _check_excludes_includes(exclude, info=orig.info,
                                       allow_bads=True)
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


@verbose
def pick_channels_forward(orig, include=[], exclude=[], verbose=None):
    """Pick channels from forward operator

    Parameters
    ----------
    orig : dict
        A forward solution.
    include : list of string
        List of channels to include (if empty, include all available).
        Defaults to [].
    exclude : list of string | 'bads'
        Channels to exclude (if empty, do not exclude any). Defaults to [].
        If 'bads', then exclude bad channels in orig.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    res : dict
        Forward solution restricted to selected channels. If include and
        exclude are empty it returns orig without copy.
    """
    if len(include) == 0 and len(exclude) == 0:
        return orig
    exclude = _check_excludes_includes(exclude,
                                       info=orig['info'], allow_bads=True)

    # Allow for possibility of channel ordering in forward solution being
    # different from that of the M/EEG file it is based on.
    sel_sol = pick_channels(orig['sol']['row_names'], include=include,
                            exclude=exclude)
    sel_info = pick_channels(orig['info']['ch_names'], include=include,
                             exclude=exclude)

    fwd = deepcopy(orig)

    # Check that forward solution and original data file agree on #channels
    if len(sel_sol) != len(sel_info):
        raise ValueError('Forward solution and functional data appear to '
                         'have different channel names, please check.')

    #   Do we have something?
    nuse = len(sel_sol)
    if nuse == 0:
        raise ValueError('Nothing remains after picking')

    logger.info('    %d out of %d channels remain after picking'
                % (nuse, fwd['nchan']))

    #   Pick the correct rows of the forward operator using sel_sol
    fwd['sol']['data'] = fwd['sol']['data'][sel_sol, :]
    fwd['_orig_sol'] = fwd['_orig_sol'][sel_sol, :]
    fwd['sol']['nrow'] = nuse

    ch_names = [fwd['sol']['row_names'][k] for k in sel_sol]
    fwd['nchan'] = nuse
    fwd['sol']['row_names'] = ch_names

    # Pick the appropriate channel names from the info-dict using sel_info
    fwd['info']['ch_names'] = [fwd['info']['ch_names'][k] for k in sel_info]
    fwd['info']['chs'] = [fwd['info']['chs'][k] for k in sel_info]
    fwd['info']['nchan'] = nuse
    fwd['info']['bads'] = [b for b in fwd['info']['bads'] if b in ch_names]

    if fwd['sol_grad'] is not None:
        fwd['sol_grad']['data'] = fwd['sol_grad']['data'][sel_sol, :]
        fwd['_orig_sol_grad'] = fwd['_orig_sol_grad'][sel_sol, :]
        fwd['sol_grad']['nrow'] = nuse
        fwd['sol_grad']['row_names'] = [fwd['sol_grad']['row_names'][k]
                                        for k in sel_sol]

    return fwd


def pick_types_forward(orig, meg=True, eeg=False, ref_meg=True, seeg=False,
                       include=[], exclude=[]):
    """Pick by channel type and names from a forward operator

    Parameters
    ----------
    orig : dict
        A forward solution
    meg : bool or string
        If True include all MEG channels. If False include None
        If string it can be 'mag' or 'grad' to select only gradiometers
        or magnetometers.
    eeg : bool
        If True include EEG channels
    ref_meg : bool
        If True include CTF / 4D reference channels
    seeg : bool
        If True include stereotactic EEG channels
    include : list of string
        List of additional channels to include. If empty do not include any.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any (default).
        If 'bads', exclude channels in orig['info']['bads'].

    Returns
    -------
    res : dict
        Forward solution restricted to selected channel types.
    """
    info = orig['info']
    sel = pick_types(info, meg, eeg, ref_meg=ref_meg, seeg=seeg,
                     include=include, exclude=exclude)
    if len(sel) == 0:
        raise ValueError('No valid channels found')
    include_ch_names = [info['ch_names'][k] for k in sel]
    return pick_channels_forward(orig, include_ch_names)


def channel_indices_by_type(info):
    """Get indices of channels by type
    """
    idx = dict(grad=[], mag=[], eeg=[], seeg=[], eog=[], ecg=[], stim=[],
               emg=[], ref_meg=[], misc=[], resp=[], chpi=[], exci=[], ias=[],
               syst=[])
    for k, ch in enumerate(info['chs']):
        for key in idx.keys():
            if channel_type(info, k) == key:
                idx[key].append(k)

    return idx


def pick_channels_cov(orig, include=[], exclude='bads'):
    """Pick channels from covariance matrix

    Parameters
    ----------
    orig : Covariance
        A covariance.
    include : list of string, (optional)
        List of channels to include (if empty, include all available).
    exclude : list of string, (optional) | 'bads'
        Channels to exclude (if empty, do not exclude any). Defaults to 'bads'.

    Returns
    -------
    res : dict
        Covariance solution restricted to selected channels.
    """
    exclude = orig['bads'] if exclude == 'bads' else exclude
    sel = pick_channels(orig['names'], include=include, exclude=exclude)
    res = deepcopy(orig)
    res['dim'] = len(sel)
    if not res['diag']:
        res['data'] = orig['data'][sel][:, sel]
    else:
        res['data'] = orig['data'][sel]
    res['names'] = [orig['names'][k] for k in sel]
    res['bads'] = [name for name in orig['bads'] if name in res['names']]
    res['eig'] = None
    res['eigvec'] = None
    return res


def _picks_by_type(info, meg_combined=False, ref_meg=False):
    """Get data channel indices as separate list of tuples
    Parameters
    ----------
    info : instance of mne.measuerment_info.Info
        The info.
    meg_combined : bool
        Whether to return combined picks for grad and mag.
    ref_meg : bool
        If True include CTF / 4D reference channels

    Returns
    -------
    picks_list : list of tuples
        The list of tuples of picks and the type string.
    """
    from ..channels.channels import _contains_ch_type
    picks_list = []
    has_mag, has_grad, has_eeg = [_contains_ch_type(info, k)
                                  for k in ('mag', 'grad', 'eeg')]
    if has_mag and (meg_combined is not True or not has_grad):
        picks_list.append(
            ('mag', pick_types(info, meg='mag', eeg=False, stim=False,
             ref_meg=ref_meg))
        )
    if has_grad and (meg_combined is not True or not has_mag):
        picks_list.append(
            ('grad', pick_types(info, meg='grad', eeg=False, stim=False,
             ref_meg=ref_meg))
        )
    if has_mag and has_grad and meg_combined is True:
        picks_list.append(
            ('meg', pick_types(info, meg=True, eeg=False, stim=False,
             ref_meg=ref_meg))
        )
    if has_eeg:
        picks_list.append(
            ('eeg', pick_types(info, meg=False, eeg=True, stim=False,
             ref_meg=ref_meg))
        )
    return picks_list


def _check_excludes_includes(chs, info=None, allow_bads=False):
    """Ensure that inputs to exclude/include are list-like or "bads".

    Parameters
    ----------
    chs : any input, should be list, tuple, string
        The channels passed to include or exclude.
    allow_strs : list of strings
        Optional strings to allow.

    Returns
    -------
    chs : list
        Channels to be excluded/excluded. If allow_bads, and chs=="bads",
        this will be the bad channels found in 'info'.
    """
    from .meas_info import Info
    if not isinstance(chs, (list, tuple, np.ndarray)):
        if allow_bads is True:
            if not isinstance(info, Info):
                raise ValueError('Supply an info object if allow_bads is true')
            elif chs != 'bads':
                raise ValueError('If chs is a string, it must be "bads"')
            else:
                chs = info['bads']
        else:
            raise ValueError(
                'include/exclude must be list, tuple, ndarray, or "bads". ' +
                'You provided type {0}'.format(type(chs)))
    return chs
