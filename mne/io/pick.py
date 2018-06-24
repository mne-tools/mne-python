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
from .compensator import get_current_comp


def get_channel_types():
    """Return all known channel types.

    Returns
    -------
    channel_types : dict
        The keys contain the channel types, and the values contain the
        corresponding values in the info['chs'][idx] dictionary.
    """
    return dict(grad=dict(kind=FIFF.FIFFV_MEG_CH,
                          unit=FIFF.FIFF_UNIT_T_M),
                mag=dict(kind=FIFF.FIFFV_MEG_CH,
                         unit=FIFF.FIFF_UNIT_T),
                ref_meg=dict(kind=FIFF.FIFFV_REF_MEG_CH),
                eeg=dict(kind=FIFF.FIFFV_EEG_CH),
                stim=dict(kind=FIFF.FIFFV_STIM_CH),
                eog=dict(kind=FIFF.FIFFV_EOG_CH),
                emg=dict(kind=FIFF.FIFFV_EMG_CH),
                ecg=dict(kind=FIFF.FIFFV_ECG_CH),
                resp=dict(kind=FIFF.FIFFV_RESP_CH),
                misc=dict(kind=FIFF.FIFFV_MISC_CH),
                exci=dict(kind=FIFF.FIFFV_EXCI_CH),
                ias=dict(kind=FIFF.FIFFV_IAS_CH),
                syst=dict(kind=FIFF.FIFFV_SYST_CH),
                seeg=dict(kind=FIFF.FIFFV_SEEG_CH),
                bio=dict(kind=FIFF.FIFFV_BIO_CH),
                chpi=dict(kind=[FIFF.FIFFV_QUAT_0, FIFF.FIFFV_QUAT_1,
                                FIFF.FIFFV_QUAT_2, FIFF.FIFFV_QUAT_3,
                                FIFF.FIFFV_QUAT_4, FIFF.FIFFV_QUAT_5,
                                FIFF.FIFFV_QUAT_6, FIFF.FIFFV_HPI_G,
                                FIFF.FIFFV_HPI_ERR, FIFF.FIFFV_HPI_MOV]),
                dipole=dict(kind=FIFF.FIFFV_DIPOLE_WAVE),
                gof=dict(kind=FIFF.FIFFV_GOODNESS_FIT),
                ecog=dict(kind=FIFF.FIFFV_ECOG_CH),
                hbo=dict(kind=FIFF.FIFFV_FNIRS_CH,
                         coil_type=FIFF.FIFFV_COIL_FNIRS_HBO),
                hbr=dict(kind=FIFF.FIFFV_FNIRS_CH,
                         coil_type=FIFF.FIFFV_COIL_FNIRS_HBR))


def channel_type(info, idx):
    """Get channel type.

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
           'seeg' | 'bio' | 'chpi' | 'dipole' | 'gof' | 'ecog' | 'hbo' | 'hbr'
        Type of channel
    """
    ch = info['chs'][idx]

    # iterate through all defined channel types until we find a match with ch
    for t, rules in get_channel_types().items():
        for key, vals in rules.items():  # all keys must match the values
            if ch.get(key, None) not in np.array(vals):
                break  # not channel type t, go to next iteration
        else:
            return t

    raise ValueError('Unknown channel type for {}'.format(ch["ch_name"]))


def pick_channels(ch_names, include, exclude=[]):
    """Pick channels by names.

    Returns the indices of the good channels in ch_names.

    Parameters
    ----------
    ch_names : list of string
        List of channels.
    include : list of string
        List of channels to include (if empty include all available).

        .. note:: This is to be treated as a set. The order of this list
           is not used or maintained in ``sel``.

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
    if not isinstance(include, set):
        include = set(include)
    if not isinstance(exclude, set):
        exclude = set(exclude)

    sel = []
    for k, name in enumerate(ch_names):
        if (len(include) == 0 or name in include) and name not in exclude:
            sel.append(k)
    return np.array(sel, int)


def pick_channels_regexp(ch_names, regexp):
    """Pick channels using regular expression.

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


def _triage_meg_pick(ch, meg):
    """Triage an MEG pick type."""
    if meg is True:
        return True
    elif ch['unit'] == FIFF.FIFF_UNIT_T_M:
        if meg == 'grad':
            return True
        elif meg == 'planar1' and ch['ch_name'].endswith('2'):
            return True
        elif meg == 'planar2' and ch['ch_name'].endswith('3'):
            return True
    elif (meg == 'mag' and ch['unit'] == FIFF.FIFF_UNIT_T):
        return True
    return False


def _triage_fnirs_pick(ch, fnirs):
    """Triage an fNIRS pick type."""
    if fnirs is True:
        return True
    elif ch['coil_type'] == FIFF.FIFFV_COIL_FNIRS_HBO and fnirs == 'hbo':
        return True
    elif ch['coil_type'] == FIFF.FIFFV_COIL_FNIRS_HBR and fnirs == 'hbr':
        return True
    return False


def _check_meg_type(meg, allow_auto=False):
    """Ensure a valid meg type."""
    if isinstance(meg, string_types):
        allowed_types = ['grad', 'mag', 'planar1', 'planar2']
        allowed_types += ['auto'] if allow_auto else []
        if meg not in allowed_types:
            raise ValueError('meg value must be one of %s or bool, not %s'
                             % (allowed_types, meg))


def pick_types(info, meg=True, eeg=False, stim=False, eog=False, ecg=False,
               emg=False, ref_meg='auto', misc=False, resp=False, chpi=False,
               exci=False, ias=False, syst=False, seeg=False, dipole=False,
               gof=False, bio=False, ecog=False, fnirs=False, include=(),
               exclude='bads', selection=None):
    """Pick channels by type and names.

    Parameters
    ----------
    info : dict
        The measurement info.
    meg : bool | str
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
        channels are only included if compensations are present. Can also be
        the string options from `meg`.
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
        Stereotactic EEG channels.
    dipole : bool
        Dipole time course channels.
    gof : bool
        Dipole goodness of fit channels.
    bio : bool
        Bio channels.
    ecog : bool
        Electrocorticography channels.
    fnirs : bool | str
        Functional near-infrared spectroscopy channels. If True include all
        fNIRS channels. If False (default) include none. If string it can be
        'hbo' (to include channels measuring oxyhemoglobin) or 'hbr' (to
        include channels measuring deoxyhemoglobin).
    include : list of string
        List of additional channels to include. If empty do not include any.
    exclude : list of string | str
        List of channels to exclude. If 'bads' (default), exclude channels
        in ``info['bads']``.
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
    info._check_consistency()
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

    _check_meg_type(ref_meg, allow_auto=True)
    _check_meg_type(meg)
    if isinstance(ref_meg, string_types) and ref_meg == 'auto':
        ref_meg = ('comps' in info and info['comps'] is not None and
                   len(info['comps']) > 0)

    for param in (eeg, stim, eog, ecg, emg, misc, resp, chpi, exci,
                  ias, syst, seeg, dipole, gof, bio, ecog):
        if not isinstance(param, bool):
            w = ('Parameters for all channel types (with the exception '
                 'of "meg", "ref_meg" and "fnirs") must be of type bool, '
                 'not {0}.')
            raise ValueError(w.format(type(param)))

    for k in range(nchan):
        kind = info['chs'][k]['kind']
        # XXX eventually we should de-duplicate this with channel_type!
        if kind == FIFF.FIFFV_MEG_CH and meg:
            pick[k] = _triage_meg_pick(info['chs'][k], meg)
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
            pick[k] = _triage_meg_pick(info['chs'][k], ref_meg)
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
        elif kind == FIFF.FIFFV_DIPOLE_WAVE and dipole:
            pick[k] = True
        elif kind == FIFF.FIFFV_GOODNESS_FIT and gof:
            pick[k] = True
        elif kind == FIFF.FIFFV_BIO_CH and bio:
            pick[k] = True
        elif kind == FIFF.FIFFV_ECOG_CH and ecog:
            pick[k] = True
        elif kind == FIFF.FIFFV_FNIRS_CH:
            pick[k] = _triage_fnirs_pick(info['chs'][k], fnirs)

    # restrict channels to selection if provided
    if selection is not None:
        # the selection only restricts these types of channels
        sel_kind = [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH,
                    FIFF.FIFFV_EEG_CH]
        for k in np.where(pick)[0]:
            if (info['chs'][k]['kind'] in sel_kind and
                    info['ch_names'][k] not in selection):
                pick[k] = False

    myinclude = [info['ch_names'][k] for k in range(nchan) if pick[k]]
    myinclude += include

    if len(myinclude) == 0:
        sel = np.array([], int)
    else:
        sel = pick_channels(info['ch_names'], myinclude, exclude)

    return sel


def pick_info(info, sel=(), copy=True):
    """Restrict an info structure to a selection of channels.

    Parameters
    ----------
    info : dict
        Info structure from evoked or raw data.
    sel : list of int | None
        Indices of channels to include.
    copy : bool
        If copy is False, info is modified inplace.

    Returns
    -------
    res : dict
        Info structure restricted to a selection of channels.
    """
    # avoid circular imports
    from .meas_info import _bad_chans_comp

    info._check_consistency()
    info = info.copy() if copy else info
    if sel is None:
        return info
    elif len(sel) == 0:
        raise ValueError('No channels match the selection.')

    # make sure required the compensation channels are present
    if len(info['comps']) > 0:
        ch_names = [info['ch_names'][idx] for idx in sel]
        _, comps_missing = _bad_chans_comp(info, ch_names)
        current_comp = get_current_comp(info)
        if len(comps_missing) > 0:
            if current_comp != 0:
                raise RuntimeError(
                    'Compensation grade %d has been applied, but '
                    'compensation channels are missing: %s\n'
                    'Either remove compensation or pick compensation '
                    'channels' % (current_comp, comps_missing))
            else:
                logger.info('Removing %d compensators from info because '
                            'not all compensation channels were picked'
                            % (len(info['comps']),))
                info['comps'] = []

    info['chs'] = [info['chs'][k] for k in sel]
    info._update_redundant()
    info['bads'] = [ch for ch in info['bads'] if ch in info['ch_names']]

    if 'comps' in info:
        comps = deepcopy(info['comps'])
        for c in comps:
            row_idx = [k for k, n in enumerate(c['data']['row_names'])
                       if n in info['ch_names']]
            row_names = [c['data']['row_names'][i] for i in row_idx]
            rowcals = c['rowcals'][row_idx]
            c['rowcals'] = rowcals
            c['data']['nrow'] = len(row_names)
            c['data']['row_names'] = row_names
            c['data']['data'] = c['data']['data'][row_idx]
        info['comps'] = comps
    info._check_consistency()
    return info


def _has_kit_refs(info, picks):
    """Determine if KIT ref channels are chosen.

    This is currently only used by make_forward_solution, which cannot
    run when KIT reference channels are included.
    """
    for p in picks:
        if info['chs'][p]['coil_type'] == FIFF.FIFFV_COIL_KIT_REF_MAG:
            return True
    return False


def pick_channels_evoked(orig, include=[], exclude='bads'):
    """Pick channels from evoked data.

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
    """Pick channels from forward operator.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    res : dict
        Forward solution restricted to selected channels. If include and
        exclude are empty it returns orig without copy.
    """
    orig['info']._check_consistency()
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
    fwd['info']['chs'] = [fwd['info']['chs'][k] for k in sel_info]
    fwd['info']._update_redundant()
    fwd['info']['bads'] = [b for b in fwd['info']['bads'] if b in ch_names]

    if fwd['sol_grad'] is not None:
        fwd['sol_grad']['data'] = fwd['sol_grad']['data'][sel_sol, :]
        fwd['_orig_sol_grad'] = fwd['_orig_sol_grad'][sel_sol, :]
        fwd['sol_grad']['nrow'] = nuse
        fwd['sol_grad']['row_names'] = [fwd['sol_grad']['row_names'][k]
                                        for k in sel_sol]

    return fwd


def pick_types_forward(orig, meg=True, eeg=False, ref_meg=True, seeg=False,
                       ecog=False, include=[], exclude=[]):
    """Pick by channel type and names from a forward operator.

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
    ecog : bool
        If True include electrocorticography channels
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
    sel = pick_types(info, meg, eeg, ref_meg=ref_meg, seeg=seeg, ecog=ecog,
                     include=include, exclude=exclude)
    if len(sel) == 0:
        raise ValueError('No valid channels found')
    include_ch_names = [info['ch_names'][k] for k in sel]

    return pick_channels_forward(orig, include_ch_names)


def channel_indices_by_type(info, picks=None):
    """Get indices of channels by type.

    Parameters
    ----------
    info : instance of mne.measuerment_info.Info
        The info.
    picks : None | list of int
        The indices of channels from which to get the type

    Returns
    -------
    idx_by_type : dict
        The dictionary that maps each channel type to the list of
        channel indidces.
    """
    idx_by_type = dict((key, list()) for key in _PICK_TYPES_KEYS if
                       key not in ('meg', 'fnirs'))
    idx_by_type.update(mag=list(), grad=list(), hbo=list(), hbr=list())
    if picks is None:
        picks = range(len(info["chs"]))
    for k in picks:
        ch_type = channel_type(info, k)
        for key in idx_by_type.keys():
            if ch_type == key:
                idx_by_type[key].append(k)
    return idx_by_type


def pick_channels_cov(orig, include=[], exclude='bads'):
    """Pick channels from covariance matrix.

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
    from ..cov import Covariance
    exclude = orig['bads'] if exclude == 'bads' else exclude
    sel = pick_channels(orig['names'], include=include, exclude=exclude)
    data = orig['data'][sel][:, sel] if not orig['diag'] else orig['data'][sel]
    names = [orig['names'][k] for k in sel]
    bads = [name for name in orig['bads'] if name in orig['names']]
    res = Covariance(
        data=data, names=names, bads=bads, projs=deepcopy(orig['projs']),
        nfree=orig['nfree'], eig=None, eigvec=None,
        method=orig.get('method', None), loglik=orig.get('loglik', None))
    return res


def _picks_by_type(info, meg_combined=False, ref_meg=False, exclude='bads'):
    """Get data channel indices as separate list of tuples.

    Parameters
    ----------
    info : instance of mne.measuerment_info.Info
        The info.
    meg_combined : bool
        Whether to return combined picks for grad and mag.
    ref_meg : bool
        If True include CTF / 4D reference channels
    exclude : list of string | str
        List of channels to exclude. If 'bads' (default), exclude channels
        in info['bads'].

    Returns
    -------
    picks_list : list of tuples
        The list of tuples of picks and the type string.
    """
    from ..channels.channels import _contains_ch_type
    picks_list = []
    has = [_contains_ch_type(info, k) for k in _DATA_CH_TYPES_SPLIT]
    has = dict(zip(_DATA_CH_TYPES_SPLIT, has))
    if has['mag'] and (meg_combined is not True or not has['grad']):
        picks_list.append(
            ('mag', pick_types(info, meg='mag', eeg=False, stim=False,
             ref_meg=ref_meg, exclude=exclude))
        )
    if has['grad'] and (meg_combined is not True or not has['mag']):
        picks_list.append(
            ('grad', pick_types(info, meg='grad', eeg=False, stim=False,
             ref_meg=ref_meg, exclude=exclude))
        )
    if has['mag'] and has['grad'] and meg_combined is True:
        picks_list.append(
            ('meg', pick_types(info, meg=True, eeg=False, stim=False,
             ref_meg=ref_meg, exclude=exclude))
        )
    for ch_type in _DATA_CH_TYPES_SPLIT:
        if ch_type in ['grad', 'mag']:  # exclude just MEG channels
            continue
        if has[ch_type]:
            picks_list.append(
                (ch_type, pick_types(info, meg=False, stim=False,
                 ref_meg=ref_meg, exclude=exclude, **{ch_type: True}))
            )
    return picks_list


def _check_excludes_includes(chs, info=None, allow_bads=False):
    """Ensure that inputs to exclude/include are list-like or "bads".

    Parameters
    ----------
    chs : any input, should be list, tuple, set, string
        The channels passed to include or exclude.
    allow_bads : bool
        Allow the user to supply "bads" as a string for auto exclusion.

    Returns
    -------
    chs : list
        Channels to be excluded/excluded. If allow_bads, and chs=="bads",
        this will be the bad channels found in 'info'.
    """
    from .meas_info import Info
    if not isinstance(chs, (list, tuple, set, np.ndarray)):
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


_PICK_TYPES_DATA_DICT = dict(
    meg=True, eeg=True, stim=False, eog=False, ecg=False, emg=False,
    misc=False, resp=False, chpi=False, exci=False, ias=False, syst=False,
    seeg=True, dipole=False, gof=False, bio=False, ecog=True, fnirs=True)
_PICK_TYPES_KEYS = tuple(list(_PICK_TYPES_DATA_DICT.keys()) + ['ref_meg'])
_DATA_CH_TYPES_SPLIT = ['mag', 'grad', 'eeg', 'seeg', 'ecog', 'hbo', 'hbr']

# Valid data types, ordered for consistency, used in viz/evoked.
_VALID_CHANNEL_TYPES = ['eeg', 'grad', 'mag', 'seeg', 'eog', 'ecg', 'emg',
                        'dipole', 'gof', 'bio', 'ecog', 'hbo', 'hbr',
                        'misc']


def _pick_data_channels(info, exclude='bads', with_ref_meg=True):
    """Pick only data channels."""
    return pick_types(info, ref_meg=with_ref_meg, exclude=exclude,
                      **_PICK_TYPES_DATA_DICT)


def _pick_aux_channels(info, exclude='bads'):
    """Pick only auxiliary channels.

    Corresponds to EOG, ECG, EMG and BIO
    """
    return pick_types(info, meg=False, eog=True, ecg=True, emg=True, bio=True,
                      ref_meg=False, exclude=exclude)


def _pick_data_or_ica(info):
    """Pick only data or ICA channels."""
    if any(ch_name.startswith('ICA') for ch_name in info['ch_names']):
        picks = pick_types(info, exclude=[], misc=True)
    else:
        picks = _pick_data_channels(info, exclude=[], with_ref_meg=True)
    return picks


def _pick_inst(inst, picks, exclude, copy=True):
    """Return an instance with picked and excluded channels."""
    if copy is True:
        inst = inst.copy()
    if picks is not None:
        pick_names = [inst.info['ch_names'][pick] for pick in picks]
    else:  # only pick channels that are plotted
        picks = _pick_data_channels(inst.info, exclude=[])
        pick_names = [inst.info['ch_names'][pick] for pick in picks]
    inst.pick_channels(pick_names)

    if exclude == 'bads':
        exclude = [ch for ch in inst.info['bads']
                   if ch in inst.info['ch_names']]
    if exclude is not None:
        inst.drop_channels(exclude)
    return inst


def _get_channel_types(info, picks=None, unique=True,
                       restrict_data_types=False):
    """Get the data channel types in an info instance."""
    picks = range(info['nchan']) if picks is None else picks
    ch_types = [channel_type(info, idx) for idx in range(info['nchan'])
                if idx in picks]
    if restrict_data_types is True:
        ch_types = [ch_type for ch_type in ch_types
                    if ch_type in _DATA_CH_TYPES_SPLIT]
    return set(ch_types) if unique is True else ch_types
