# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np

from .constants import FIFF
from .proj import _has_eeg_average_ref_proj, make_eeg_average_ref_proj
from .proj import setup_proj
from .pick import pick_types, pick_channels
from .base import BaseRaw
from ..evoked import Evoked
from ..epochs import BaseEpochs
from ..utils import (logger, warn, verbose, _validate_type, _check_preload,
                     _check_option)
from ..defaults import DEFAULTS


def _copy_channel(inst, ch_name, new_ch_name):
    """Add a copy of a channel specified by ch_name.

    Input data can be in the form of Raw, Epochs or Evoked.

    The instance object is modified inplace.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Data containing the EEG channels
    ch_name : str
        Name of the channel to copy.
    new_ch_name : str
        Name given to the copy of the channel.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        The data with a copy of a given channel.
    """
    new_inst = inst.copy().pick_channels([ch_name])
    new_inst.rename_channels({ch_name: new_ch_name})
    inst.add_channels([new_inst], force_update_info=True)
    return inst


def _apply_reference(inst, ref_from, ref_to=None):
    """Apply a custom EEG referencing scheme.

    This function modifies the instance in-place.

    Calculates a reference signal by taking the mean of a set of channels and
    applies the reference to another set of channels. Input data can be in the
    form of Raw, Epochs or Evoked.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Data containing the EEG channels and reference channel(s).
    ref_from : list of str
        The names of the channels to use to construct the reference. If an
        empty list is specified, the data is assumed to already have a proper
        reference and MNE will not attempt any re-referencing of the data.
    ref_to : list of str | None
        The names of the channels to apply the reference to. If None (which is
        the default), all EEG channels are chosen.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        The data with EEG channels re-referenced.
    ref_data : array, shape (n_times,)
        Array of reference data subtracted from EEG channels.

    Notes
    -----
    This function operates in-place.

    1. If the reference is applied to any EEG channels, this function removes
       any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. The data must be preloaded.

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.
    set_bipolar_reference : Convenience function for creating a bipolar
                            reference.
    """
    # Check to see that data is preloaded
    _check_preload(inst, "Applying a reference")

    eeg_idx = pick_types(inst.info, eeg=True, meg=False, ref_meg=False)

    if ref_to is None:
        ref_to = [inst.ch_names[i] for i in eeg_idx]
        extra = 'EEG channels found'
    else:
        extra = 'channels supplied'
    if len(ref_to) == 0:
        raise ValueError('No %s to apply the reference to' % (extra,))

    # After referencing, existing SSPs might not be valid anymore.
    projs_to_remove = []
    for i, proj in enumerate(inst.info['projs']):
        # Remove any average reference projections
        if proj['desc'] == 'Average EEG reference' or \
                proj['kind'] == FIFF.FIFFV_PROJ_ITEM_EEG_AVREF:
            logger.info('Removing existing average EEG reference '
                        'projection.')
            # Don't remove the projection right away, but do this at the end of
            # this loop.
            projs_to_remove.append(i)

        # Inactive SSPs may block re-referencing
        elif (not proj['active'] and
              len([ch for ch in (ref_from + ref_to)
                   if ch in proj['data']['col_names']]) > 0):

            raise RuntimeError(
                'Inactive signal space projection (SSP) operators are '
                'present that operate on sensors involved in the desired '
                'referencing scheme. These projectors need to be applied '
                'using the apply_proj() method function before the desired '
                'reference can be set.'
            )

    for i in projs_to_remove:
        del inst.info['projs'][i]

    # Need to call setup_proj after changing the projs:
    inst._projector, _ = \
        setup_proj(inst.info, add_eeg_ref=False, activate=False)

    # Compute reference
    if len(ref_from) > 0:
        # this is guaranteed below, but we should avoid the crazy pick_channels
        # behavior that [] gives all. Also use ordered=True just to make sure
        # that all supplied channels actually exist.
        assert len(ref_to) > 0
        ref_from = pick_channels(inst.ch_names, ref_from, ordered=True)
        ref_to = pick_channels(inst.ch_names, ref_to, ordered=True)

        data = inst._data
        ref_data = data[..., ref_from, :].mean(-2, keepdims=True)
        data[..., ref_to, :] -= ref_data
        ref_data = ref_data[..., 0, :]

        # If the reference touches EEG electrodes, note in the info that a
        # non-CAR has been applied.
        if len(np.intersect1d(ref_to, eeg_idx)) > 0:
            inst.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_ON
    else:
        ref_data = None

    return inst, ref_data


def add_reference_channels(inst, ref_channels, copy=True):
    """Add reference channels to data that consists of all zeros.

    Adds reference channels to data that were not included during recording.
    This is useful when you need to re-reference your data to different
    channels. These added channels will consist of all zeros.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Instance of Raw or Epochs with EEG channels and reference channel(s).
    ref_channels : str | list of str
        Name of the electrode(s) which served as the reference in the
        recording. If a name is provided, a corresponding channel is added
        and its data is set to 0. This is useful for later re-referencing.
    copy : bool
        Specifies whether the data will be copied (True) or modified in-place
        (False). Defaults to True.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with added EEG reference channels.
    """
    # Check to see that data is preloaded
    if not inst.preload:
        raise RuntimeError('Data needs to be preloaded.')
    if isinstance(ref_channels, str):
        ref_channels = [ref_channels]
    elif not isinstance(ref_channels, list):
        raise ValueError("`ref_channels` should be either str or list of str. "
                         "%s was provided." % type(ref_channels))
    for ch in ref_channels:
        if ch in inst.info['ch_names']:
            raise ValueError("Channel %s already specified in inst." % ch)

    # Once CAR is applied (active), don't allow adding channels
    if _has_eeg_average_ref_proj(inst.info['projs'], check_active=True):
        raise RuntimeError('Average reference already applied to data.')

    if copy:
        inst = inst.copy()

    if isinstance(inst, (BaseRaw, Evoked)):
        data = inst._data
        refs = np.zeros((len(ref_channels), data.shape[1]))
        data = np.vstack((data, refs))
        inst._data = data
    elif isinstance(inst, BaseEpochs):
        data = inst._data
        x, y, z = data.shape
        refs = np.zeros((x * len(ref_channels), z))
        data = np.vstack((data.reshape((x * y, z), order='F'), refs))
        data = data.reshape(x, y + len(ref_channels), z, order='F')
        inst._data = data
    else:
        raise TypeError("inst should be Raw, Epochs, or Evoked instead of %s."
                        % type(inst))
    nchan = len(inst.info['ch_names'])

    # only do this if we actually have digitisation points
    if inst.info.get('dig', None) is not None:
        # "zeroth" EEG electrode dig points is reference
        ref_dig_loc = [dl for dl in inst.info['dig'] if (
                       dl['kind'] == FIFF.FIFFV_POINT_EEG and
                       dl['ident'] == 0)]
        if len(ref_channels) > 1 or len(ref_dig_loc) != len(ref_channels):
            ref_dig_array = np.zeros(12)
            warn('The locations of multiple reference channels are ignored '
                 '(set to zero).')
        else:  # n_ref_channels == 1 and a single ref digitization exists
            ref_dig_array = np.concatenate((ref_dig_loc[0]['r'],
                                           ref_dig_loc[0]['r'], np.zeros(6)))
            # Replace the (possibly new) Ref location for each channel
            for idx in pick_types(inst.info, meg=False, eeg=True, exclude=[]):
                inst.info['chs'][idx]['loc'][3:6] = ref_dig_loc[0]['r']
    else:
        # we should actually be able to do this from the montage, but
        # it looks like the montage isn't stored, so we can't extract
        # this information. The user will just have to call set_montage()
        # by setting this to zero, we fall back to the old behavior
        # when missing digitisation
        ref_dig_array = np.zeros(12)

    for ch in ref_channels:
        chan_info = {'ch_name': ch,
                     'coil_type': FIFF.FIFFV_COIL_EEG,
                     'kind': FIFF.FIFFV_EEG_CH,
                     'logno': nchan + 1,
                     'scanno': nchan + 1,
                     'cal': 1,
                     'range': 1.,
                     'unit_mul': 0.,
                     'unit': FIFF.FIFF_UNIT_V,
                     'coord_frame': FIFF.FIFFV_COORD_HEAD,
                     'loc': ref_dig_array}
        inst.info['chs'].append(chan_info)
        inst.info._update_redundant()
    if isinstance(inst, BaseRaw):
        inst._cals = np.hstack((inst._cals, [1] * len(ref_channels)))
    inst.info._check_consistency()
    set_eeg_reference(inst, ref_channels=ref_channels, copy=False)
    return inst


_ref_dict = {
    FIFF.FIFFV_MNE_CUSTOM_REF_ON: 'on',
    FIFF.FIFFV_MNE_CUSTOM_REF_OFF: 'off',
    FIFF.FIFFV_MNE_CUSTOM_REF_CSD: 'CSD',
}


def _check_can_reref(inst):
    _validate_type(inst, (BaseRaw, BaseEpochs, Evoked), "Instance")
    current_custom = inst.info['custom_ref_applied']
    if current_custom not in (FIFF.FIFFV_MNE_CUSTOM_REF_ON,
                              FIFF.FIFFV_MNE_CUSTOM_REF_OFF):
        raise RuntimeError('Cannot set new reference on data with custom '
                           'reference type %r' % (_ref_dict[current_custom],))


@verbose
def set_eeg_reference(inst, ref_channels='average', copy=True,
                      projection=False, ch_type='auto', verbose=None):
    """Specify which reference to use for EEG data.

    Use this function to explicitly specify the desired reference for EEG.
    This can be either an existing electrode or a new virtual channel.
    This function will re-reference the data according to the desired
    reference.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Instance of Raw or Epochs with EEG channels and reference channel(s).
    ref_channels : list of str | str
        The name(s) of the channel(s) used to construct the reference. To apply
        an average reference, specify ``'average'`` here (default). Specify an
        empty list to remove a potentially existing average reference
        projection. Defaults to an average reference.
    copy : bool
        Specifies whether the data will be copied (True) or modified in-place
        (False). Defaults to True.
    projection : bool
        If ``ref_channels='average'`` this argument specifies if the average
        reference should be computed as a projection (True) or not (False;
        default). If ``projection=True``, the average reference is added as a
        projection and is not applied to the data (it can be applied
        afterwards with the ``apply_proj`` method). If ``projection=False``,
        the average reference is directly applied to the data.
        If ``ref_channels`` is not ``'average'``, ``projection`` must be set to
        ``False`` (the default in this case).
    ch_type : 'auto' | 'eeg' | 'ecog' | 'seeg'
        The name of the channel type to apply the reference to. If 'auto', the
        first channel type of eeg, ecog or seeg that is found (in that order)
        will be selected.

        .. versionadded:: 0.19
    %(verbose)s

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with EEG channels re-referenced. If ``ref_channels='average'`` and
        ``projection=True`` a projection will be added instead of directly
        re-referencing the data.
    ref_data : array
        Array of reference data subtracted from EEG channels. This will be
        ``None`` if ``ref_channels='average'`` and ``projection=True``.
    %(set_eeg_reference_see_also_notes)s
    """
    _check_can_reref(inst)

    if projection:  # average reference projector
        if ref_channels != 'average':
            raise ValueError('Setting projection=True is only supported for '
                             'ref_channels="average", got %r.'
                             % (ref_channels,))
        if _has_eeg_average_ref_proj(inst.info['projs']):
            warn('An average reference projection was already added. The data '
                 'has been left untouched.')
            return inst, None
        else:
            # Creating an average reference may fail. In this case, make
            # sure that the custom_ref_applied flag is left untouched.
            custom_ref_applied = inst.info['custom_ref_applied']
            try:
                inst.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_OFF
                inst.add_proj(make_eeg_average_ref_proj(inst.info,
                              activate=False))
            except Exception:
                inst.info['custom_ref_applied'] = custom_ref_applied
                raise
            # If the data has been preloaded, projections will no
            # longer be automatically applied.
            if inst.preload:
                logger.info('Average reference projection was added, '
                            'but has not been applied yet. Use the '
                            'apply_proj method to apply it.')
            return inst, None

    inst = inst.copy() if copy else inst

    _check_option('ch_type', ch_type, ('auto', 'eeg', 'ecog', 'seeg'))
    # if ch_type is 'auto', search through list to find first reasonable
    # reference-able channel type.
    possible_types = ['eeg', 'ecog', 'seeg']
    if ch_type == 'auto':
        for type_ in possible_types:
            if type_ in inst:
                ch_type = type_
                logger.info('%s channel type selected for '
                            're-referencing' % DEFAULTS['titles'][type_])
                break
        # if auto comes up empty, or the user specifies a bad ch_type.
        else:
            raise ValueError('No EEG, ECoG or sEEG channels found '
                             'to rereference.')

    ch_dict = {ch_type: True, 'meg': False, 'ref_meg': False}
    eeg_idx = pick_types(inst.info, **ch_dict)
    ch_sel = [inst.ch_names[i] for i in eeg_idx]

    if ref_channels == 'average' and not projection:  # apply average reference
        logger.info('Applying average reference.')
        ref_channels = ch_sel

    if ref_channels == []:
        logger.info('EEG data marked as already having the desired reference.')
    else:
        logger.info('Applying a custom %s '
                    'reference.' % DEFAULTS['titles'][type_])

    return _apply_reference(inst, ref_channels, ch_sel)


@verbose
def set_bipolar_reference(inst, anode, cathode, ch_name=None, ch_info=None,
                          drop_refs=True, copy=True, verbose=None):
    """Re-reference selected channels using a bipolar referencing scheme.

    A bipolar reference takes the difference between two channels (the anode
    minus the cathode) and adds it as a new virtual channel. The original
    channels will be dropped.

    Multiple anodes and cathodes can be specified, in which case multiple
    virtual channels will be created. The 1st anode will be subtracted from the
    1st cathode, the 2nd anode from the 2nd cathode, etc.

    By default, the virtual channels will be annotated with channel info of
    the anodes, their locations set to (0, 0, 0) and coil types set to
    EEG_BIPOLAR.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Data containing the unreferenced channels.
    anode : str | list of str
        The name(s) of the channel(s) to use as anode in the bipolar reference.
    cathode : str | list of str
        The name(s) of the channel(s) to use as cathode in the bipolar
        reference.
    ch_name : str | list of str | None
        The channel name(s) for the virtual channel(s) containing the resulting
        signal. By default, bipolar channels are named after the anode and
        cathode, but it is recommended to supply a more meaningful name.
    ch_info : dict | list of dict | None
        This parameter can be used to supply a dictionary (or a dictionary for
        each bipolar channel) containing channel information to merge in,
        overwriting the default values. Defaults to None.
    drop_refs : bool
        Whether to drop the anode/cathode channels from the instance.
    copy : bool
        Whether to operate on a copy of the data (True) or modify it in-place
        (False). Defaults to True.
    %(verbose)s

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with the specified channels re-referenced.

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.

    Notes
    -----
    1. If the anodes contain any EEG channels, this function removes
       any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. The data must be preloaded.

    .. versionadded:: 0.9.0
    """
    _check_can_reref(inst)
    if not isinstance(anode, list):
        anode = [anode]

    if not isinstance(cathode, list):
        cathode = [cathode]

    if len(anode) != len(cathode):
        raise ValueError('Number of anodes (got %d) must equal the number '
                         'of cathodes (got %d).' % (len(anode), len(cathode)))

    if ch_name is None:
        ch_name = ['%s-%s' % ac for ac in zip(anode, cathode)]
    elif not isinstance(ch_name, list):
        ch_name = [ch_name]
    if len(ch_name) != len(anode):
        raise ValueError('Number of channel names must equal the number of '
                         'anodes/cathodes (got %d).' % len(ch_name))

    # Check for duplicate channel names (it is allowed to give the name of the
    # anode or cathode channel, as they will be replaced).
    for ch, a, c in zip(ch_name, anode, cathode):
        if ch not in [a, c] and ch in inst.ch_names:
            raise ValueError('There is already a channel named "%s", please '
                             'specify a different name for the bipolar '
                             'channel using the ch_name parameter.' % ch)

    if ch_info is None:
        ch_info = [{} for _ in anode]
    elif not isinstance(ch_info, list):
        ch_info = [ch_info]
    if len(ch_info) != len(anode):
        raise ValueError('Number of channel info dictionaries must equal the '
                         'number of anodes/cathodes.')

    # Merge specified and anode channel information dictionaries
    new_chs = []
    for an, ci in zip(anode, ch_info):
        an_idx = inst.ch_names.index(an)
        this_chs = deepcopy(inst.info['chs'][an_idx])

        # Set channel location and coil type
        this_chs['loc'] = np.zeros(12)
        this_chs['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR

        this_chs.update(ci)
        new_chs.append(this_chs)

    if copy:
        inst = inst.copy()

    for i, (an, ca, name, chs) in enumerate(
            zip(anode, cathode, ch_name, new_chs)):
        if an in anode[i + 1:] or an in cathode[i + 1:] or not drop_refs:
            # Make a copy of the channel if it's still needed later
            # otherwise it's modified inplace
            _copy_channel(inst, an, 'TMP')
            an = 'TMP'
        _apply_reference(inst, [ca], [an])
        an_idx = inst.ch_names.index(an)
        inst.info['chs'][an_idx] = chs
        inst.info['chs'][an_idx]['ch_name'] = name
        logger.info('Bipolar channel added as "%s".' % name)
        inst.info._update_redundant()

    # Drop remaining channels.
    if drop_refs:
        drop_channels = list((set(anode) | set(cathode)) & set(inst.ch_names))
        inst.drop_channels(drop_channels)

    return inst
