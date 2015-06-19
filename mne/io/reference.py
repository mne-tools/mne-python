# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from .constants import FIFF
from .proj import _has_eeg_average_ref_proj, make_eeg_average_ref_proj
from .pick import pick_types
from .base import _BaseRaw
from ..evoked import Evoked
from ..epochs import Epochs
from ..utils import logger


def _apply_reference(inst, ref_from, ref_to=None, copy=True):
    """Apply a custom EEG referencing scheme.

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
        The names of the channels to apply the reference to. By default,
        all EEG channels are chosen.
    copy : bool
        Specifies whether the data will be copied (True) or modified in place
        (False). Defaults to True.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        The data with EEG channels rereferenced.
    ref_data : array, shape (n_times,)
        Array of reference data subtracted from EEG channels.

    Notes
    -----
    1. Do not use this function to apply an average reference. By default, an
       average reference projection has already been added upon loading raw
       data.

    2. If the reference is applied to any EEG channels, this function removes
       any pre-existing average reference projections.

    3. During source localization, the EEG signal should have an average
       reference.

    4. The data must be preloaded.

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.
    set_bipolar_reference : Convenience function for creating a bipolar
                            reference.
    """
    # Check to see that data is preloaded
    if not isinstance(inst, Evoked) and not inst.preload:
        raise RuntimeError('Data needs to be preloaded. Use '
                           'preload=True (or string) in the constructor.')

    eeg_idx = pick_types(inst.info, eeg=True, meg=False, ref_meg=False)

    if ref_to is None:
        ref_to = [inst.ch_names[i] for i in eeg_idx]

    if copy:
        inst = inst.copy()

    # After referencing, existing SSPs might not be valid anymore.
    for i, proj in enumerate(inst.info['projs']):
        if (not proj['active'] and
            len([ch for ch in (ref_from + ref_to)
                 if ch in proj['data']['col_names']]) > 0):

            # Remove any average reference projections, apply any other types
            if proj['desc'] == 'Average EEG reference' or \
                    proj['kind'] == FIFF.FIFFV_MNE_PROJ_ITEM_EEG_AVREF:
                logger.info('Removing existing average EEG reference '
                            'projection.')
                del inst.info['projs'][i]
            else:
                logger.info(
                    'Inactive signal space projection (SSP) operators are '
                    'present that operate on sensors involved in the current '
                    'referencing scheme. Applying them now. Be aware that '
                    'after re-referencing, these operators will be invalid.')
                inst.apply_proj()
            break

    ref_from = [inst.ch_names.index(ch) for ch in ref_from]
    ref_to = [inst.ch_names.index(ch) for ch in ref_to]

    if isinstance(inst, Evoked):
        data = inst.data
    else:
        data = inst._data

    # Compute reference
    if len(ref_from) > 0:
        ref_data = data[..., ref_from, :].mean(-2)

        if isinstance(inst, Epochs):
            data[:, ref_to, :] -= ref_data[:, np.newaxis, :]
        else:
            data[ref_to] -= ref_data
    else:
        ref_data = None

    # If the reference touches EEG electrodes, note in the info that a non-CAR
    # has been applied.
    if len(np.intersect1d(ref_to, eeg_idx)) > 0:
        inst.info['custom_ref_applied'] = True

    return inst, ref_data


def add_reference_channels(inst, ref_channels, copy=True):
    """Add reference channels to data that consists of all zeros.

    Adds reference channels to data that were not included during recording.
    This is useful when you need to re-reference your data to different
    channel. These added channels will consist of all zeros.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Instance of Raw or Epochs with EEG channels and reference channel(s).
    ref_channels : str | list of str
        Name of the electrode(s) which served as the reference in the
        recording. If a name is provided, a corresponding channel is added
        and its data is set to 0. This is useful for later re-referencing.
    copy : bool
        Specifies whether the data will be copied (True) or modified in place
        (False). Defaults to True.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with added EEG reference channels.
    """
    # Check to see that data is preloaded
    if not isinstance(inst, Evoked) and not inst.preload:
        raise RuntimeError('Data needs to be preloaded.')
    if isinstance(ref_channels, str):
        ref_channels = [ref_channels]
    elif not isinstance(ref_channels, list):
        raise ValueError("`ref_channels` should be either str or list of str. "
                         "%s was provided." % type(ref_channels))
    for ch in ref_channels:
        if ch in inst.info['ch_names']:
            raise ValueError("Channel %s already specified in inst." % ch)

    if copy:
        inst = inst.copy()

    if isinstance(inst, Evoked):
        data = inst.data
        refs = np.zeros((len(ref_channels), data.shape[1]))
        data = np.vstack((data, refs))
        inst.data = data
    elif isinstance(inst, _BaseRaw):
        data = inst._data
        refs = np.zeros((len(ref_channels), data.shape[1]))
        data = np.vstack((data, refs))
        inst._data = data
    elif isinstance(inst, Epochs):
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
    if ch in ref_channels:
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
                     'eeg_loc': np.zeros(3),
                     'loc': np.zeros(12)}
        inst.info['chs'].append(chan_info)
    inst.info['ch_names'].extend(ref_channels)
    inst.info['nchan'] = len(inst.info['ch_names'])
    if isinstance(inst, _BaseRaw):
        inst._cals = np.hstack((inst._cals, [1] * len(ref_channels)))

    return inst


def set_eeg_reference(inst, ref_channels=None, copy=True):
    """Rereference EEG channels to new reference channel(s).

    If multiple reference channels are specified, they will be averaged. If
    no reference channels are specified, an average reference will be applied.

    Parameters
    ----------
    inst : instance of Raw | Epochs | Evoked
        Instance of Raw or Epochs with EEG channels and reference channel(s).
    ref_channels : list of str | None
        The names of the channels to use to construct the reference. If None is
        specified here, an average reference will be applied in the form of an
        SSP projector. If an empty list is specified, the data is assumed to
        already have a proper reference and MNE will not attempt any
        re-referencing of the data. Defaults to an average reference (None).
    copy : bool
        Specifies whether the data will be copied (True) or modified in place
        (False). Defaults to True.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with EEG channels re-referenced.
    ref_data : array
        Array of reference data subtracted from EEG channels.

    Notes
    -----
    1. If a reference is requested that is not the average reference, this
       function removes any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. In order to apply a reference other than an average reference, the data
       must be preloaded.

    .. versionadded:: 0.9.0

    See Also
    --------
    set_bipolar_reference : Convenience function for creating bipolar
                            references.
    """
    if ref_channels is None:
        # CAR requested
        if _has_eeg_average_ref_proj(inst.info['projs']):
            logger.warning('An average reference projection was already '
                           'added. The data has been left untouched.')
            return inst, None
        else:
            inst.info['custom_ref_applied'] = False
            inst.add_proj(make_eeg_average_ref_proj(inst.info, activate=False))
            return inst, None
    else:
        logger.info('Applying a custom EEG reference.')
        return _apply_reference(inst, ref_channels, copy=copy)


def set_bipolar_reference(inst, anode, cathode, ch_name=None, ch_info=None,
                          copy=True):
    """Rereference selected channels using a bipolar referencing scheme.

    A bipolar reference takes the difference between two channels (the anode
    minus the cathode) and adds it as a new virtual channel. The original
    channels will be dropped.

    Multiple anodes and cathodes can be specified, in which case multiple
    vitual channels will be created. The 1st anode will be substracted from the
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
    copy : bool
        Whether to operate on a copy of the data (True) or modify it in-place
        (False). Defaults to True.

    Returns
    -------
    inst : instance of Raw | Epochs | Evoked
        Data with the specified channels re-referenced.

    Notes
    -----
    1. If the anodes contain any EEG channels, this function removes
       any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. The data must be preloaded.

    .. versionadded:: 0.9.0

    See Also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.
    """
    if not isinstance(anode, list):
        anode = [anode]

    if not isinstance(cathode, list):
        cathode = [cathode]

    if len(anode) != len(cathode):
        raise ValueError('Number of anodes must equal the number of cathodes.')

    if ch_name is None:
        ch_name = ['%s-%s' % ac for ac in zip(anode, cathode)]
    elif not isinstance(ch_name, list):
        ch_name = [ch_name]
    if len(ch_name) != len(anode):
        raise ValueError('Number of channel names must equal the number of '
                         'anodes/cathodes.')

    # Check for duplicate channel names (it is allowed to give the name of the
    # anode or cathode channel, as they will be replaced).
    for ch, a, c in zip(ch_name, anode, cathode):
        if ch not in [a, c] and ch in inst.ch_names:
            raise ValueError('There is already a channel named "%s", please '
                             'specify a different name for the bipolar '
                             'channel using the ch_name parameter.' % ch)

    if ch_info is None:
        ch_info = [{} for an in anode]
    elif not isinstance(ch_info, list):
        ch_info = [ch_info]
    if len(ch_info) != len(anode):
        raise ValueError('Number of channel info dictionaries must equal the '
                         'number of anodes/cathodes.')

    # Merge specified and anode channel information dictionaries
    new_ch_info = []
    for an, ci in zip(anode, ch_info):
        new_info = inst.info['chs'][inst.ch_names.index(an)].copy()

        # Set channel location and coil type
        if 'eeg_loc' in new_info:
            new_info['eeg_loc'] = np.zeros((3, 2))
        new_info['loc'] = np.zeros(12)
        new_info['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR

        new_info.update(ci)
        new_ch_info.append(new_info)

    if copy:
        inst = inst.copy()

    # Perform bipolar referencing
    for an, ca, name, info in zip(anode, cathode, ch_name, new_ch_info):
        inst, _ = _apply_reference(inst, [ca], [an], copy=False)
        an_idx = inst.ch_names.index(an)
        inst.info['chs'][an_idx] = info
        inst.info['chs'][an_idx]['ch_name'] = name
        inst.info['ch_names'][an_idx] = name
        logger.info('Bipolar channel added as "%s".' % name)

    # Drop cathode channels
    inst.drop_channels(cathode)

    return inst
