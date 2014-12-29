# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

from .constants import FIFF
from .proj import _has_eeg_average_ref_proj, make_eeg_average_ref_proj
from .pick import pick_types
from ..utils import logger


def _apply_reference(raw, ref_from, ref_to=None, copy=True):
    """Apply a custom EEG referencing scheme.

    Calculates a reference signal by taking the mean of a set of channels and
    applies the reference to another set of channels.

    Parameters
    ----------
    raw : instance of Raw
        Instance of Raw with EEG channels and reference channel(s).
    ref_from : list of str
        The names of the channels to use to construct the reference. If an
        empty list is specified, the data is assumed to already have a proper
        reference and MNE will not attempt any re-referencing of the data.
    ref_to : list of str | None
        The names of the channels to apply the reference to. By default,
        all EEG channels are chosen.
    copy : bool
        Specifies whether instance of Raw will be copied or modified in place.
        Defaults to True.

    Returns
    -------
    raw : instance of Raw
        Instance of Raw with EEG channels rereferenced.
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

    See also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.
    set_bipolar_reference : Convenience function for creating a bipolar
                            reference.
    """
    # Check to see that raw data is preloaded
    if not raw.preload:
        raise RuntimeError('Raw data needs to be preloaded. Use '
                           'preload=True (or string) in the constructor.')

    eeg_idx = pick_types(raw.info, eeg=True, meg=False, ref_meg=False)

    ref_from = [raw.ch_names.index(ch) for ch in ref_from]
    if ref_to is None:
        ref_to = eeg_idx
    else:
        ref_to = [raw.ch_names.index(ch) for ch in ref_to]

    if copy:
        raw = raw.copy()

    # Compute reference
    if len(ref_from) > 0:
        ref_data = raw._data[ref_from].mean(0)
        raw._data[ref_to] -= ref_data
    else:
        ref_data = None

    # If the reference touches EEG electrodes, remove any pre-existing common
    # reference and note in the info that a non-CAR has been applied.
    if len(np.intersect1d(ref_to, eeg_idx)) > 0:
        raw.info['custom_ref_applied'] = True

        # Remove any existing average reference projections
        for i, proj in enumerate(raw.info['projs']):
            if proj['desc'] == 'Average EEG reference' or \
                    proj['kind'] == FIFF.FIFFV_MNE_PROJ_ITEM_EEG_AVREF:
                logger.info('Removing existing average EEG reference '
                            'projection.')
                del raw.info['projs'][i]

    return raw, ref_data


def set_eeg_reference(raw, ref_channels=None, copy=True):
    """Rereference EEG channels to new reference channel(s).

    If multiple reference channels are specified, they will be averaged. If
    no reference channels are specified, an average reference will be applied.

    Parameters
    ----------
    raw : instance of Raw
        Instance of Raw with EEG channels and reference channel(s).
    ref_channels : list of str | None
        The names of the channels to use to construct the reference. If None is
        specified here, an average reference will be applied in the form of an
        SSP projector. If an empty list is specified, the data is assumed to
        already have a proper reference and MNE will not attempt any
        re-referencing of the data.
    copy : bool
        Specifies whether instance of Raw will be copied (True) or modified in
        place (False). Defaults to True.

    Returns
    -------
    raw : instance of Raw
        Instance of Raw with eeg channels rereferenced.
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

    See also
    --------
    set_bipolar_reference : Convenience function for creating bipolar
                            references.
    """
    if ref_channels is None:
        # CAR requested
        if _has_eeg_average_ref_proj(raw.info['projs']):
            logger.warning('An average reference projection was already '
                           'added. The data has been left untouched.')
            return raw, None
        else:
            raw.add_proj(make_eeg_average_ref_proj(raw.info), activate=False)
            return raw, None
    else:
        logger.info('Applying a custom EEG reference.')
        return _apply_reference(raw, ref_channels, copy=copy)


def set_bipolar_reference(raw, anode, cathode, ch_name=None, ch_info=None,
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
    raw : instance of Raw
        Instance of Raw containing the unreferenced channels.
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
        overwriting the default values.
    copy : bool
        Whether to operate on a copy of the raw data (True) or modify it
        in-place (False). Defaults to True.

    Returns
    -------
    raw : instance of Raw
        Instance of Raw with the specified channels rereferenced.

    See also
    --------
    set_eeg_reference : Convenience function for creating an EEG reference.

    Notes
    -----
    1. If the anodes contain any EEG channels, this function removes
       any pre-existing average reference projections.

    2. During source localization, the EEG signal should have an average
       reference.

    3. The data must be preloaded.
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
        if ch not in [a, c] and ch in raw.ch_names:
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
        new_info = raw.info['chs'][raw.ch_names.index(an)].copy()

        # Set channel location and coil type
        if 'eeg_loc' in new_info:
            new_info['eeg_loc'] = np.zeros((3, 2))
        new_info['loc'] = np.zeros(12)
        new_info['coil_type'] = FIFF.FIFFV_COIL_EEG_BIPOLAR

        new_info.update(ci)
        new_ch_info.append(new_info)

    if copy:
        raw = raw.copy()

    # Perform bipolar referencing
    for an, ca, name, info in zip(anode, cathode, ch_name, new_ch_info):
        raw, _ = _apply_reference(raw, [ca], [an], copy=False)
        an_idx = raw.ch_names.index(an)
        raw.info['chs'][an_idx] = info
        raw.info['chs'][an_idx]['ch_name'] = name
        raw.info['ch_names'][an_idx] = name
        logger.info('Bipolar channel added as "%s".' % name)

    # Drop cathode channels
    raw.drop_channels(cathode)

    return raw
