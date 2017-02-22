# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op

import numpy as np
from scipy import sparse

from ..externals.six import string_types

from ..utils import verbose, logger, warn, copy_function_doc_to_method_doc
from ..io.compensator import get_current_comp
from ..io.constants import FIFF
from ..io.meas_info import anonymize_info
from ..io.pick import (channel_type, pick_info, pick_types,
                       _check_excludes_includes, _PICK_TYPES_KEYS)


def _get_meg_system(info):
    """Educated guess for the helmet type based on channels."""
    system = '306m'
    for ch in info['chs']:
        if ch['kind'] == FIFF.FIFFV_MEG_CH:
            # Only take first 16 bits, as higher bits store CTF grad comp order
            coil_type = ch['coil_type'] & 0xFFFF
            if coil_type == FIFF.FIFFV_COIL_NM_122:
                system = '122m'
                break
            elif coil_type // 1000 == 3:  # All Vectorview coils are 30xx
                system = '306m'
                break
            elif (coil_type == FIFF.FIFFV_COIL_MAGNES_MAG or
                  coil_type == FIFF.FIFFV_COIL_MAGNES_GRAD):
                nmag = np.sum([c['kind'] == FIFF.FIFFV_MEG_CH
                               for c in info['chs']])
                system = 'Magnes_3600wh' if nmag > 150 else 'Magnes_2500wh'
                break
            elif coil_type == FIFF.FIFFV_COIL_CTF_GRAD:
                system = 'CTF_275'
                break
            elif coil_type == FIFF.FIFFV_COIL_KIT_GRAD:
                system = 'KIT'
                break
            elif coil_type == FIFF.FIFFV_COIL_BABY_GRAD:
                system = 'BabySQUID'
                break
    return system


def _contains_ch_type(info, ch_type):
    """Check whether a certain channel type is in an info object.

    Parameters
    ---------
    info : instance of Info
        The measurement information.
    ch_type : str
        the channel type to be checked for

    Returns
    -------
    has_ch_type : bool
        Whether the channel type is present or not.
    """
    if not isinstance(ch_type, string_types):
        raise ValueError('`ch_type` is of class {actual_class}. It must be '
                         '`str`'.format(actual_class=type(ch_type)))

    meg_extras = ['mag', 'grad', 'planar1', 'planar2']
    fnirs_extras = ['hbo', 'hbr']
    valid_channel_types = sorted([key for key in _PICK_TYPES_KEYS
                                  if key != 'meg'] + meg_extras + fnirs_extras)
    if ch_type not in valid_channel_types:
        raise ValueError('ch_type must be one of %s, not "%s"'
                         % (valid_channel_types, ch_type))
    if info is None:
        raise ValueError('Cannot check for channels of type "%s" because info '
                         'is None' % (ch_type,))
    return ch_type in [channel_type(info, ii) for ii in range(info['nchan'])]


def _get_ch_type(inst, ch_type):
    """Helper to choose a single channel type (usually for plotting).

    Usually used in plotting to plot a single datatype, e.g. look for mags,
    then grads, then ... to plot.
    """
    if ch_type is None:
        for type_ in ['mag', 'grad', 'planar1', 'planar2', 'eeg']:
            if type_ in inst:
                ch_type = type_
                break
        else:
            raise RuntimeError('No plottable channel types found')
    return ch_type


@verbose
def equalize_channels(candidates, verbose=None):
    """Equalize channel picks for a collection of MNE-Python objects.

    Parameters
    ----------
    candidates : list
        list Raw | Epochs | Evoked | AverageTFR
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    This function operates inplace.
    """
    from ..io.base import BaseRaw
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..time_frequency import AverageTFR

    if not all(isinstance(c, (BaseRaw, BaseEpochs, Evoked, AverageTFR))
               for c in candidates):
        raise ValueError('candidates must be Raw, Epochs, Evoked, or '
                         'AverageTFR')

    chan_max_idx = np.argmax([c.info['nchan'] for c in candidates])
    chan_template = candidates[chan_max_idx].ch_names
    logger.info('Identiying common channels ...')
    channels = [set(c.ch_names) for c in candidates]
    common_channels = set(chan_template).intersection(*channels)
    dropped = list()
    for c in candidates:
        drop_them = list(set(c.ch_names) - common_channels)
        if drop_them:
            c.drop_channels(drop_them)
            dropped.extend(drop_them)
    if dropped:
        dropped = list(set(dropped))
        logger.info('Dropped the following channels:\n%s' % dropped)
    else:
        logger.info('all channels are corresponding, nothing to do.')


class ContainsMixin(object):
    """Mixin class for Raw, Evoked, Epochs."""

    def __contains__(self, ch_type):
        """Check channel type membership.

        Parameters
        ----------
        ch_type : str
            Channel type to check for. Can be e.g. 'meg', 'eeg', 'stim', etc.

        Returns
        -------
        in : bool
            Whether or not the instance contains the given channel type.

        Examples
        --------
        Channel type membership can be tested as::

            >>> 'meg' in inst  # doctest: +SKIP
            True
            >>> 'seeg' in inst  # doctest: +SKIP
            False

        """
        if ch_type == 'meg':
            has_ch_type = (_contains_ch_type(self.info, 'mag') or
                           _contains_ch_type(self.info, 'grad'))
        else:
            has_ch_type = _contains_ch_type(self.info, ch_type)
        return has_ch_type

    @property
    def compensation_grade(self):
        """The current gradient compensation grade."""
        return get_current_comp(self.info)


# XXX Eventually de-duplicate with _kind_dict of mne/io/meas_info.py
_human2fiff = {'ecg': FIFF.FIFFV_ECG_CH,
               'eeg': FIFF.FIFFV_EEG_CH,
               'emg': FIFF.FIFFV_EMG_CH,
               'eog': FIFF.FIFFV_EOG_CH,
               'exci': FIFF.FIFFV_EXCI_CH,
               'ias': FIFF.FIFFV_IAS_CH,
               'misc': FIFF.FIFFV_MISC_CH,
               'resp': FIFF.FIFFV_RESP_CH,
               'seeg': FIFF.FIFFV_SEEG_CH,
               'stim': FIFF.FIFFV_STIM_CH,
               'syst': FIFF.FIFFV_SYST_CH,
               'bio': FIFF.FIFFV_BIO_CH,
               'ecog': FIFF.FIFFV_ECOG_CH,
               'hbo': FIFF.FIFFV_FNIRS_CH,
               'hbr': FIFF.FIFFV_FNIRS_CH}
_human2unit = {'ecg': FIFF.FIFF_UNIT_V,
               'eeg': FIFF.FIFF_UNIT_V,
               'emg': FIFF.FIFF_UNIT_V,
               'eog': FIFF.FIFF_UNIT_V,
               'exci': FIFF.FIFF_UNIT_NONE,
               'ias': FIFF.FIFF_UNIT_NONE,
               'misc': FIFF.FIFF_UNIT_V,
               'resp': FIFF.FIFF_UNIT_NONE,
               'seeg': FIFF.FIFF_UNIT_V,
               'stim': FIFF.FIFF_UNIT_NONE,
               'syst': FIFF.FIFF_UNIT_NONE,
               'bio': FIFF.FIFF_UNIT_V,
               'ecog': FIFF.FIFF_UNIT_V,
               'hbo': FIFF.FIFF_UNIT_MOL,
               'hbr': FIFF.FIFF_UNIT_MOL}
_unit2human = {FIFF.FIFF_UNIT_V: 'V',
               FIFF.FIFF_UNIT_T: 'T',
               FIFF.FIFF_UNIT_T_M: 'T/m',
               FIFF.FIFF_UNIT_MOL: 'M',
               FIFF.FIFF_UNIT_NONE: 'NA'}


def _check_set(ch, projs, ch_type):
    """Helper to make sure type change is compatible with projectors."""
    new_kind = _human2fiff[ch_type]
    if ch['kind'] != new_kind:
        for proj in projs:
            if ch['ch_name'] in proj['data']['col_names']:
                raise RuntimeError('Cannot change channel type for channel %s '
                                   'in projector "%s"'
                                   % (ch['ch_name'], proj['desc']))
    ch['kind'] = new_kind


class SetChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs."""

    @verbose
    def set_eeg_reference(self, ref_channels=None, verbose=None):
        """Rereference EEG channels to new reference channel(s).

        If multiple reference channels are specified, they will be averaged. If
        no reference channels are specified, an average reference will be
        applied.

        Parameters
        ----------
        ref_channels : list of str | None
            The names of the channels to use to construct the reference. If
            None (default), an average reference will be added as an SSP
            projector but not immediately applied to the data. If an empty list
            is specified, the data is assumed to already have a proper
            reference and MNE will not attempt any re-referencing of the data.
            Defaults to an average reference (None).
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            Data with EEG channels re-referenced. For ``ref_channels=None``,
            an average projector will be added instead of directly subtarcting
            data.

        Notes
        -----
        1. If a reference is requested that is not the average reference, this
           function removes any pre-existing average reference projections.

        2. During source localization, the EEG signal should have an average
           reference.

        3. In order to apply a reference other than an average reference, the
           data must be preloaded.

        .. versionadded:: 0.13.0

        See Also
        --------
        mne.set_bipolar_reference
        """
        from ..io.reference import set_eeg_reference
        return set_eeg_reference(self, ref_channels, copy=False)[0]

    def _get_channel_positions(self, picks=None):
        """Get channel locations from info.

        Parameters
        ----------
        picks : array-like of int | None
            Indices of channels to include. If None (default), all meg and eeg
            channels that are available are returned (bad channels excluded).

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        if picks is None:
            picks = pick_types(self.info, meg=True, eeg=True)
        chs = self.info['chs']
        pos = np.array([chs[k]['loc'][:3] for k in picks])
        n_zero = np.sum(np.sum(np.abs(pos), axis=1) == 0)
        if n_zero > 1:  # XXX some systems have origin (0, 0, 0)
            raise ValueError('Could not extract channel positions for '
                             '{} channels'.format(n_zero))
        return pos

    def _set_channel_positions(self, pos, names):
        """Update channel locations in info.

        Parameters
        ----------
        pos : array-like | np.ndarray, shape (n_points, 3)
            The channel positions to be set.
        names : list of str
            The names of the channels to be set.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        if len(pos) != len(names):
            raise ValueError('Number of channel positions not equal to '
                             'the number of names given.')
        pos = np.asarray(pos, dtype=np.float)
        if pos.shape[-1] != 3 or pos.ndim != 2:
            msg = ('Channel positions must have the shape (n_points, 3) '
                   'not %s.' % (pos.shape,))
            raise ValueError(msg)
        for name, p in zip(names, pos):
            if name in self.ch_names:
                idx = self.ch_names.index(name)
                self.info['chs'][idx]['loc'][:3] = p
            else:
                msg = ('%s was not found in the info. Cannot be updated.'
                       % name)
                raise ValueError(msg)

    def set_channel_types(self, mapping):
        """Define the sensor type of channels.

        Note: The following sensor types are accepted:
            ecg, eeg, emg, eog, exci, ias, misc, resp, seeg, stim, syst, ecog,
            hbo, hbr

        Parameters
        ----------
        mapping : dict
            a dictionary mapping a channel to a sensor type (str)
            {'EEG061': 'eog'}.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        ch_names = self.info['ch_names']

        # first check and assemble clean mappings of index and name
        unit_changes = dict()
        for ch_name, ch_type in mapping.items():
            if ch_name not in ch_names:
                raise ValueError("This channel name (%s) doesn't exist in "
                                 "info." % ch_name)

            c_ind = ch_names.index(ch_name)
            if ch_type not in _human2fiff:
                raise ValueError('This function cannot change to this '
                                 'channel type: %s. Accepted channel types '
                                 'are %s.'
                                 % (ch_type,
                                    ", ".join(sorted(_human2unit.keys()))))
            # Set sensor type
            _check_set(self.info['chs'][c_ind], self.info['projs'], ch_type)
            unit_old = self.info['chs'][c_ind]['unit']
            unit_new = _human2unit[ch_type]
            if unit_old not in _unit2human:
                raise ValueError("Channel '%s' has unknown unit (%s). Please "
                                 "fix the measurement info of your data."
                                 % (ch_name, unit_old))
            if unit_old != _human2unit[ch_type]:
                this_change = (_unit2human[unit_old], _unit2human[unit_new])
                if this_change not in unit_changes:
                    unit_changes[this_change] = list()
                unit_changes[this_change].append(ch_name)
            self.info['chs'][c_ind]['unit'] = _human2unit[ch_type]
            if ch_type in ['eeg', 'seeg', 'ecog']:
                coil_type = FIFF.FIFFV_COIL_EEG
            elif ch_type == 'hbo':
                coil_type = FIFF.FIFFV_COIL_FNIRS_HBO
            elif ch_type == 'hbr':
                coil_type = FIFF.FIFFV_COIL_FNIRS_HBR
            else:
                coil_type = FIFF.FIFFV_COIL_NONE
            self.info['chs'][c_ind]['coil_type'] = coil_type
        msg = "The unit for channel(s) {0} has changed from {1} to {2}."
        for this_change, names in unit_changes.items():
            warn(msg.format(", ".join(sorted(names)), *this_change))

    def rename_channels(self, mapping):
        """Rename channels.

        Parameters
        ----------
        mapping : dict | callable
            a dictionary mapping the old channel to a new channel name
            e.g. {'EEG061' : 'EEG161'}. Can also be a callable function
            that takes and returns a string (new in version 0.10.0).

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        rename_channels(self.info, mapping)

    @verbose
    def set_montage(self, montage, verbose=None):
        """Set EEG sensor configuration and head digitization.

        Parameters
        ----------
        montage : instance of Montage or DigMontage
            The montage to use.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Notes
        -----
        Operates in place.

        .. versionadded:: 0.9.0
        """
        from .montage import _set_montage
        _set_montage(self.info, montage)

    def plot_sensors(self, kind='topomap', ch_type=None, title=None,
                     show_names=False, ch_groups=None, axes=None, block=False,
                     show=True):
        """Plot sensor positions.

        Parameters
        ----------
        kind : str
            Whether to plot the sensors as 3d, topomap or as an interactive
            sensor selection dialog. Available options 'topomap', '3d',
            'select'. If 'select', a set of channels can be selected
            interactively by using lasso selector or clicking while holding
            control key. The selected channels are returned along with the
            figure instance. Defaults to 'topomap'.
        ch_type : None | str
            The channel type to plot. Available options 'mag', 'grad', 'eeg',
            'seeg', 'ecog', 'all'. If ``'all'``, all the available mag, grad,
            eeg, seeg and ecog channels are plotted. If None (default), then
            channels are chosen in the order given above.
        title : str | None
            Title for the figure. If None (default), equals to ``'Sensor
            positions (%s)' % ch_type``.
        show_names : bool
            Whether to display all channel names. Defaults to False.
        ch_groups : 'position' | array of shape (ch_groups, picks) | None
            Channel groups for coloring the sensors. If None (default), default
            coloring scheme is used. If 'position', the sensors are divided
            into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
            array, the channels are divided by picks given in the array.

            .. versionadded:: 0.13.0

        axes : instance of Axes | instance of Axes3D | None
            Axes to draw the sensors to. If ``kind='3d'``, axes must be an
            instance of Axes3D. If None (default), a new axes will be created.

            .. versionadded:: 0.13.0

        block : bool
            Whether to halt program execution until the figure is closed.
            Defaults to False.

            .. versionadded:: 0.13.0

        show : bool
            Show figure if True. Defaults to True.

        Returns
        -------
        fig : instance of matplotlib figure
            Figure containing the sensor topography.
        selection : list
            A list of selected channels. Only returned if ``kind=='select'``.

        See Also
        --------
        mne.viz.plot_layout

        Notes
        -----
        This function plots the sensor locations from the info structure using
        matplotlib. For drawing the sensors using mayavi see
        :func:`mne.viz.plot_trans`.

        .. versionadded:: 0.12.0
        """
        from ..viz.utils import plot_sensors
        return plot_sensors(self.info, kind=kind, ch_type=ch_type, title=title,
                            show_names=show_names, ch_groups=ch_groups,
                            axes=axes, block=block, show=show)

    @copy_function_doc_to_method_doc(anonymize_info)
    def anonymize(self):
        """
        .. versionadded:: 0.13.0
        """
        anonymize_info(self.info)
        return self


class UpdateChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs, AverageTFR."""

    def pick_types(self, meg=True, eeg=False, stim=False, eog=False,
                   ecg=False, emg=False, ref_meg='auto', misc=False,
                   resp=False, chpi=False, exci=False, ias=False, syst=False,
                   seeg=False, dipole=False, gof=False, bio=False, ecog=False,
                   fnirs=False, include=[], exclude='bads', selection=None):
        """Pick some channels by type and names.

        Parameters
        ----------
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
            If True include CTF / 4D reference channels. If 'auto', the
            reference channels are only included if compensations are present.
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
            fNIRS channels. If False (default) include none. If string it can
            be 'hbo' (to include channels measuring oxyhemoglobin) or 'hbr' (to
            include channels measuring deoxyhemoglobin).
        include : list of string
            List of additional channels to include. If empty do not include
            any.
        exclude : list of string | str
            List of channels to exclude. If 'bads' (default), exclude channels
            in ``info['bads']``.
        selection : list of string
            Restrict sensor channels (MEG, EEG) to this list of channel names.

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        idx = pick_types(
            self.info, meg=meg, eeg=eeg, stim=stim, eog=eog, ecg=ecg, emg=emg,
            ref_meg=ref_meg, misc=misc, resp=resp, chpi=chpi, exci=exci,
            ias=ias, syst=syst, seeg=seeg, dipole=dipole, gof=gof, bio=bio,
            ecog=ecog, fnirs=fnirs, include=include, exclude=exclude,
            selection=selection)
        self._pick_drop_channels(idx)
        return self

    def pick_channels(self, ch_names):
        """Pick some channels.

        Parameters
        ----------
        ch_names : list
            The list of channels to select.

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        See Also
        --------
        drop_channels

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        _check_excludes_includes(ch_names)
        idx = [self.ch_names.index(c) for c in ch_names if c in self.ch_names]
        self._pick_drop_channels(idx)
        return self

    def drop_channels(self, ch_names):
        """Drop some channels.

        Parameters
        ----------
        ch_names : list
            List of the names of the channels to remove.

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        See Also
        --------
        pick_channels

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        msg = ("'ch_names' should be a list of strings (the name[s] of the "
               "channel to be dropped), not a {0}.")
        if isinstance(ch_names, string_types):
            raise ValueError(msg.format("string"))
        else:
            if not all([isinstance(ch_name, string_types)
                        for ch_name in ch_names]):
                raise ValueError(msg.format(type(ch_names[0])))

        missing = [ch_name for ch_name in ch_names
                   if ch_name not in self.ch_names]
        if len(missing) > 0:
            msg = "Channel(s) {0} not found, nothing dropped."
            raise ValueError(msg.format(", ".join(missing)))

        bad_idx = [self.ch_names.index(ch_name) for ch_name in ch_names
                   if ch_name in self.ch_names]
        idx = np.setdiff1d(np.arange(len(self.ch_names)), bad_idx)
        self._pick_drop_channels(idx)

        return self

    def _pick_drop_channels(self, idx):
        # avoid circular imports
        from ..io.base import _check_preload
        from ..time_frequency import AverageTFR

        _check_preload(self, 'adding or dropping channels')

        if getattr(self, 'picks', None) is not None:
            self.picks = self.picks[idx]

        if hasattr(self, '_cals'):
            self._cals = self._cals[idx]

        pick_info(self.info, idx, copy=False)

        if getattr(self, '_projector', None) is not None:
            self._projector = self._projector[idx][:, idx]

        if self.preload:
            # All others (Evoked, Epochs, Raw) have chs axis=-2
            axis = -3 if isinstance(self, AverageTFR) else -2
            self._data = self._data.take(idx, axis=axis)

    def add_channels(self, add_list, force_update_info=False):
        """Append new channels to the instance.

        Parameters
        ----------
        add_list : list
            A list of objects to append to self. Must contain all the same
            type as the current object
        force_update_info : bool
            If True, force the info for objects to be appended to match the
            values in `self`. This should generally only be used when adding
            stim channels for which important metadata won't be overwritten.

            .. versionadded:: 0.12

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.
        """
        # avoid circular imports
        from ..io import BaseRaw, _merge_info
        from ..epochs import BaseEpochs

        if not isinstance(add_list, (list, tuple)):
            raise AssertionError('Input must be a list or tuple of objs')

        # Object-specific checks
        if not all([inst.preload for inst in add_list] + [self.preload]):
            raise AssertionError('All data must be preloaded')
        if isinstance(self, BaseRaw):
            con_axis = 0
            comp_class = BaseRaw
        elif isinstance(self, BaseEpochs):
            con_axis = 1
            comp_class = BaseEpochs
        else:
            con_axis = 0
            comp_class = type(self)
        if not all(isinstance(inst, comp_class) for inst in add_list):
            raise AssertionError('All input data must be of same type')
        data = [inst._data for inst in [self] + add_list]

        # Make sure that all dimensions other than channel axis are the same
        compare_axes = [i for i in range(data[0].ndim) if i != con_axis]
        shapes = np.array([dat.shape for dat in data])[:, compare_axes]
        if not ((shapes[0] - shapes) == 0).all():
            raise AssertionError('All dimensions except channels must match')

        # Create final data / info objects
        data = np.concatenate(data, axis=con_axis)
        infos = [self.info] + [inst.info for inst in add_list]
        new_info = _merge_info(infos, force_update_to_first=force_update_info)

        # Now update the attributes
        self._data = data
        self.info = new_info
        if isinstance(self, BaseRaw):
            self._cals = np.concatenate([getattr(inst, '_cals')
                                         for inst in [self] + add_list])
        return self


class InterpolationMixin(object):
    """Mixin class for Raw, Evoked, Epochs."""

    def interpolate_bads(self, reset_bads=True, mode='accurate'):
        """Interpolate bad MEG and EEG channels.

        Operates in place.

        Parameters
        ----------
        reset_bads : bool
            If True, remove the bads from info.
        mode : str
            Either `'accurate'` or `'fast'`, determines the quality of the
            Legendre polynomial expansion used for interpolation of MEG
            channels.

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from .interpolation import _interpolate_bads_eeg, _interpolate_bads_meg

        if getattr(self, 'preload', None) is False:
            raise ValueError('Data must be preloaded.')

        _interpolate_bads_eeg(self)
        _interpolate_bads_meg(self, mode=mode)

        if reset_bads is True:
            self.info['bads'] = []

        return self


def rename_channels(info, mapping):
    """Rename channels.

    Parameters
    ----------
    info : dict
        Measurement info.
    mapping : dict | callable
        a dictionary mapping the old channel to a new channel name
        e.g. {'EEG061' : 'EEG161'}. Can also be a callable function
        that takes and returns a string (new in version 0.10.0).
    """
    info._check_consistency()
    bads = list(info['bads'])  # make our own local copies
    ch_names = list(info['ch_names'])

    # first check and assemble clean mappings of index and name
    if isinstance(mapping, dict):
        orig_names = sorted(list(mapping.keys()))
        missing = [orig_name not in ch_names for orig_name in orig_names]
        if any(missing):
            raise ValueError("Channel name(s) in mapping missing from info: "
                             "%s" % np.array(orig_names)[np.array(missing)])
        new_names = [(ch_names.index(ch_name), new_name)
                     for ch_name, new_name in mapping.items()]
    elif callable(mapping):
        new_names = [(ci, mapping(ch_name))
                     for ci, ch_name in enumerate(ch_names)]
    else:
        raise ValueError('mapping must be callable or dict, not %s'
                         % (type(mapping),))

    # check we got all strings out of the mapping
    if any(not isinstance(new_name[1], string_types)
           for new_name in new_names):
        raise ValueError('New channel mapping must only be to strings')

    # do the remapping locally
    for c_ind, new_name in new_names:
        for bi, bad in enumerate(bads):
            if bad == ch_names[c_ind]:
                bads[bi] = new_name
        ch_names[c_ind] = new_name

    # check that all the channel names are unique
    if len(ch_names) != len(np.unique(ch_names)):
        raise ValueError('New channel names are not unique, renaming failed')

    # do the reampping in info
    info['bads'] = bads
    for ch, ch_name in zip(info['chs'], ch_names):
        ch['ch_name'] = ch_name
    info._update_redundant()
    info._check_consistency()


def _recursive_flatten(cell, dtype):
    """Helper to unpack mat files in Python."""
    while not isinstance(cell[0], dtype):
        cell = [c for d in cell for c in d]
    return cell


def read_ch_connectivity(fname, picks=None):
    """Parse FieldTrip neighbors .mat file.

    More information on these neighbor definitions can be found on the
    related FieldTrip documentation pages:
    http://fieldtrip.fcdonders.nl/template/neighbours

    Parameters
    ----------
    fname : str
        The file name. Example: 'neuromag306mag', 'neuromag306planar',
        'ctf275', 'biosemi64', etc.
    picks : array-like of int, shape (n_channels,)
        The indices of the channels to include. Must match the template.
        Defaults to None.

    Returns
    -------
    ch_connectivity : scipy.sparse matrix
        The connectivity matrix.
    ch_names : list
        The list of channel names present in connectivity matrix.
    """
    from scipy.io import loadmat
    if not op.isabs(fname):
        templates_dir = op.realpath(op.join(op.dirname(__file__),
                                            'data', 'neighbors'))
        templates = os.listdir(templates_dir)
        for f in templates:
            if f == fname:
                break
            if f == fname + '_neighb.mat':
                fname += '_neighb.mat'
                break
        else:
            raise ValueError('I do not know about this neighbor '
                             'template: "{}"'.format(fname))

        fname = op.join(templates_dir, fname)

    nb = loadmat(fname)['neighbours']
    ch_names = _recursive_flatten(nb['label'], string_types)
    neighbors = [_recursive_flatten(c, string_types) for c in
                 nb['neighblabel'].flatten()]
    assert len(ch_names) == len(neighbors)
    if picks is not None:
        if max(picks) >= len(ch_names):
            raise ValueError('The picks must be compatible with '
                             'channels. Found a pick ({}) which exceeds '
                             'the channel range ({})'
                             .format(max(picks), len(ch_names)))
    connectivity = _ch_neighbor_connectivity(ch_names, neighbors)
    if picks is not None:
        # picking before constructing matrix is buggy
        connectivity = connectivity[picks][:, picks]
        ch_names = [ch_names[p] for p in picks]
    return connectivity, ch_names


def _ch_neighbor_connectivity(ch_names, neighbors):
    """Compute sensor connectivity matrix.

    Parameters
    ----------
    ch_names : list of str
        The channel names.
    neighbors : list of list
        A list of list of channel names. The neighbors to
        which the channels in ch_names are connected with.
        Must be of the same length as ch_names.

    Returns
    -------
    ch_connectivity : scipy.sparse matrix
        The connectivity matrix.
    """
    if len(ch_names) != len(neighbors):
        raise ValueError('`ch_names` and `neighbors` must '
                         'have the same length')
    set_neighbors = set([c for d in neighbors for c in d])
    rest = set(ch_names) - set_neighbors
    if len(rest) > 0:
        raise ValueError('Some of your neighbors are not present in the '
                         'list of channel names')

    for neigh in neighbors:
        if (not isinstance(neigh, list) and
           not all(isinstance(c, string_types) for c in neigh)):
            raise ValueError('`neighbors` must be a list of lists of str')

    ch_connectivity = np.eye(len(ch_names), dtype=bool)
    for ii, neigbs in enumerate(neighbors):
        ch_connectivity[ii, [ch_names.index(i) for i in neigbs]] = True

    ch_connectivity = sparse.csr_matrix(ch_connectivity)
    return ch_connectivity


def fix_mag_coil_types(info):
    """Fix magnetometer coil types.

    Parameters
    ----------
    info : dict
        The info dict to correct. Corrections are done in-place.

    Notes
    -----
    This function changes magnetometer coil types 3022 (T1: SQ20483N) and
    3023 (T2: SQ20483-A) to 3024 (T3: SQ20950N) in the channel definition
    records in the info structure.

    Neuromag Vectorview systems can contain magnetometers with two
    different coil sizes (3022 and 3023 vs. 3024). The systems
    incorporating coils of type 3024 were introduced last and are used at
    the majority of MEG sites. At some sites with 3024 magnetometers,
    the data files have still defined the magnetometers to be of type
    3022 to ensure compatibility with older versions of Neuromag software.
    In the MNE software as well as in the present version of Neuromag
    software coil type 3024 is fully supported. Therefore, it is now safe
    to upgrade the data files to use the true coil type.

    .. note:: The effect of the difference between the coil sizes on the
              current estimates computed by the MNE software is very small.
              Therefore the use of mne_fix_mag_coil_types is not mandatory.
    """
    old_mag_inds = _get_T1T2_mag_inds(info)

    for ii in old_mag_inds:
        info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T3
    logger.info('%d of %d T1/T2 magnetometer types replaced with T3.' %
                (len(old_mag_inds), len(pick_types(info, meg='mag'))))
    info._check_consistency()


def _get_T1T2_mag_inds(info):
    """Helper to find T1/T2 magnetometer coil types."""
    picks = pick_types(info, meg='mag')
    old_mag_inds = []
    for ii in picks:
        ch = info['chs'][ii]
        if ch['coil_type'] in (FIFF.FIFFV_COIL_VV_MAG_T1,
                               FIFF.FIFFV_COIL_VV_MAG_T2):
            old_mag_inds.append(ii)
    return old_mag_inds
