# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: BSD-3-Clause


import os.path as op
from pathlib import Path
import sys
from collections import OrderedDict
from dataclasses import dataclass
from copy import deepcopy
from functools import partial
import string
from typing import Union

import numpy as np

from ..defaults import HEAD_SIZE_DEFAULT, _handle_default
from ..utils import (verbose, logger, warn,
                     _check_preload, _validate_type, fill_doc, _check_option,
                     _get_stim_channel, _check_fname, _check_dict_keys,
                     _on_missing)
from ..io.constants import FIFF
from ..io.meas_info import (anonymize_info, Info, MontageMixin, create_info,
                            _rename_comps)
from ..io.pick import (channel_type, pick_info, pick_types, _picks_by_type,
                       _check_excludes_includes, _contains_ch_type,
                       channel_indices_by_type, pick_channels, _picks_to_idx,
                       get_channel_type_constants,
                       _pick_data_channels)
from ..io.tag import _rename_list
from ..io.write import DATE_NONE
from ..io.proj import setup_proj


def _get_meg_system(info):
    """Educated guess for the helmet type based on channels."""
    have_helmet = True
    for ch in info['chs']:
        if ch['kind'] == FIFF.FIFFV_MEG_CH:
            # Only take first 16 bits, as higher bits store CTF grad comp order
            coil_type = ch['coil_type'] & 0xFFFF
            nmag = np.sum(
                [c['kind'] == FIFF.FIFFV_MEG_CH for c in info['chs']])
            if coil_type == FIFF.FIFFV_COIL_NM_122:
                system = '122m'
                break
            elif coil_type // 1000 == 3:  # All Vectorview coils are 30xx
                system = '306m'
                break
            elif (coil_type == FIFF.FIFFV_COIL_MAGNES_MAG or
                  coil_type == FIFF.FIFFV_COIL_MAGNES_GRAD):
                system = 'Magnes_3600wh' if nmag > 150 else 'Magnes_2500wh'
                break
            elif coil_type == FIFF.FIFFV_COIL_CTF_GRAD:
                system = 'CTF_275'
                break
            elif coil_type == FIFF.FIFFV_COIL_KIT_GRAD:
                system = 'KIT'
                # Our helmet does not match very well, so let's just create it
                have_helmet = False
                break
            elif coil_type == FIFF.FIFFV_COIL_BABY_GRAD:
                system = 'BabySQUID'
                break
            elif coil_type == FIFF.FIFFV_COIL_ARTEMIS123_GRAD:
                system = 'ARTEMIS123'
                have_helmet = False
                break
    else:
        system = 'unknown'
        have_helmet = False
    return system, have_helmet


def _get_ch_type(inst, ch_type, allow_ref_meg=False):
    """Choose a single channel type (usually for plotting).

    Usually used in plotting to plot a single datatype, e.g. look for mags,
    then grads, then ... to plot.
    """
    if ch_type is None:
        allowed_types = ['mag', 'grad', 'planar1', 'planar2', 'eeg', 'csd',
                         'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude',
                         'fnirs_fd_phase', 'fnirs_od', 'hbo', 'hbr',
                         'ecog', 'seeg', 'dbs']
        allowed_types += ['ref_meg'] if allow_ref_meg else []
        for type_ in allowed_types:
            if isinstance(inst, Info):
                if _contains_ch_type(inst, type_):
                    ch_type = type_
                    break
            elif type_ in inst:
                ch_type = type_
                break
        else:
            raise RuntimeError('No plottable channel types found')
    return ch_type


@verbose
def equalize_channels(instances, copy=True, verbose=None):
    """Equalize channel picks and ordering across multiple MNE-Python objects.

    First, all channels that are not common to each object are dropped. Then,
    using the first object in the list as a template, the channels of each
    object are re-ordered to match the template. The end result is that all
    given objects define the same channels, in the same order.

    Parameters
    ----------
    instances : list
        A list of MNE-Python objects to equalize the channels for. Objects can
        be of type Raw, Epochs, Evoked, AverageTFR, Forward, Covariance,
        CrossSpectralDensity or Info.
    copy : bool
        When dropping and/or re-ordering channels, an object will be copied
        when this parameter is set to ``True``. When set to ``False`` (the
        default) the dropping and re-ordering of channels happens in-place.

        .. versionadded:: 0.20.0
    %(verbose)s

    Returns
    -------
    equalized_instances : list
        A list of MNE-Python objects that have the same channels defined in the
        same order.

    Notes
    -----
    This function operates inplace.
    """
    from ..cov import Covariance
    from ..io.base import BaseRaw
    from ..io.meas_info import Info
    from ..epochs import BaseEpochs
    from ..evoked import Evoked
    from ..forward import Forward
    from ..time_frequency import _BaseTFR, CrossSpectralDensity

    # Instances need to have a `ch_names` attribute and a `pick_channels`
    # method that supports `ordered=True`.
    allowed_types = (BaseRaw, BaseEpochs, Evoked, _BaseTFR, Forward,
                     Covariance, CrossSpectralDensity, Info)
    allowed_types_str = ("Raw, Epochs, Evoked, TFR, Forward, Covariance, "
                         "CrossSpectralDensity or Info")
    for inst in instances:
        _validate_type(inst, allowed_types, "Instances to be modified",
                       allowed_types_str)

    chan_template = instances[0].ch_names
    logger.info('Identifying common channels ...')
    channels = [set(inst.ch_names) for inst in instances]
    common_channels = set(chan_template).intersection(*channels)
    all_channels = set(chan_template).union(*channels)
    dropped = list(set(all_channels - common_channels))

    # Preserve the order of chan_template
    order = np.argsort([chan_template.index(ch) for ch in common_channels])
    common_channels = np.array(list(common_channels))[order].tolist()

    # Update all instances to match the common_channels list
    reordered = False
    equalized_instances = []
    for inst in instances:
        # Only perform picking when needed
        if inst.ch_names != common_channels:
            if isinstance(inst, Info):
                sel = pick_channels(inst.ch_names, common_channels, exclude=[],
                                    ordered=True)
                inst = pick_info(inst, sel, copy=copy, verbose=False)
            else:
                if copy:
                    inst = inst.copy()
                inst.pick_channels(common_channels, ordered=True)
            if len(inst.ch_names) == len(common_channels):
                reordered = True
        equalized_instances.append(inst)

    if dropped:
        logger.info('Dropped the following channels:\n%s' % dropped)
    elif reordered:
        logger.info('Channels have been re-ordered.')

    return equalized_instances


channel_type_constants = get_channel_type_constants(include_defaults=True)
_human2fiff = {k: v.get('kind', FIFF.FIFFV_COIL_NONE) for k, v in
               channel_type_constants.items()}
_human2unit = {k: v.get('unit', FIFF.FIFF_UNIT_NONE) for k, v in
               channel_type_constants.items()}
_unit2human = {FIFF.FIFF_UNIT_V: 'V',
               FIFF.FIFF_UNIT_T: 'T',
               FIFF.FIFF_UNIT_T_M: 'T/m',
               FIFF.FIFF_UNIT_MOL: 'M',
               FIFF.FIFF_UNIT_NONE: 'NA',
               FIFF.FIFF_UNIT_CEL: 'C',
               FIFF.FIFF_UNIT_S: 'S'}


def _check_set(ch, projs, ch_type):
    """Ensure type change is compatible with projectors."""
    new_kind = _human2fiff[ch_type]
    if ch['kind'] != new_kind:
        for proj in projs:
            if ch['ch_name'] in proj['data']['col_names']:
                raise RuntimeError('Cannot change channel type for channel %s '
                                   'in projector "%s"'
                                   % (ch['ch_name'], proj['desc']))
    ch['kind'] = new_kind


class SetChannelsMixin(MontageMixin):
    """Mixin class for Raw, Evoked, Epochs."""

    @verbose
    def set_eeg_reference(self, ref_channels='average', projection=False,
                          ch_type='auto', forward=None, *, joint=False,
                          verbose=None):
        """Specify which reference to use for EEG data.

        Use this function to explicitly specify the desired reference for EEG.
        This can be either an existing electrode or a new virtual channel.
        This function will re-reference the data according to the desired
        reference.

        Parameters
        ----------
        %(ref_channels_set_eeg_reference)s
        %(projection_set_eeg_reference)s
        %(ch_type_set_eeg_reference)s
        %(forward_set_eeg_reference)s
        %(joint_set_eeg_reference)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            Data with EEG channels re-referenced. If ``ref_channels='average'``
            and ``projection=True`` a projection will be added instead of
            directly re-referencing the data.
        %(set_eeg_reference_see_also_notes)s
        """
        from ..io.reference import set_eeg_reference
        return set_eeg_reference(self, ref_channels=ref_channels, copy=False,
                                 projection=projection, ch_type=ch_type,
                                 forward=forward, joint=joint)[0]

    def _get_channel_positions(self, picks=None):
        """Get channel locations from info.

        Parameters
        ----------
        picks : str | list | slice | None
            None gets good data indices.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        picks = _picks_to_idx(self.info, picks)
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
        pos = np.asarray(pos, dtype=np.float64)
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

    @verbose
    def set_channel_types(self, mapping, verbose=None):
        """Define the sensor type of channels.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping a channel to a sensor type (str), e.g.,
            ``{'EEG061': 'eog'}``.
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The instance (modified in place).

            .. versionchanged:: 0.20
               Return the instance.

        Notes
        -----
        The following sensor types are accepted:

            ecg, eeg, emg, eog, exci, ias, misc, resp, seeg, dbs, stim, syst,
            ecog, hbo, hbr, fnirs_cw_amplitude, fnirs_fd_ac_amplitude,
            fnirs_fd_phase, fnirs_od, temperature, gsr

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
            if ch_type in ['eeg', 'seeg', 'ecog', 'dbs']:
                coil_type = FIFF.FIFFV_COIL_EEG
            elif ch_type == 'hbo':
                coil_type = FIFF.FIFFV_COIL_FNIRS_HBO
            elif ch_type == 'hbr':
                coil_type = FIFF.FIFFV_COIL_FNIRS_HBR
            elif ch_type == 'fnirs_cw_amplitude':
                coil_type = FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE
            elif ch_type == 'fnirs_fd_ac_amplitude':
                coil_type = FIFF.FIFFV_COIL_FNIRS_FD_AC_AMPLITUDE
            elif ch_type == 'fnirs_fd_phase':
                coil_type = FIFF.FIFFV_COIL_FNIRS_FD_PHASE
            elif ch_type == 'fnirs_od':
                coil_type = FIFF.FIFFV_COIL_FNIRS_OD
            else:
                coil_type = FIFF.FIFFV_COIL_NONE
            self.info['chs'][c_ind]['coil_type'] = coil_type
        msg = "The unit for channel(s) {0} has changed from {1} to {2}."
        for this_change, names in unit_changes.items():
            warn(msg.format(", ".join(sorted(names)), *this_change))
        return self

    @verbose
    def rename_channels(self, mapping, allow_duplicates=False, verbose=None):
        """Rename channels.

        Parameters
        ----------
        %(mapping_rename_channels_duplicates)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The instance (modified in place).

            .. versionchanged:: 0.20
               Return the instance.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from ..io import BaseRaw

        ch_names_orig = list(self.info['ch_names'])
        rename_channels(self.info, mapping, allow_duplicates)

        # Update self._orig_units for Raw
        if isinstance(self, BaseRaw):
            # whatever mapping was provided, now we can just use a dict
            mapping = dict(zip(ch_names_orig, self.info['ch_names']))
            for old_name, new_name in mapping.items():
                if old_name in self._orig_units:
                    self._orig_units[new_name] = self._orig_units.pop(old_name)
            ch_names = self.annotations.ch_names
            for ci, ch in enumerate(ch_names):
                ch_names[ci] = tuple(mapping.get(name, name) for name in ch)

        return self

    @verbose
    def plot_sensors(self, kind='topomap', ch_type=None, title=None,
                     show_names=False, ch_groups=None, to_sphere=True,
                     axes=None, block=False, show=True, sphere=None,
                     verbose=None):
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
            'seeg', 'dbs', 'ecog', 'all'. If ``'all'``, all the available mag,
            grad, eeg, seeg, dbs, and ecog channels are plotted. If
            None (default), then channels are chosen in the order given above.
        title : str | None
            Title for the figure. If None (default), equals to ``'Sensor
            positions (%%s)' %% ch_type``.
        show_names : bool | array of str
            Whether to display all channel names. If an array, only the channel
            names in the array are shown. Defaults to False.
        ch_groups : 'position' | array of shape (n_ch_groups, n_picks) | None
            Channel groups for coloring the sensors. If None (default), default
            coloring scheme is used. If 'position', the sensors are divided
            into 8 regions. See ``order`` kwarg of :func:`mne.viz.plot_raw`. If
            array, the channels are divided by picks given in the array.

            .. versionadded:: 0.13.0
        to_sphere : bool
            Whether to project the 3d locations to a sphere. When False, the
            sensor array appears similar as to looking downwards straight above
            the subject's head. Has no effect when kind='3d'. Defaults to True.

            .. versionadded:: 0.14.0
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
        %(sphere_topomap_auto)s
        %(verbose)s

        Returns
        -------
        fig : instance of Figure
            Figure containing the sensor topography.
        selection : list
            A list of selected channels. Only returned if ``kind=='select'``.

        See Also
        --------
        mne.viz.plot_layout

        Notes
        -----
        This function plots the sensor locations from the info structure using
        matplotlib. For drawing the sensors using PyVista see
        :func:`mne.viz.plot_alignment`.

        .. versionadded:: 0.12.0
        """
        from ..viz.utils import plot_sensors
        return plot_sensors(self.info, kind=kind, ch_type=ch_type, title=title,
                            show_names=show_names, ch_groups=ch_groups,
                            to_sphere=to_sphere, axes=axes, block=block,
                            show=show, sphere=sphere, verbose=verbose)

    @verbose
    def anonymize(self, daysback=None, keep_his=False, verbose=None):
        """Anonymize measurement information in place.

        Parameters
        ----------
        %(daysback_anonymize_info)s
        %(keep_his_anonymize_info)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The modified instance.

        Notes
        -----
        %(anonymize_info_notes)s

        .. versionadded:: 0.13.0
        """
        anonymize_info(self.info, daysback=daysback, keep_his=keep_his,
                       verbose=verbose)
        self.set_meas_date(self.info['meas_date'])  # unify annot update
        return self

    def set_meas_date(self, meas_date):
        """Set the measurement start date.

        Parameters
        ----------
        meas_date : datetime | float | tuple | None
            The new measurement date.
            If datetime object, it must be timezone-aware and in UTC.
            A tuple of (seconds, microseconds) or float (alias for
            ``(meas_date, 0)``) can also be passed and a datetime
            object will be automatically created. If None, will remove
            the time reference.

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The modified raw instance. Operates in place.

        See Also
        --------
        mne.io.Raw.anonymize

        Notes
        -----
        If you want to remove all time references in the file, call
        :func:`mne.io.anonymize_info(inst.info) <mne.io.anonymize_info>`
        after calling ``inst.set_meas_date(None)``.

        .. versionadded:: 0.20
        """
        from ..annotations import _handle_meas_date
        meas_date = _handle_meas_date(meas_date)
        with self.info._unlock():
            self.info['meas_date'] = meas_date

        # clear file_id and meas_id if needed
        if meas_date is None:
            for key in ('file_id', 'meas_id'):
                value = self.info.get(key)
                if value is not None:
                    assert 'msecs' not in value
                    value['secs'] = DATE_NONE[0]
                    value['usecs'] = DATE_NONE[1]
                    # The following copy is needed for a test CTF dataset
                    # otherwise value['machid'][:] = 0 would suffice
                    _tmp = value['machid'].copy()
                    _tmp[:] = 0
                    value['machid'] = _tmp

        if hasattr(self, 'annotations'):
            self.annotations._orig_time = meas_date
        return self


class UpdateChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs, Spectrum, AverageTFR."""

    @verbose
    def pick_types(self, meg=False, eeg=False, stim=False, eog=False,
                   ecg=False, emg=False, ref_meg='auto', *, misc=False,
                   resp=False, chpi=False, exci=False, ias=False, syst=False,
                   seeg=False, dipole=False, gof=False, bio=False,
                   ecog=False, fnirs=False, csd=False, dbs=False,
                   temperature=False, gsr=False,
                   include=(), exclude='bads', selection=None, verbose=None):
        """Pick some channels by type and names.

        Parameters
        ----------
        %(pick_types_params)s
        %(verbose)s

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
        idx = pick_types(
            self.info, meg=meg, eeg=eeg, stim=stim, eog=eog, ecg=ecg, emg=emg,
            ref_meg=ref_meg, misc=misc, resp=resp, chpi=chpi, exci=exci,
            ias=ias, syst=syst, seeg=seeg, dipole=dipole, gof=gof, bio=bio,
            ecog=ecog, fnirs=fnirs, csd=csd, dbs=dbs, include=include,
            exclude=exclude, selection=selection, temperature=temperature,
            gsr=gsr)

        self._pick_drop_channels(idx)

        # remove dropped channel types from reject and flat
        if getattr(self, 'reject', None) is not None:
            # use list(self.reject) to avoid RuntimeError for changing
            # dictionary size during iteration
            for ch_type in list(self.reject):
                if ch_type not in self:
                    del self.reject[ch_type]

        if getattr(self, 'flat', None) is not None:
            for ch_type in list(self.flat):
                if ch_type not in self:
                    del self.flat[ch_type]

        return self

    @verbose
    def pick_channels(self, ch_names, ordered=False, *, verbose=None):
        """Pick some channels.

        Parameters
        ----------
        ch_names : list
            The list of channels to select.
        ordered : bool
            If True (default False), ensure that the order of the channels in
            the modified instance matches the order of ``ch_names``.

            .. versionadded:: 0.20.0
        %(verbose)s

            .. versionadded:: 1.1

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        See Also
        --------
        drop_channels
        pick_types
        reorder_channels

        Notes
        -----
        The channel names given are assumed to be a set, i.e. the order
        does not matter. The original order of the channels is preserved.
        You can use ``reorder_channels`` to set channel order if necessary.

        .. versionadded:: 0.9.0
        """
        picks = pick_channels(self.info['ch_names'], ch_names, ordered=ordered)
        return self._pick_drop_channels(picks)

    @verbose
    def pick(self, picks, exclude=(), *, verbose=None):
        """Pick a subset of channels.

        Parameters
        ----------
        %(picks_all)s
        exclude : list | str
            Set of channels to exclude, only used when picking based on
            types (e.g., exclude="bads" when picks="meg").
        %(verbose)s

            .. versionadded:: 0.24.0

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.
        """
        picks = _picks_to_idx(self.info, picks, 'all', exclude,
                              allow_empty=False)
        return self._pick_drop_channels(picks)

    def reorder_channels(self, ch_names):
        """Reorder channels.

        Parameters
        ----------
        ch_names : list
            The desired channel order.

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        See Also
        --------
        drop_channels
        pick_types
        pick_channels

        Notes
        -----
        Channel names must be unique. Channels that are not in ``ch_names``
        are dropped.

        .. versionadded:: 0.16.0
        """
        _check_excludes_includes(ch_names)
        idx = list()
        for ch_name in ch_names:
            ii = self.ch_names.index(ch_name)
            if ii in idx:
                raise ValueError('Channel name repeated: %s' % (ch_name,))
            idx.append(ii)
        return self._pick_drop_channels(idx)

    @fill_doc
    def drop_channels(self, ch_names, on_missing='raise'):
        """Drop channel(s).

        Parameters
        ----------
        ch_names : iterable or str
            Iterable (e.g. list) of channel name(s) or channel name to remove.
        %(on_missing_ch_names)s

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        See Also
        --------
        reorder_channels
        pick_channels
        pick_types

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        if isinstance(ch_names, str):
            ch_names = [ch_names]

        try:
            all_str = all([isinstance(ch, str) for ch in ch_names])
        except TypeError:
            raise ValueError("'ch_names' must be iterable, got "
                             "type {} ({}).".format(type(ch_names), ch_names))

        if not all_str:
            raise ValueError("Each element in 'ch_names' must be str, got "
                             "{}.".format([type(ch) for ch in ch_names]))

        missing = [ch for ch in ch_names if ch not in self.ch_names]
        if len(missing) > 0:
            msg = "Channel(s) {0} not found, nothing dropped."
            _on_missing(on_missing, msg.format(", ".join(missing)))

        bad_idx = [self.ch_names.index(ch) for ch in ch_names
                   if ch in self.ch_names]
        idx = np.setdiff1d(np.arange(len(self.ch_names)), bad_idx)
        return self._pick_drop_channels(idx)

    @verbose
    def _pick_drop_channels(self, idx, *, verbose=None):
        # avoid circular imports
        from ..io import BaseRaw
        from ..time_frequency import AverageTFR, EpochsTFR
        from ..time_frequency.spectrum import BaseSpectrum

        msg = 'adding, dropping, or reordering channels'
        if isinstance(self, BaseRaw):
            if self._projector is not None:
                _check_preload(self, f'{msg} after calling .apply_proj()')
        else:
            _check_preload(self, msg)

        if getattr(self, 'picks', None) is not None:
            self.picks = self.picks[idx]

        if getattr(self, '_read_picks', None) is not None:
            self._read_picks = [r[idx] for r in self._read_picks]

        if hasattr(self, '_cals'):
            self._cals = self._cals[idx]

        pick_info(self.info, idx, copy=False)

        for key in ('_comp', '_projector'):
            mat = getattr(self, key, None)
            if mat is not None:
                setattr(self, key, mat[idx][:, idx])

        if isinstance(self, BaseSpectrum):
            axis = self._dims.index('channel')
        elif isinstance(self, (AverageTFR, EpochsTFR)):
            axis = -3
        else:  # All others (Evoked, Epochs, Raw) have chs axis=-2
            axis = -2
        if hasattr(self, '_data'):  # skip non-preloaded Raw
            self._data = self._data.take(idx, axis=axis)
        else:
            assert isinstance(self, BaseRaw) and not self.preload

        if isinstance(self, BaseRaw):
            self.annotations._prune_ch_names(self.info, on_missing='ignore')
            self._orig_units = {
                k: v for k, v in self._orig_units.items()
                if k in self.ch_names}

        self._pick_projs()
        return self

    def _pick_projs(self):
        """Keep only projectors which apply to at least 1 data channel."""
        drop_idx = []
        for idx, proj in enumerate(self.info['projs']):
            if not set(self.info['ch_names']) & set(proj['data']['col_names']):
                drop_idx.append(idx)

        for idx in drop_idx:
            logger.info(f"Removing projector {self.info['projs'][idx]}")

        if drop_idx and hasattr(self, 'del_proj'):
            self.del_proj(drop_idx)

        return self

    def add_channels(self, add_list, force_update_info=False):
        """Append new channels to the instance.

        Parameters
        ----------
        add_list : list
            A list of objects to append to self. Must contain all the same
            type as the current object.
        force_update_info : bool
            If True, force the info for objects to be appended to match the
            values in ``self``. This should generally only be used when adding
            stim channels for which important metadata won't be overwritten.

            .. versionadded:: 0.12

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        See Also
        --------
        drop_channels

        Notes
        -----
        If ``self`` is a Raw instance that has been preloaded into a
        :obj:`numpy.memmap` instance, the memmap will be resized.
        """
        # avoid circular imports
        from ..io import BaseRaw, _merge_info
        from ..epochs import BaseEpochs

        _validate_type(add_list, (list, tuple), 'Input')

        # Object-specific checks
        for inst in add_list + [self]:
            _check_preload(inst, "adding channels")
        if isinstance(self, BaseRaw):
            con_axis = 0
            comp_class = BaseRaw
        elif isinstance(self, BaseEpochs):
            con_axis = 1
            comp_class = BaseEpochs
        else:
            con_axis = 0
            comp_class = type(self)
        for inst in add_list:
            _validate_type(inst, comp_class, 'All input')
        data = [inst._data for inst in [self] + add_list]

        # Make sure that all dimensions other than channel axis are the same
        compare_axes = [i for i in range(data[0].ndim) if i != con_axis]
        shapes = np.array([dat.shape for dat in data])[:, compare_axes]
        for shape in shapes:
            if not ((shapes[0] - shape) == 0).all():
                raise ValueError(
                    'All data dimensions except channels must match, got '
                    f'{shapes[0]} != {shape}')
        del shapes

        # Create final data / info objects
        infos = [self.info] + [inst.info for inst in add_list]
        new_info = _merge_info(infos, force_update_to_first=force_update_info)

        # Now update the attributes
        if isinstance(self._data, np.memmap) and con_axis == 0 and \
                sys.platform != 'darwin':  # resizing not available--no mremap
            # Use a resize and fill in other ones
            out_shape = (sum(d.shape[0] for d in data),) + data[0].shape[1:]
            n_bytes = np.prod(out_shape) * self._data.dtype.itemsize
            self._data.flush()
            self._data.base.resize(n_bytes)
            self._data = np.memmap(self._data.filename, mode='r+',
                                   dtype=self._data.dtype, shape=out_shape)
            assert self._data.shape == out_shape
            assert self._data.nbytes == n_bytes
            offset = len(data[0])
            for d in data[1:]:
                this_len = len(d)
                self._data[offset:offset + this_len] = d
                offset += this_len
        else:
            self._data = np.concatenate(data, axis=con_axis)
        self.info = new_info
        if isinstance(self, BaseRaw):
            self._cals = np.concatenate([getattr(inst, '_cals')
                                         for inst in [self] + add_list])
            # We should never use these since data are preloaded, let's just
            # set it to something large and likely to break (2 ** 31 - 1)
            extra_idx = [2147483647] * sum(info['nchan'] for info in infos[1:])
            assert all(len(r) == infos[0]['nchan'] for r in self._read_picks)
            self._read_picks = [
                np.concatenate([r, extra_idx]) for r in self._read_picks]
            assert all(len(r) == self.info['nchan'] for r in self._read_picks)
            for other in add_list:
                self._orig_units.update(other._orig_units)
        elif isinstance(self, BaseEpochs):
            self.picks = np.arange(self._data.shape[1])
            if hasattr(self, '_projector'):
                activate = False if self._do_delayed_proj else self.proj
                self._projector, self.info = setup_proj(self.info, False,
                                                        activate=activate)

        return self

    @fill_doc
    def add_reference_channels(self, ref_channels):
        """Add reference channels to data that consists of all zeros.

        Adds reference channels to data that were not included during
        recording. This is useful when you need to re-reference your data
        to different channels. These added channels will consist of all zeros.

        Parameters
        ----------
        %(ref_channels)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
               The modified instance.
        """
        from ..io.reference import add_reference_channels

        return add_reference_channels(self, ref_channels, copy=False)


class InterpolationMixin(object):
    """Mixin class for Raw, Evoked, Epochs."""

    @verbose
    def interpolate_bads(self, reset_bads=True, mode='accurate',
                         origin='auto', method=None, exclude=(),
                         verbose=None):
        """Interpolate bad MEG and EEG channels.

        Operates in place.

        Parameters
        ----------
        reset_bads : bool
            If True, remove the bads from info.
        mode : str
            Either ``'accurate'`` or ``'fast'``, determines the quality of the
            Legendre polynomial expansion used for interpolation of channels
            using the minimum-norm method.
        origin : array-like, shape (3,) | str
            Origin of the sphere in the head coordinate frame and in meters.
            Can be ``'auto'`` (default), which means a head-digitization-based
            origin fit.

            .. versionadded:: 0.17
        method : dict
            Method to use for each channel type.
            Currently only the key "eeg" has multiple options:

            - ``"spline"`` (default)
                Use spherical spline interpolation.
            - ``"MNE"``
                Use minimum-norm projection to a sphere and back.
                This is the method used for MEG channels.

            The value for "meg" is "MNE", and the value for
            "fnirs" is "nearest". The default (None) is thus an alias for::

                method=dict(meg="MNE", eeg="spline", fnirs="nearest")

            .. versionadded:: 0.21
        exclude : list | tuple
            The channels to exclude from interpolation. If excluded a bad
            channel will stay in bads.
        %(verbose)s

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from ..bem import _check_origin
        from .interpolation import _interpolate_bads_eeg,\
            _interpolate_bads_meeg, _interpolate_bads_nirs

        _check_preload(self, "interpolation")
        method = _handle_default('interpolation_method', method)
        for key in method:
            _check_option('method[key]', key, ('meg', 'eeg', 'fnirs'))
        _check_option("method['eeg']", method['eeg'], ('spline', 'MNE'))
        _check_option("method['meg']", method['meg'], ('MNE',))
        _check_option("method['fnirs']", method['fnirs'], ('nearest',))

        if len(self.info['bads']) == 0:
            warn('No bad channels to interpolate. Doing nothing...')
            return self
        logger.info('Interpolating bad channels')
        origin = _check_origin(origin, self.info)
        if method['eeg'] == 'spline':
            _interpolate_bads_eeg(self, origin=origin, exclude=exclude)
            eeg_mne = False
        else:
            eeg_mne = True
        _interpolate_bads_meeg(self, mode=mode, origin=origin, eeg=eeg_mne,
                               exclude=exclude)
        _interpolate_bads_nirs(self, exclude=exclude)

        if reset_bads is True:
            self.info['bads'] = \
                [ch for ch in self.info['bads'] if ch in exclude]

        return self


@verbose
def rename_channels(info, mapping, allow_duplicates=False, verbose=None):
    """Rename channels.

    Parameters
    ----------
    %(info_not_none)s Note: modified in place.
    %(mapping_rename_channels_duplicates)s
    %(verbose)s
    """
    _validate_type(info, Info, 'info')
    info._check_consistency()
    bads = list(info['bads'])  # make our own local copies
    ch_names = list(info['ch_names'])

    # first check and assemble clean mappings of index and name
    if isinstance(mapping, dict):
        _check_dict_keys(mapping, ch_names, key_description="channel name(s)",
                         valid_key_source="info")
        new_names = [(ch_names.index(ch_name), new_name)
                     for ch_name, new_name in mapping.items()]
    elif callable(mapping):
        new_names = [(ci, mapping(ch_name))
                     for ci, ch_name in enumerate(ch_names)]
    else:
        raise ValueError('mapping must be callable or dict, not %s'
                         % (type(mapping),))

    # check we got all strings out of the mapping
    for new_name in new_names:
        _validate_type(new_name[1], 'str', 'New channel mappings')

    # do the remapping locally
    for c_ind, new_name in new_names:
        for bi, bad in enumerate(bads):
            if bad == ch_names[c_ind]:
                bads[bi] = new_name
        ch_names[c_ind] = new_name

    # check that all the channel names are unique
    if len(ch_names) != len(np.unique(ch_names)) and not allow_duplicates:
        raise ValueError('New channel names are not unique, renaming failed')

    # do the remapping in info
    info['bads'] = bads
    ch_names_mapping = dict()
    for ch, ch_name in zip(info['chs'], ch_names):
        ch_names_mapping[ch['ch_name']] = ch_name
        ch['ch_name'] = ch_name
    # .get b/c fwd info omits it
    _rename_comps(info.get('comps', []), ch_names_mapping)
    if 'projs' in info:  # fwd might omit it
        for proj in info['projs']:
            proj['data']['col_names'][:] = \
                _rename_list(proj['data']['col_names'], ch_names_mapping)
    info._update_redundant()
    info._check_consistency()


def _recursive_flatten(cell, dtype):
    """Unpack mat files in Python."""
    if len(cell) > 0:
        while not isinstance(cell[0], dtype):
            cell = [c for d in cell for c in d]
    return cell


@dataclass
class _BuiltinChannelAdjacency:
    name: str
    description: str
    fname: str
    source_url: Union[str, None]


_ft_neighbor_url_t = string.Template(
    'https://github.com/fieldtrip/fieldtrip/raw/master/'
    'template/neighbours/$fname'
)

_BUILTIN_CHANNEL_ADJACENCIES = [
    _BuiltinChannelAdjacency(
        name='biosemi16',
        description='Biosemi 16-electrode cap',
        fname='biosemi16_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='biosemi16_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='biosemi32',
        description='Biosemi 32-electrode cap',
        fname='biosemi32_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='biosemi32_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='biosemi64',
        description='Biosemi 64-electrode cap',
        fname='biosemi64_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='biosemi64_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='bti148',
        description='BTI 148-channel system',
        fname='bti148_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='bti148_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='bti248',
        description='BTI 248-channel system',
        fname='bti248_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='bti248_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='bti248grad',
        description='BTI 248 gradiometer system',
        fname='bti248grad_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='bti248grad_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='ctf64',
        description='CTF 64 axial gradiometer',
        fname='ctf64_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='ctf64_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='ctf151',
        description='CTF 151 axial gradiometer',
        fname='ctf151_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='ctf151_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='ctf275',
        description='CTF 275 axial gradiometer',
        fname='ctf275_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='ctf275_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='easycap32ch-avg',
        description='',
        fname='easycap32ch-avg_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycap32ch-avg_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='easycap64ch-avg',
        description='',
        fname='easycap64ch-avg_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycap64ch-avg_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='easycap128ch-avg',
        description='',
        fname='easycap128ch-avg_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycap128ch-avg_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='easycapM1',
        description='Easycap M1',
        fname='easycapM1_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycapM1_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='easycapM11',
        description='Easycap M11',
        fname='easycapM11_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycapM11_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='easycapM14',
        description='Easycap M14',
        fname='easycapM14_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycapM14_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='easycapM15',
        description='Easycap M15',
        fname='easycapM15_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='easycapM15_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='KIT-157',
        description='',
        fname='KIT-157_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='KIT-208',
        description='',
        fname='KIT-208_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='KIT-NYU-2019',
        description='',
        fname='KIT-NYU-2019_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='KIT-UMD-1',
        description='',
        fname='KIT-UMD-1_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='KIT-UMD-2',
        description='',
        fname='KIT-UMD-2_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='KIT-UMD-3',
        description='',
        fname='KIT-UMD-3_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='KIT-UMD-4',
        description='',
        fname='KIT-UMD-4_neighb.mat',
        source_url=None,
    ),
    _BuiltinChannelAdjacency(
        name='neuromag306mag',
        description='Neuromag306, only magnetometers',
        fname='neuromag306mag_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='neuromag306mag_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='neuromag306planar',
        description='Neuromag306, only planar gradiometers',
        fname='neuromag306planar_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='neuromag306planar_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='neuromag122cmb',
        description='Neuromag122, only combined planar gradiometers',
        fname='neuromag122cmb_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='neuromag122cmb_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='neuromag306cmb',
        description='Neuromag306, only combined planar gradiometers',
        fname='neuromag306cmb_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='neuromag306cmb_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='ecog256',
        description='ECOG 256channels, average referenced',
        fname='ecog256_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='ecog256_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='ecog256bipolar',
        description='ECOG 256channels, bipolar referenced',
        fname='ecog256bipolar_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='ecog256bipolar_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='eeg1010_neighb',
        description='',
        fname='eeg1010_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='eeg1010_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='elec1005',
        description='Standard 10-05 system',
        fname='elec1005_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='elec1005_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='elec1010',
        description='Standard 10-10 system',
        fname='elec1010_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='elec1010_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='elec1020',
        description='Standard 10-20 system',
        fname='elec1020_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='elec1020_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='itab28',
        description='ITAB 28-channel system',
        fname='itab28_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='itab28_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='itab153',
        description='ITAB 153-channel system',
        fname='itab153_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='itab153_neighb.mat'),
    ),
    _BuiltinChannelAdjacency(
        name='language29ch-avg',
        description='MPI for Psycholinguistic: Averaged 29-channel cap',
        fname='language29ch-avg_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='language29ch-avg_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='mpi_59_channels',
        description='MPI for Psycholinguistic: 59-channel cap',
        fname='mpi_59_channels_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='mpi_59_channels_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='yokogawa160',
        description='',
        fname='yokogawa160_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='yokogawa160_neighb.mat'),  # noqa: E501
    ),
    _BuiltinChannelAdjacency(
        name='yokogawa440',
        description='',
        fname='yokogawa440_neighb.mat',
        source_url=_ft_neighbor_url_t.substitute(fname='yokogawa440_neighb.mat'),  # noqa: E501
    ),
]


@fill_doc
def get_builtin_ch_adjacencies(*, descriptions=False):
    """Get a list of all FieldTrip neighbor definitions shipping with MNE.

    The names of the these neighbor definitions can be passed to
    :func:`read_ch_adjacency`.

    Parameters
    ----------
    descriptions : bool
        Whether to return not only the neighbor definition names, but also
        their corresponding descriptions. If ``True``, a list of tuples is
        returned, where the first tuple element is the neighbor definition name
        and the second is the description. If ``False`` (default), only the
        names are returned.

    Returns
    -------
    neighbor_name : list of str | list of tuple
        If ``descriptions=False``, the names of all builtin FieldTrip neighbor
        definitions that can be loaded directly via :func:`read_ch_adjacency`.

        If ``descriptions=True``, a list of tuples ``(name, description)``.

    Notes
    -----
    .. versionadded:: 1.1
    """
    if descriptions:
        return sorted(
            [(m.name, m.description) for m in _BUILTIN_CHANNEL_ADJACENCIES],
            key=lambda x: x[0].casefold()  # only sort based on name
        )
    else:
        return sorted(
            [m.name for m in _BUILTIN_CHANNEL_ADJACENCIES],
            key=str.casefold
        )


@fill_doc
def read_ch_adjacency(fname, picks=None):
    """Read a channel adjacency ("neighbors") file that ships with MNE.

    More information on these neighbor definitions can be found on the related
    `FieldTrip documentation pages
    <http://www.fieldtriptoolbox.org/template/neighbours/>`__.

    Parameters
    ----------
    fname : str
        The path to the file to load, or the name of a channel adjacency
        matrix that ships with MNE-Python.

        .. note::
            You can retrieve the names of all
            built-in channel adjacencies via
            :func:`mne.channels.get_builtin_ch_adjacencies`.
    %(picks_all)s
        Picks must match the template.

    Returns
    -------
    ch_adjacency : scipy.sparse.csr_matrix, shape (n_channels, n_channels)
        The adjacency matrix.
    ch_names : list
        The list of channel names present in adjacency matrix.

    See Also
    --------
    get_builtin_ch_adjacencies
    mne.viz.plot_ch_adjacency
    find_ch_adjacency
    mne.stats.combine_adjacency

    Notes
    -----
    If the neighbor definition you need is not shipped by MNE-Python,
    you may use :func:`find_ch_adjacency` to compute the
    adjacency matrix based on your 2D sensor locations.

    Note that depending on your use case, you may need to additionally use
    :func:`mne.stats.combine_adjacency` to prepare a final "adjacency"
    to pass to the eventual function.
    """
    from scipy.io import loadmat
    if op.isabs(fname):
        fname = _check_fname(
            fname=fname,
            overwrite='read',
            must_exist=True
        )
    else:  # built-in FieldTrip neighbors
        ch_adj_name = fname
        del fname
        if ch_adj_name.endswith('_neighb.mat'):  # backward-compat
            ch_adj_name = ch_adj_name.replace('_neighb.mat', '')

        if ch_adj_name not in get_builtin_ch_adjacencies():
            raise ValueError(
                f'No built-in channel adjacency matrix found with name: '
                f'{ch_adj_name}. Valid names are: '
                f'{", ".join(get_builtin_ch_adjacencies())}'
            )

        ch_adj = [a for a in _BUILTIN_CHANNEL_ADJACENCIES
                  if a.name == ch_adj_name][0]
        fname = ch_adj.fname
        templates_dir = Path(__file__).resolve().parent / 'data' / 'neighbors'
        fname = _check_fname(  # only needed to convert to a string
            fname=templates_dir / fname,
            overwrite='read',
            must_exist=True
        )

    nb = loadmat(fname)['neighbours']
    ch_names = _recursive_flatten(nb['label'], str)
    picks = _picks_to_idx(len(ch_names), picks)
    neighbors = [_recursive_flatten(c, str) for c in
                 nb['neighblabel'].flatten()]
    assert len(ch_names) == len(neighbors)
    adjacency = _ch_neighbor_adjacency(ch_names, neighbors)
    # picking before constructing matrix is buggy
    adjacency = adjacency[picks][:, picks]
    ch_names = [ch_names[p] for p in picks]

    # make sure MEG channel names contain space after "MEG"
    for idx, ch_name in enumerate(ch_names):
        if ch_name.startswith('MEG') and not ch_name[3] == ' ':
            ch_name = ch_name.replace('MEG', 'MEG ')
            ch_names[idx] = ch_name

    return adjacency, ch_names


def _ch_neighbor_adjacency(ch_names, neighbors):
    """Compute sensor adjacency matrix.

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
    ch_adjacency : scipy.sparse.spmatrix
        The adjacency matrix.
    """
    from scipy import sparse
    if len(ch_names) != len(neighbors):
        raise ValueError('`ch_names` and `neighbors` must '
                         'have the same length')
    set_neighbors = {c for d in neighbors for c in d}
    rest = set_neighbors - set(ch_names)
    if len(rest) > 0:
        raise ValueError('Some of your neighbors are not present in the '
                         'list of channel names')

    for neigh in neighbors:
        if (not isinstance(neigh, list) and
                not all(isinstance(c, str) for c in neigh)):
            raise ValueError('`neighbors` must be a list of lists of str')

    ch_adjacency = np.eye(len(ch_names), dtype=bool)
    for ii, neigbs in enumerate(neighbors):
        ch_adjacency[ii, [ch_names.index(i) for i in neigbs]] = True
    ch_adjacency = sparse.csr_matrix(ch_adjacency)
    return ch_adjacency


@fill_doc
def find_ch_adjacency(info, ch_type):
    """Find the adjacency matrix for the given channels.

    This function tries to infer the appropriate adjacency matrix template
    for the given channels. If a template is not found, the adjacency matrix
    is computed using Delaunay triangulation based on 2D sensor locations.

    Parameters
    ----------
    %(info_not_none)s
    ch_type : str | None
        The channel type for computing the adjacency matrix. Currently
        supports ``'mag'``, ``'grad'``, ``'eeg'`` and ``None``.
        If ``None``, the info must contain only one channel type.

    Returns
    -------
    ch_adjacency : scipy.sparse.csr_matrix, shape (n_channels, n_channels)
        The adjacency matrix.
    ch_names : list
        The list of channel names present in adjacency matrix.

    See Also
    --------
    mne.viz.plot_ch_adjacency
    mne.stats.combine_adjacency
    get_builtin_ch_adjacencies
    read_ch_adjacency

    Notes
    -----
    .. versionadded:: 0.15

    Automatic detection of an appropriate adjacency matrix template only
    works for MEG data at the moment. This means that the adjacency matrix
    is always computed for EEG data and never loaded from a template file. If
    you want to load a template for a given montage use
    :func:`read_ch_adjacency` directly.

    .. warning::
        If Delaunay triangulation is used to calculate the adjacency matrix it
        may yield partially unexpected results (e.g., include unwanted edges
        between non-adjacent sensors). Therefore, it is recommended to check
        (and, if necessary, manually modify) the result by inspecting it
        via :func:`mne.viz.plot_ch_adjacency`.

    Note that depending on your use case, you may need to additionally use
    :func:`mne.stats.combine_adjacency` to prepare a final "adjacency"
    to pass to the eventual function.
    """
    if ch_type is None:
        picks = channel_indices_by_type(info)
        if sum([len(p) != 0 for p in picks.values()]) != 1:
            raise ValueError('info must contain only one channel type if '
                             'ch_type is None.')
        ch_type = channel_type(info, 0)
    else:
        _check_option('ch_type', ch_type, ['mag', 'grad', 'eeg'])
    (has_vv_mag, has_vv_grad, is_old_vv, has_4D_mag, ctf_other_types,
     has_CTF_grad, n_kit_grads, has_any_meg, has_eeg_coils,
     has_eeg_coils_and_meg, has_eeg_coils_only,
     has_neuromag_122_grad, has_csd_coils) = _get_ch_info(info)
    conn_name = None
    if has_vv_mag and ch_type == 'mag':
        conn_name = 'neuromag306mag'
    elif has_vv_grad and ch_type == 'grad':
        conn_name = 'neuromag306planar'
    elif has_4D_mag:
        if 'MEG 248' in info['ch_names']:
            idx = info['ch_names'].index('MEG 248')
            grad = info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_MAGNES_GRAD
            mag = info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_MAGNES_MAG
            if ch_type == 'grad' and grad:
                conn_name = 'bti248grad'
            elif ch_type == 'mag' and mag:
                conn_name = 'bti248'
        elif 'MEG 148' in info['ch_names'] and ch_type == 'mag':
            idx = info['ch_names'].index('MEG 148')
            if info['chs'][idx]['coil_type'] == FIFF.FIFFV_COIL_MAGNES_MAG:
                conn_name = 'bti148'
    elif has_CTF_grad and ch_type == 'mag':
        if info['nchan'] < 100:
            conn_name = 'ctf64'
        elif info['nchan'] > 200:
            conn_name = 'ctf275'
        else:
            conn_name = 'ctf151'
    elif n_kit_grads > 0:
        from ..io.kit.constants import KIT_NEIGHBORS
        conn_name = KIT_NEIGHBORS.get(info['kit_system_id'])

    if conn_name is not None:
        logger.info(f'Reading adjacency matrix for {conn_name}.')
        return read_ch_adjacency(conn_name)
    logger.info('Could not find a adjacency matrix for the data. '
                'Computing adjacency based on Delaunay triangulations.')
    return _compute_ch_adjacency(info, ch_type)


@fill_doc
def _compute_ch_adjacency(info, ch_type):
    """Compute channel adjacency matrix using Delaunay triangulations.

    Parameters
    ----------
    %(info_not_none)s
    ch_type : str
        The channel type for computing the adjacency matrix. Currently
        supports 'mag', 'grad' and 'eeg'.

    Returns
    -------
    ch_adjacency : scipy.sparse.csr_matrix, shape (n_channels, n_channels)
        The adjacency matrix.
    ch_names : list
        The list of channel names present in adjacency matrix.
    """
    from scipy import sparse
    from scipy.spatial import Delaunay
    from .. import spatial_tris_adjacency
    from ..channels.layout import _find_topomap_coords, _pair_grad_sensors
    combine_grads = (ch_type == 'grad'
                     and any([coil_type in [ch['coil_type']
                                            for ch in info['chs']]
                              for coil_type in
                              [FIFF.FIFFV_COIL_VV_PLANAR_T1,
                               FIFF.FIFFV_COIL_NM_122]]))

    picks = dict(_picks_by_type(info, exclude=[]))[ch_type]
    ch_names = [info['ch_names'][pick] for pick in picks]
    if combine_grads:
        pairs = _pair_grad_sensors(info, topomap_coords=False, exclude=[])
        if len(pairs) != len(picks):
            raise RuntimeError('Cannot find a pair for some of the '
                               'gradiometers. Cannot compute adjacency '
                               'matrix.')
        # only for one of the pair
        xy = _find_topomap_coords(info, picks[::2], sphere=HEAD_SIZE_DEFAULT)
    else:
        xy = _find_topomap_coords(info, picks, sphere=HEAD_SIZE_DEFAULT)
    tri = Delaunay(xy)
    neighbors = spatial_tris_adjacency(tri.simplices)

    if combine_grads:
        ch_adjacency = np.eye(len(picks), dtype=bool)
        for idx, neigbs in zip(neighbors.row, neighbors.col):
            for ii in range(2):  # make sure each pair is included
                for jj in range(2):
                    ch_adjacency[idx * 2 + ii, neigbs * 2 + jj] = True
                    ch_adjacency[idx * 2 + ii, idx * 2 + jj] = True  # pair
        ch_adjacency = sparse.csr_matrix(ch_adjacency)
    else:
        ch_adjacency = sparse.lil_matrix(neighbors)
        ch_adjacency.setdiag(np.repeat(1, ch_adjacency.shape[0]))
        ch_adjacency = ch_adjacency.tocsr()

    return ch_adjacency, ch_names


@fill_doc
def fix_mag_coil_types(info, use_cal=False):
    """Fix magnetometer coil types.

    Parameters
    ----------
    %(info_not_none)s Corrections are done in-place.
    use_cal : bool
        If True, further refine the check for old coil types by checking
        ``info['chs'][ii]['cal']``.

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
              Therefore the use of ``fix_mag_coil_types`` is not mandatory.
    """
    old_mag_inds = _get_T1T2_mag_inds(info, use_cal)

    for ii in old_mag_inds:
        info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T3
    logger.info('%d of %d magnetometer types replaced with T3.' %
                (len(old_mag_inds),
                 len(pick_types(info, meg='mag', exclude=[]))))
    info._check_consistency()


def _get_T1T2_mag_inds(info, use_cal=False):
    """Find T1/T2 magnetometer coil types."""
    picks = pick_types(info, meg='mag', exclude=[])
    old_mag_inds = []
    # From email exchanges, systems with the larger T2 coil only use the cal
    # value of 2.09e-11. Newer T3 magnetometers use 4.13e-11 or 1.33e-10
    # (Triux). So we can use a simple check for > 3e-11.
    for ii in picks:
        ch = info['chs'][ii]
        if ch['coil_type'] in (FIFF.FIFFV_COIL_VV_MAG_T1,
                               FIFF.FIFFV_COIL_VV_MAG_T2):
            if use_cal:
                if ch['cal'] > 3e-11:
                    old_mag_inds.append(ii)
            else:
                old_mag_inds.append(ii)
    return old_mag_inds


def _get_ch_info(info):
    """Get channel info for inferring acquisition device."""
    chs = info['chs']
    # Only take first 16 bits, as higher bits store CTF comp order
    coil_types = {ch['coil_type'] & 0xFFFF for ch in chs}
    channel_types = {ch['kind'] for ch in chs}

    has_vv_mag = any(k in coil_types for k in
                     [FIFF.FIFFV_COIL_VV_MAG_T1, FIFF.FIFFV_COIL_VV_MAG_T2,
                      FIFF.FIFFV_COIL_VV_MAG_T3])
    has_vv_grad = any(k in coil_types for k in [FIFF.FIFFV_COIL_VV_PLANAR_T1,
                                                FIFF.FIFFV_COIL_VV_PLANAR_T2,
                                                FIFF.FIFFV_COIL_VV_PLANAR_T3])
    has_neuromag_122_grad = any(k in coil_types
                                for k in [FIFF.FIFFV_COIL_NM_122])

    is_old_vv = ' ' in chs[0]['ch_name']

    has_4D_mag = FIFF.FIFFV_COIL_MAGNES_MAG in coil_types
    ctf_other_types = (FIFF.FIFFV_COIL_CTF_REF_MAG,
                       FIFF.FIFFV_COIL_CTF_REF_GRAD,
                       FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD)
    has_CTF_grad = (FIFF.FIFFV_COIL_CTF_GRAD in coil_types or
                    (FIFF.FIFFV_MEG_CH in channel_types and
                     any(k in ctf_other_types for k in coil_types)))
    # hack due to MNE-C bug in IO of CTF
    # only take first 16 bits, as higher bits store CTF comp order
    n_kit_grads = sum(ch['coil_type'] & 0xFFFF == FIFF.FIFFV_COIL_KIT_GRAD
                      for ch in chs)

    has_any_meg = any([has_vv_mag, has_vv_grad, has_4D_mag, has_CTF_grad,
                       n_kit_grads])
    has_eeg_coils = (FIFF.FIFFV_COIL_EEG in coil_types and
                     FIFF.FIFFV_EEG_CH in channel_types)
    has_eeg_coils_and_meg = has_eeg_coils and has_any_meg
    has_eeg_coils_only = has_eeg_coils and not has_any_meg
    has_csd_coils = (FIFF.FIFFV_COIL_EEG_CSD in coil_types and
                     FIFF.FIFFV_EEG_CH in channel_types)

    return (has_vv_mag, has_vv_grad, is_old_vv, has_4D_mag, ctf_other_types,
            has_CTF_grad, n_kit_grads, has_any_meg, has_eeg_coils,
            has_eeg_coils_and_meg, has_eeg_coils_only, has_neuromag_122_grad,
            has_csd_coils)


@fill_doc
def make_1020_channel_selections(info, midline="z"):
    """Return dict mapping from ROI names to lists of picks for 10/20 setups.

    This passes through all channel names, and uses a simple heuristic to
    separate channel names into three Region of Interest-based selections:
    Left, Midline and Right. The heuristic is that channels ending on any of
    the characters in ``midline`` are filed under that heading, otherwise those
    ending in odd numbers under "Left", those in even numbers under "Right".
    Other channels are ignored. This is appropriate for 10/20 files, but not
    for other channel naming conventions.
    If an info object is provided, lists are sorted from posterior to anterior.

    Parameters
    ----------
    %(info_not_none)s If possible, the channel lists will be sorted
        posterior-to-anterior; otherwise they default to the order specified in
        ``info["ch_names"]``.
    midline : str
        Names ending in any of these characters are stored under the
        ``Midline`` key. Defaults to 'z'. Note that capitalization is ignored.

    Returns
    -------
    selections : dict
        A dictionary mapping from ROI names to lists of picks (integers).
    """
    _validate_type(info, "info")

    try:
        from .layout import find_layout
        layout = find_layout(info)
        pos = layout.pos
        ch_names = layout.names
    except RuntimeError:  # no channel positions found
        ch_names = info["ch_names"]
        pos = None

    selections = dict(Left=[], Midline=[], Right=[])
    for pick, channel in enumerate(ch_names):
        last_char = channel[-1].lower()  # in 10/20, last char codes hemisphere
        if last_char in midline:
            selection = "Midline"
        elif last_char.isdigit():
            selection = "Left" if int(last_char) % 2 else "Right"
        else:  # ignore the channel
            continue
        selections[selection].append(pick)

    if pos is not None:
        # sort channels from front to center
        # (y-coordinate of the position info in the layout)
        selections = {selection: np.array(picks)[pos[picks, 1].argsort()]
                      for selection, picks in selections.items()}

    return selections


@verbose
def combine_channels(inst, groups, method='mean', keep_stim=False,
                     drop_bad=False, verbose=None):
    """Combine channels based on specified channel grouping.

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        An MNE-Python object to combine the channels for. The object can be of
        type Raw, Epochs, or Evoked.
    groups : dict
        Specifies which channels are aggregated into a single channel, with
        aggregation method determined by the ``method`` parameter. One new
        pseudo-channel is made per dict entry; the dict values must be lists of
        picks (integer indices of ``ch_names``). For example::

            groups=dict(Left=[1, 2, 3, 4], Right=[5, 6, 7, 8])

        Note that within a dict entry all channels must have the same type.
    method : str | callable
        Which method to use to combine channels. If a :class:`str`, must be one
        of 'mean', 'median', or 'std' (standard deviation). If callable, the
        callable must accept one positional input (data of shape ``(n_channels,
        n_times)``, or ``(n_epochs, n_channels, n_times)``) and return an
        :class:`array <numpy.ndarray>` of shape ``(n_times,)``, or ``(n_epochs,
        n_times)``. For example with an instance of Raw or Evoked::

            method = lambda data: np.mean(data, axis=0)

        Another example with an instance of Epochs::

            method = lambda data: np.median(data, axis=1)

        Defaults to ``'mean'``.
    keep_stim : bool
        If ``True``, include stimulus channels in the resulting object.
        Defaults to ``False``.
    drop_bad : bool
        If ``True``, drop channels marked as bad before combining. Defaults to
        ``False``.
    %(verbose)s

    Returns
    -------
    combined_inst : instance of Raw, Epochs, or Evoked
        An MNE-Python object of the same type as the input ``inst``, containing
        one virtual channel for each group in ``groups`` (and, if ``keep_stim``
        is ``True``, also containing stimulus channels).
    """
    from ..io import BaseRaw, RawArray
    from .. import BaseEpochs, EpochsArray, Evoked, EvokedArray

    ch_axis = 1 if isinstance(inst, BaseEpochs) else 0
    ch_idx = list(range(inst.info['nchan']))
    ch_names = inst.info['ch_names']
    ch_types = inst.get_channel_types()
    inst_data = inst.data if isinstance(inst, Evoked) else inst.get_data()
    groups = OrderedDict(deepcopy(groups))

    # Convert string values of ``method`` into callables
    # XXX Possibly de-duplicate with _make_combine_callable of mne/viz/utils.py
    if isinstance(method, str):
        method_dict = {key: partial(getattr(np, key), axis=ch_axis)
                       for key in ('mean', 'median', 'std')}
        try:
            method = method_dict[method]
        except KeyError:
            raise ValueError('"method" must be a callable, or one of "mean", '
                             f'"median", or "std"; got "{method}".')

    # Instantiate channel info and data
    new_ch_names, new_ch_types, new_data = [], [], []
    if not isinstance(keep_stim, bool):
        raise TypeError('"keep_stim" must be of type bool, not '
                        f'{type(keep_stim)}.')
    if keep_stim:
        stim_ch_idx = list(pick_types(inst.info, meg=False, stim=True))
        if stim_ch_idx:
            new_ch_names = [ch_names[idx] for idx in stim_ch_idx]
            new_ch_types = [ch_types[idx] for idx in stim_ch_idx]
            new_data = [np.take(inst_data, idx, axis=ch_axis)
                        for idx in stim_ch_idx]
        else:
            warn('Could not find stimulus channels.')

    # Get indices of bad channels
    ch_idx_bad = []
    if not isinstance(drop_bad, bool):
        raise TypeError('"drop_bad" must be of type bool, not '
                        f'{type(drop_bad)}.')
    if drop_bad and inst.info['bads']:
        ch_idx_bad = pick_channels(ch_names, inst.info['bads'])

    # Check correctness of combinations
    for this_group, this_picks in groups.items():
        # Check if channel indices are out of bounds
        if not all(idx in ch_idx for idx in this_picks):
            raise ValueError('Some channel indices are out of bounds.')
        # Check if heterogeneous sensor type combinations
        this_ch_type = np.array(ch_types)[this_picks]
        if len(set(this_ch_type)) > 1:
            types = ', '.join(set(this_ch_type))
            raise ValueError('Cannot combine sensors of different types; '
                             f'"{this_group}" contains types {types}.')
        # Remove bad channels
        these_bads = [idx for idx in this_picks if idx in ch_idx_bad]
        this_picks = [idx for idx in this_picks if idx not in ch_idx_bad]
        if these_bads:
            logger.info('Dropped the following channels in group '
                        f'{this_group}: {these_bads}')
        #  Check if combining less than 2 channel
        if len(set(this_picks)) < 2:
            warn(f'Less than 2 channels in group "{this_group}" when '
                 f'combining by method "{method}".')
        # If all good create more detailed dict without bad channels
        groups[this_group] = dict(picks=this_picks, ch_type=this_ch_type[0])

    # Combine channels and add them to the new instance
    for this_group, this_group_dict in groups.items():
        new_ch_names.append(this_group)
        new_ch_types.append(this_group_dict['ch_type'])
        this_picks = this_group_dict['picks']
        this_data = np.take(inst_data, this_picks, axis=ch_axis)
        new_data.append(method(this_data))
    new_data = np.swapaxes(new_data, 0, ch_axis)
    info = create_info(sfreq=inst.info['sfreq'], ch_names=new_ch_names,
                       ch_types=new_ch_types)
    # create new instances and make sure to copy important attributes
    if isinstance(inst, BaseRaw):
        combined_inst = RawArray(new_data, info, first_samp=inst.first_samp)
    elif isinstance(inst, BaseEpochs):
        combined_inst = EpochsArray(new_data, info, events=inst.events,
                                    tmin=inst.times[0], baseline=inst.baseline)
        if inst.metadata is not None:
            combined_inst.metadata = inst.metadata.copy()
    elif isinstance(inst, Evoked):
        combined_inst = EvokedArray(new_data, info, tmin=inst.times[0],
                                    baseline=inst.baseline)

    return combined_inst


# NeuroMag channel groupings
_SELECTIONS = ['Vertex', 'Left-temporal', 'Right-temporal', 'Left-parietal',
               'Right-parietal', 'Left-occipital', 'Right-occipital',
               'Left-frontal', 'Right-frontal']
_EEG_SELECTIONS = ['EEG 1-32', 'EEG 33-64', 'EEG 65-96', 'EEG 97-128']


def _divide_to_regions(info, add_stim=True):
    """Divide channels to regions by positions."""
    from scipy.stats import zscore
    picks = _pick_data_channels(info, exclude=[])
    chs_in_lobe = len(picks) // 4
    pos = np.array([ch['loc'][:3] for ch in info['chs']])
    x, y, z = pos.T

    frontal = picks[np.argsort(y[picks])[-chs_in_lobe:]]
    picks = np.setdiff1d(picks, frontal)

    occipital = picks[np.argsort(y[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, occipital)

    temporal = picks[np.argsort(z[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, temporal)

    lt, rt = _divide_side(temporal, x)
    lf, rf = _divide_side(frontal, x)
    lo, ro = _divide_side(occipital, x)
    lp, rp = _divide_side(picks, x)  # Parietal lobe from the remaining picks.

    # Because of the way the sides are divided, there may be outliers in the
    # temporal lobes. Here we switch the sides for these outliers. For other
    # lobes it is not a big problem because of the vicinity of the lobes.
    with np.errstate(invalid='ignore'):  # invalid division, greater compare
        zs = np.abs(zscore(x[rt]))
        outliers = np.array(rt)[np.where(zs > 2.)[0]]
    rt = list(np.setdiff1d(rt, outliers))

    with np.errstate(invalid='ignore'):  # invalid division, greater compare
        zs = np.abs(zscore(x[lt]))
        outliers = np.append(outliers, (np.array(lt)[np.where(zs > 2.)[0]]))
    lt = list(np.setdiff1d(lt, outliers))

    l_mean = np.mean(x[lt])
    r_mean = np.mean(x[rt])
    for outlier in outliers:
        if abs(l_mean - x[outlier]) < abs(r_mean - x[outlier]):
            lt.append(outlier)
        else:
            rt.append(outlier)

    if add_stim:
        stim_ch = _get_stim_channel(None, info, raise_error=False)
        if len(stim_ch) > 0:
            for region in [lf, rf, lo, ro, lp, rp, lt, rt]:
                region.append(info['ch_names'].index(stim_ch[0]))
    return OrderedDict([('Left-frontal', lf), ('Right-frontal', rf),
                        ('Left-parietal', lp), ('Right-parietal', rp),
                        ('Left-occipital', lo), ('Right-occipital', ro),
                        ('Left-temporal', lt), ('Right-temporal', rt)])


def _divide_side(lobe, x):
    """Make a separation between left and right lobe evenly."""
    lobe = np.asarray(lobe)
    median = np.median(x[lobe])

    left = lobe[np.where(x[lobe] < median)[0]]
    right = lobe[np.where(x[lobe] > median)[0]]
    medians = np.where(x[lobe] == median)[0]

    left = np.sort(np.concatenate([left, lobe[medians[1::2]]]))
    right = np.sort(np.concatenate([right, lobe[medians[::2]]]))
    return list(left), list(right)


@verbose
def read_vectorview_selection(name, fname=None, info=None, verbose=None):
    """Read Neuromag Vector View channel selection from a file.

    Parameters
    ----------
    name : str | list of str
        Name of the selection. If a list, the selections are combined.
        Supported selections are: ``'Vertex'``, ``'Left-temporal'``,
        ``'Right-temporal'``, ``'Left-parietal'``, ``'Right-parietal'``,
        ``'Left-occipital'``, ``'Right-occipital'``, ``'Left-frontal'`` and
        ``'Right-frontal'``. Selections can also be matched and combined by
        spcecifying common substrings. For example, ``name='temporal`` will
        produce a combination of ``'Left-temporal'`` and ``'Right-temporal'``.
    fname : str
        Filename of the selection file (if ``None``, built-in selections are
        used).
    %(info)s Used to determine which channel naming convention to use, e.g.
        ``'MEG 0111'`` (with space) for old Neuromag systems and ``'MEG0111'``
        (without space) for new ones.
    %(verbose)s

    Returns
    -------
    sel : list of str
        List with channel names in the selection.
    """
    # convert name to list of string
    if not isinstance(name, (list, tuple)):
        name = [name]
    if isinstance(info, Info):
        picks = pick_types(info, meg=True, exclude=())
        if len(picks) > 0 and ' ' not in info['ch_names'][picks[0]]:
            spacing = 'new'
        else:
            spacing = 'old'
    elif info is not None:
        raise TypeError('info must be an instance of Info or None, not %s'
                        % (type(info),))
    else:  # info is None
        spacing = 'old'

    # use built-in selections by default
    if fname is None:
        fname = op.join(op.dirname(__file__), '..', 'data', 'mne_analyze.sel')

    fname = _check_fname(fname, must_exist=True, overwrite='read')

    # use this to make sure we find at least one match for each name
    name_found = {n: False for n in name}
    with open(fname, 'r') as fid:
        sel = []
        for line in fid:
            line = line.strip()
            # skip blank lines and comments
            if len(line) == 0 or line[0] == '#':
                continue
            # get the name of the selection in the file
            pos = line.find(':')
            if pos < 0:
                logger.info('":" delimiter not found in selections file, '
                            'skipping line')
                continue
            sel_name_file = line[:pos]
            # search for substring match with name provided
            for n in name:
                if sel_name_file.find(n) >= 0:
                    sel.extend(line[pos + 1:].split('|'))
                    name_found[n] = True
                    break

    # make sure we found at least one match for each name
    for n, found in name_found.items():
        if not found:
            raise ValueError('No match for selection name "%s" found' % n)

    # make the selection a sorted list with unique elements
    sel = list(set(sel))
    sel.sort()
    if spacing == 'new':  # "new" or "old" by now, "old" is default
        sel = [s.replace('MEG ', 'MEG') for s in sel]
    return sel
