# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import sys

import numpy as np
from scipy import sparse

from ..defaults import HEAD_SIZE_DEFAULT
from ..utils import (verbose, logger, warn, copy_function_doc_to_method_doc,
                     _check_preload, _validate_type, fill_doc, _check_option)
from ..io.compensator import get_current_comp
from ..io.constants import FIFF
from ..io.meas_info import anonymize_info, Info, MontageMixin
from ..io.pick import (channel_type, pick_info, pick_types, _picks_by_type,
                       _check_excludes_includes, _contains_ch_type,
                       channel_indices_by_type, pick_channels, _picks_to_idx,
                       _get_channel_types)


def _get_meg_system(info):
    """Educated guess for the helmet type based on channels."""
    have_helmet = True
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
                         'fnirs_raw', 'fnirs_od', 'hbo', 'hbr', 'ecog', 'seeg']
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

    @fill_doc
    def get_channel_types(self, picks=None, unique=False, only_data_chs=False):
        """Get a list of channel type for each channel.

        Parameters
        ----------
        %(picks_all)s
        unique : bool
            Whether to return only unique channel types. Default is ``False``.
        only_data_chs : bool
            Whether to ignore non-data channels. Default is ``False``.

        Returns
        -------
        channel_types : list
            The channel types.
        """
        return _get_channel_types(self.info, picks=picks, unique=unique,
                                  only_data_chs=only_data_chs)


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
               'fnirs_raw': FIFF.FIFFV_FNIRS_CH,
               'fnirs_od': FIFF.FIFFV_FNIRS_CH,
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
               'fnirs_raw': FIFF.FIFF_UNIT_V,
               'fnirs_od': FIFF.FIFF_UNIT_NONE,
               'hbo': FIFF.FIFF_UNIT_MOL,
               'hbr': FIFF.FIFF_UNIT_MOL}
_unit2human = {FIFF.FIFF_UNIT_V: 'V',
               FIFF.FIFF_UNIT_T: 'T',
               FIFF.FIFF_UNIT_T_M: 'T/m',
               FIFF.FIFF_UNIT_MOL: 'M',
               FIFF.FIFF_UNIT_NONE: 'NA',
               FIFF.FIFF_UNIT_CEL: 'C'}


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
                          ch_type='auto', verbose=None):
        """Specify which reference to use for EEG data.

        Use this function to explicitly specify the desired reference for EEG.
        This can be either an existing electrode or a new virtual channel.
        This function will re-reference the data according to the desired
        reference.

        Parameters
        ----------
        ref_channels : list of str | str
            The name(s) of the channel(s) used to construct the reference. To
            apply an average reference, specify ``'average'`` here (default).
            If an empty list is specified, the data is assumed to already have
            a proper reference and MNE will not attempt any re-referencing of
            the data. Defaults to an average reference.
        projection : bool
            If ``ref_channels='average'`` this argument specifies if the
            average reference should be computed as a projection (True) or not
            (False; default). If ``projection=True``, the average reference is
            added as a projection and is not applied to the data (it can be
            applied afterwards with the ``apply_proj`` method). If
            ``projection=False``, the average reference is directly applied to
            the data. If ``ref_channels`` is not ``'average'``, ``projection``
            must be set to ``False`` (the default in this case).
        ch_type : 'auto' | 'eeg' | 'ecog' | 'seeg'
            The name of the channel type to apply the reference to. If 'auto',
            the first channel type of eeg, ecog or seeg that is found (in that
            order) will be selected.

            .. versionadded:: 0.19
        %(verbose_meth)s

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
                                 projection=projection, ch_type=ch_type)[0]

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

    @verbose
    def set_channel_types(self, mapping, verbose=None):
        """Define the sensor type of channels.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping a channel to a sensor type (str), e.g.,
            ``{'EEG061': 'eog'}``.
        %(verbose_meth)s

        Returns
        -------
        inst : instance of Raw | Epochs | Evoked
            The instance (modified in place).

            .. versionchanged:: 0.20
               Return the instance.

        Notes
        -----
        The following sensor types are accepted:

            ecg, eeg, emg, eog, exci, ias, misc, resp, seeg, stim, syst, ecog,
            hbo, hbr, fnirs_raw, fnirs_od

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
            elif ch_type == 'fnirs_raw':
                coil_type = FIFF.FIFFV_COIL_FNIRS_RAW
            elif ch_type == 'fnirs_od':
                coil_type = FIFF.FIFFV_COIL_FNIRS_OD
            else:
                coil_type = FIFF.FIFFV_COIL_NONE
            self.info['chs'][c_ind]['coil_type'] = coil_type
        msg = "The unit for channel(s) {0} has changed from {1} to {2}."
        for this_change, names in unit_changes.items():
            warn(msg.format(", ".join(sorted(names)), *this_change))
        return self

    def rename_channels(self, mapping):
        """Rename channels.

        Parameters
        ----------
        mapping : dict | callable
            A dictionary mapping the old channel to a new channel name
            e.g. {'EEG061' : 'EEG161'}. Can also be a callable function
            that takes and returns a string (new in version 0.10.0).

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
        rename_channels(self.info, mapping)
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
            'seeg', 'ecog', 'all'. If ``'all'``, all the available mag, grad,
            eeg, seeg and ecog channels are plotted. If None (default), then
            channels are chosen in the order given above.
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
        %(topomap_sphere_auto)s
        %(verbose_meth)s

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
        matplotlib. For drawing the sensors using mayavi see
        :func:`mne.viz.plot_alignment`.

        .. versionadded:: 0.12.0
        """
        from ..viz.utils import plot_sensors
        return plot_sensors(self.info, kind=kind, ch_type=ch_type, title=title,
                            show_names=show_names, ch_groups=ch_groups,
                            to_sphere=to_sphere, axes=axes, block=block,
                            show=show, sphere=sphere, verbose=verbose)

    @copy_function_doc_to_method_doc(anonymize_info)
    def anonymize(self, daysback=None, keep_his=False, verbose=None):
        """
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
        self.info['meas_date'] = meas_date
        if hasattr(self, 'annotations'):
            self.annotations._orig_time = meas_date
        return self


class UpdateChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs, AverageTFR."""

    @verbose
    def pick_types(self, meg=True, eeg=False, stim=False, eog=False,
                   ecg=False, emg=False, ref_meg='auto', misc=False,
                   resp=False, chpi=False, exci=False, ias=False, syst=False,
                   seeg=False, dipole=False, gof=False, bio=False, ecog=False,
                   fnirs=False, csd=False, include=(), exclude='bads',
                   selection=None, verbose=None):
        """Pick some channels by type and names.

        Parameters
        ----------
        meg : bool | str
            If True include all MEG channels. If False include None.
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
        ref_meg : bool | str
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
        csd : bool
            EEG-CSD channels.
        include : list of str
            List of additional channels to include. If empty do not include
            any.
        exclude : list of str | str
            List of channels to exclude. If 'bads' (default), exclude channels
            in ``info['bads']``.
        selection : list of str
            Restrict sensor channels (MEG, EEG) to this list of channel names.
        %(verbose_meth)s

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
            ecog=ecog, fnirs=fnirs, include=include, exclude=exclude,
            selection=selection)
        return self._pick_drop_channels(idx)

    def pick_channels(self, ch_names, ordered=False):
        """Pick some channels.

        Parameters
        ----------
        ch_names : list
            The list of channels to select.
        ordered : bool
            If True (default False), ensure that the order of the channels in
            the modified instance matches the order of ``ch_names``.

            .. versionadded:: 0.20.0

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
        return self._pick_drop_channels(
            pick_channels(self.info['ch_names'], ch_names, ordered=ordered))

    @fill_doc
    def pick(self, picks, exclude=()):
        """Pick a subset of channels.

        Parameters
        ----------
        %(picks_all)s
        exclude : list | str
            Set of channels to exclude, only used when picking based on
            types (e.g., exclude="bads" when picks="meg").

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

    def drop_channels(self, ch_names):
        """Drop channel(s).

        Parameters
        ----------
        ch_names : iterable or str
            Iterable (e.g. list) of channel name(s) or channel name to remove.

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
            raise ValueError(msg.format(", ".join(missing)))

        bad_idx = [self.ch_names.index(ch) for ch in ch_names
                   if ch in self.ch_names]
        idx = np.setdiff1d(np.arange(len(self.ch_names)), bad_idx)
        return self._pick_drop_channels(idx)

    def _pick_drop_channels(self, idx):
        # avoid circular imports
        from ..time_frequency import AverageTFR, EpochsTFR

        _check_preload(self, 'adding, dropping, or reordering channels')

        if getattr(self, 'picks', None) is not None:
            self.picks = self.picks[idx]

        if hasattr(self, '_cals'):
            self._cals = self._cals[idx]

        pick_info(self.info, idx, copy=False)

        if getattr(self, '_projector', None) is not None:
            self._projector = self._projector[idx][:, idx]

        # All others (Evoked, Epochs, Raw) have chs axis=-2
        axis = -3 if isinstance(self, (AverageTFR, EpochsTFR)) else -2
        self._data = self._data.take(idx, axis=axis)
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
            values in `self`. This should generally only be used when adding
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
                raise AssertionError('All data dimensions except channels '
                                     'must match, got %s != %s'
                                     % (shapes[0], shape))
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
        return self


class InterpolationMixin(object):
    """Mixin class for Raw, Evoked, Epochs."""

    @verbose
    def interpolate_bads(self, reset_bads=True, mode='accurate',
                         origin='auto', verbose=None):
        """Interpolate bad MEG and EEG channels.

        Operates in place.

        Parameters
        ----------
        reset_bads : bool
            If True, remove the bads from info.
        mode : str
            Either ``'accurate'`` or ``'fast'``, determines the quality of the
            Legendre polynomial expansion used for interpolation of MEG
            channels.
        origin : array-like, shape (3,) | str
            Origin of the sphere in the head coordinate frame and in meters.
            Can be ``'auto'`` (default), which means a head-digitization-based
            origin fit.

            .. versionadded:: 0.17
        %(verbose_meth)s

        Returns
        -------
        inst : instance of Raw, Epochs, or Evoked
            The modified instance.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        from ..bem import _check_origin
        from .interpolation import _interpolate_bads_eeg, _interpolate_bads_meg

        _check_preload(self, "interpolation")

        if len(self.info['bads']) == 0:
            warn('No bad channels to interpolate. Doing nothing...')
            return self
        origin = _check_origin(origin, self.info)
        _interpolate_bads_eeg(self, origin=origin)
        _interpolate_bads_meg(self, mode=mode, origin=origin)

        if reset_bads is True:
            self.info['bads'] = []

        return self


def rename_channels(info, mapping):
    """Rename channels.

    .. warning::  The channel names must have at most 15 characters

    Parameters
    ----------
    info : dict
        Measurement info.
    mapping : dict | callable
        A dictionary mapping the old channel to a new channel name
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
    for new_name in new_names:
        _validate_type(new_name[1], 'str', 'New channel mappings')

    bad_new_names = [name for _, name in new_names if len(name) > 15]
    if len(bad_new_names):
        raise ValueError('Channel names cannot be longer than 15 '
                         'characters. These channel names are not '
                         'valid : %s' % new_names)

    # do the remapping locally
    for c_ind, new_name in new_names:
        for bi, bad in enumerate(bads):
            if bad == ch_names[c_ind]:
                bads[bi] = new_name
        ch_names[c_ind] = new_name

    # check that all the channel names are unique
    if len(ch_names) != len(np.unique(ch_names)):
        raise ValueError('New channel names are not unique, renaming failed')

    # do the remapping in info
    info['bads'] = bads
    for ch, ch_name in zip(info['chs'], ch_names):
        ch['ch_name'] = ch_name
    info._update_redundant()
    info._check_consistency()


def _recursive_flatten(cell, dtype):
    """Unpack mat files in Python."""
    if len(cell) > 0:
        while not isinstance(cell[0], dtype):
            cell = [c for d in cell for c in d]
    return cell


@fill_doc
def read_ch_connectivity(fname, picks=None):
    """Parse FieldTrip neighbors .mat file.

    More information on these neighbor definitions can be found on the related
    `FieldTrip documentation pages
    <http://www.fieldtrip.org/template/neighbours>`__.

    Parameters
    ----------
    fname : str
        The file name. Example: 'neuromag306mag', 'neuromag306planar',
        'ctf275', 'biosemi64', etc.
    %(picks_all)s
        Picks Must match the template.

    Returns
    -------
    ch_connectivity : scipy.sparse.csr_matrix, shape (n_channels, n_channels)
        The connectivity matrix.
    ch_names : list
        The list of channel names present in connectivity matrix.

    See Also
    --------
    find_ch_connectivity

    Notes
    -----
    This function is closely related to :func:`find_ch_connectivity`. If you
    don't know the correct file for the neighbor definitions,
    :func:`find_ch_connectivity` can compute the connectivity matrix from 2d
    sensor locations.
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
    ch_names = _recursive_flatten(nb['label'], str)
    picks = _picks_to_idx(len(ch_names), picks)
    neighbors = [_recursive_flatten(c, str) for c in
                 nb['neighblabel'].flatten()]
    assert len(ch_names) == len(neighbors)
    connectivity = _ch_neighbor_connectivity(ch_names, neighbors)
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
    set_neighbors = {c for d in neighbors for c in d}
    rest = set_neighbors - set(ch_names)
    if len(rest) > 0:
        raise ValueError('Some of your neighbors are not present in the '
                         'list of channel names')

    for neigh in neighbors:
        if (not isinstance(neigh, list) and
                not all(isinstance(c, str) for c in neigh)):
            raise ValueError('`neighbors` must be a list of lists of str')

    ch_connectivity = np.eye(len(ch_names), dtype=bool)
    for ii, neigbs in enumerate(neighbors):
        ch_connectivity[ii, [ch_names.index(i) for i in neigbs]] = True
    ch_connectivity = sparse.csr_matrix(ch_connectivity)
    return ch_connectivity


def find_ch_connectivity(info, ch_type):
    """Find the connectivity matrix for the given channels.

    This function tries to infer the appropriate connectivity matrix template
    for the given channels. If a template is not found, the connectivity matrix
    is computed using Delaunay triangulation based on 2d sensor locations.

    Parameters
    ----------
    info : instance of Info
        The measurement info.
    ch_type : str | None
        The channel type for computing the connectivity matrix. Currently
        supports 'mag', 'grad', 'eeg' and None. If None, the info must contain
        only one channel type.

    Returns
    -------
    ch_connectivity : scipy.sparse.csr_matrix, shape (n_channels, n_channels)
        The connectivity matrix.
    ch_names : list
        The list of channel names present in connectivity matrix.

    See Also
    --------
    read_ch_connectivity

    Notes
    -----
    .. versionadded:: 0.15

    Automatic detection of an appropriate connectivity matrix template only
    works for MEG data at the moment. This means that the connectivity matrix
    is always computed for EEG data and never loaded from a template file. If
    you want to load a template for a given montage use
    :func:`read_ch_connectivity` directly.
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
     has_neuromag_122_grad) = _get_ch_info(info)
    conn_name = None
    if has_vv_mag and ch_type == 'mag':
        conn_name = 'neuromag306mag'
    elif has_vv_grad and ch_type == 'grad':
        conn_name = 'neuromag306planar'
    elif has_neuromag_122_grad:
        conn_name = 'neuromag122'
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
        logger.info('Reading connectivity matrix for %s.' % conn_name)
        return read_ch_connectivity(conn_name)
    logger.info('Could not find a connectivity matrix for the data. '
                'Computing connectivity based on Delaunay triangulations.')
    return _compute_ch_connectivity(info, ch_type)


def _compute_ch_connectivity(info, ch_type):
    """Compute channel connectivity matrix using Delaunay triangulations.

    Parameters
    ----------
    info : instance of mne.measuerment_info.Info
        The measurement info.
    ch_type : str
        The channel type for computing the connectivity matrix. Currently
        supports 'mag', 'grad' and 'eeg'.

    Returns
    -------
    ch_connectivity : scipy.sparse matrix, shape (n_channels, n_channels)
        The connectivity matrix.
    ch_names : list
        The list of channel names present in connectivity matrix.
    """
    from scipy.spatial import Delaunay
    from .. import spatial_tris_connectivity
    from ..channels.layout import _find_topomap_coords, _pair_grad_sensors
    combine_grads = (ch_type == 'grad' and FIFF.FIFFV_COIL_VV_PLANAR_T1 in
                     np.unique([ch['coil_type'] for ch in info['chs']]))

    picks = dict(_picks_by_type(info, exclude=[]))[ch_type]
    ch_names = [info['ch_names'][pick] for pick in picks]
    if combine_grads:
        pairs = _pair_grad_sensors(info, topomap_coords=False, exclude=[])
        if len(pairs) != len(picks):
            raise RuntimeError('Cannot find a pair for some of the '
                               'gradiometers. Cannot compute connectivity '
                               'matrix.')
        # only for one of the pair
        xy = _find_topomap_coords(info, picks[::2], sphere=HEAD_SIZE_DEFAULT)
    else:
        xy = _find_topomap_coords(info, picks, sphere=HEAD_SIZE_DEFAULT)
    tri = Delaunay(xy)
    neighbors = spatial_tris_connectivity(tri.simplices)

    if combine_grads:
        ch_connectivity = np.eye(len(picks), dtype=bool)
        for idx, neigbs in zip(neighbors.row, neighbors.col):
            for ii in range(2):  # make sure each pair is included
                for jj in range(2):
                    ch_connectivity[idx * 2 + ii, neigbs * 2 + jj] = True
                    ch_connectivity[idx * 2 + ii, idx * 2 + jj] = True  # pair
        ch_connectivity = sparse.csr_matrix(ch_connectivity)
    else:
        ch_connectivity = sparse.lil_matrix(neighbors)
        ch_connectivity.setdiag(np.repeat(1, ch_connectivity.shape[0]))
        ch_connectivity = ch_connectivity.tocsr()

    return ch_connectivity, ch_names


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
              Therefore the use of ``fix_mag_coil_types`` is not mandatory.
    """
    old_mag_inds = _get_T1T2_mag_inds(info)

    for ii in old_mag_inds:
        info['chs'][ii]['coil_type'] = FIFF.FIFFV_COIL_VV_MAG_T3
    logger.info('%d of %d T1/T2 magnetometer types replaced with T3.' %
                (len(old_mag_inds), len(pick_types(info, meg='mag'))))
    info._check_consistency()


def _get_T1T2_mag_inds(info):
    """Find T1/T2 magnetometer coil types."""
    picks = pick_types(info, meg='mag')
    old_mag_inds = []
    for ii in picks:
        ch = info['chs'][ii]
        if ch['coil_type'] in (FIFF.FIFFV_COIL_VV_MAG_T1,
                               FIFF.FIFFV_COIL_VV_MAG_T2):
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

    return (has_vv_mag, has_vv_grad, is_old_vv, has_4D_mag, ctf_other_types,
            has_CTF_grad, n_kit_grads, has_any_meg, has_eeg_coils,
            has_eeg_coils_and_meg, has_eeg_coils_only, has_neuromag_122_grad)


def make_1020_channel_selections(info, midline="z"):
    """Return dict mapping from ROI names to lists of picks for 10/20 setups.

    This passes through all channel names, and uses a simple heuristic to
    separate channel names into three Region of Interest-based selections:
    Left, Midline and Right. The heuristic is that channels ending on any of
    the characters in `midline` are filed under that heading, otherwise those
    ending in odd numbers under "Left", those in even numbers under "Right".
    Other channels are ignored. This is appropriate for 10/20 files, but not
    for other channel naming conventions.
    If an info object is provided, lists are sorted from posterior to anterior.

    Parameters
    ----------
    info : instance of Info
        Where to obtain the channel names from. The picks will
        be in relation to the position in `info["ch_names"]`. If possible, this
        lists will be sorted by y value position of the channel locations,
        i.e., from back to front.
    midline : str
        Names ending in any of these characters are stored under the `Midline`
        key. Defaults to 'z'. Note that capitalization is ignored.

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
