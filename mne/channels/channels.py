# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <denis.engemann@gmail.com>
#          Andrew Dykstra <andrew.r.dykstra@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import warnings

import numpy as np
from scipy import sparse

from ..externals.six import string_types

from ..utils import verbose, logger
from ..io.pick import (channel_type, pick_info, pick_types,
                       _check_excludes_includes)
from ..io.constants import FIFF


def _get_meg_system(info):
    """Educated guess for the helmet type based on channels"""
    system = '306m'
    for ch in info['chs']:
        if ch['kind'] == FIFF.FIFFV_MEG_CH:
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
    """Check whether a certain channel type is in an info object

    Parameters
    ---------
    info : instance of mne.io.meas_info.Info
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

    valid_channel_types = ['grad', 'mag', 'planar1', 'planar2', 'eeg', 'stim',
                           'eog', 'emg', 'ecg', 'ref_meg', 'resp', 'exci',
                           'ias', 'syst', 'seeg', 'misc']

    if ch_type not in valid_channel_types:
        raise ValueError('ch_type must be one of %s, not "%s"'
                         % (valid_channel_types, ch_type))
    if info is None:
        raise ValueError('Cannot check for channels of type "%s" because info '
                         'is None' % (ch_type,))
    return ch_type in [channel_type(info, ii) for ii in range(info['nchan'])]


def _get_ch_type(inst, ch_type):
    """Helper to choose a single channel type (usually for plotting)

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
    """Equalize channel picks for a collection of MNE-Python objects

    Parameters
    ----------
    candidates : list
        list Raw | Epochs | Evoked.
    verbose : None | bool
        whether to be verbose or not.

    Notes
    -----
    This function operates inplace.
    """
    from ..io.base import _BaseRaw
    from ..epochs import _BaseEpochs
    from ..evoked import Evoked
    from ..time_frequency import AverageTFR

    if not all(isinstance(c, (_BaseRaw, _BaseEpochs, Evoked, AverageTFR))
               for c in candidates):
        valid = ['Raw', 'Epochs', 'Evoked', 'AverageTFR']
        raise ValueError('candidates must be ' + ' or '.join(valid))

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
    """Mixin class for Raw, Evoked, Epochs
    """
    def __contains__(self, ch_type):
        """Check channel type membership"""
        if ch_type == 'meg':
            has_ch_type = (_contains_ch_type(self.info, 'mag') or
                           _contains_ch_type(self.info, 'grad'))
        else:
            has_ch_type = _contains_ch_type(self.info, ch_type)
        return has_ch_type


class SetChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs
    """
    def _get_channel_positions(self, picks=None):
        """Gets channel locations from info

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
        """Update channel locations in info

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
            ecg, eeg, emg, eog, exci, ias, misc, resp, seeg, stim, syst

        Parameters
        ----------
        mapping : dict
            a dictionary mapping a channel to a sensor type (str)
            {'EEG061': 'eog'}.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        human2fiff = {'ecg': FIFF.FIFFV_ECG_CH,
                      'eeg': FIFF.FIFFV_EEG_CH,
                      'emg': FIFF.FIFFV_EMG_CH,
                      'eog': FIFF.FIFFV_EOG_CH,
                      'exci': FIFF.FIFFV_EXCI_CH,
                      'ias': FIFF.FIFFV_IAS_CH,
                      'misc': FIFF.FIFFV_MISC_CH,
                      'resp': FIFF.FIFFV_RESP_CH,
                      'seeg': FIFF.FIFFV_SEEG_CH,
                      'stim': FIFF.FIFFV_STIM_CH,
                      'syst': FIFF.FIFFV_SYST_CH}

        human2unit = {'ecg': FIFF.FIFF_UNIT_V,
                      'eeg': FIFF.FIFF_UNIT_V,
                      'emg': FIFF.FIFF_UNIT_V,
                      'eog': FIFF.FIFF_UNIT_V,
                      'exci': FIFF.FIFF_UNIT_NONE,
                      'ias': FIFF.FIFF_UNIT_NONE,
                      'misc': FIFF.FIFF_UNIT_V,
                      'resp': FIFF.FIFF_UNIT_NONE,
                      'seeg': FIFF.FIFF_UNIT_V,
                      'stim': FIFF.FIFF_UNIT_NONE,
                      'syst': FIFF.FIFF_UNIT_NONE}

        unit2human = {FIFF.FIFF_UNIT_V: 'V',
                      FIFF.FIFF_UNIT_NONE: 'NA'}
        ch_names = self.info['ch_names']

        # first check and assemble clean mappings of index and name
        for ch_name, ch_type in mapping.items():
            if ch_name not in ch_names:
                raise ValueError("This channel name (%s) doesn't exist in "
                                 "info." % ch_name)

            c_ind = ch_names.index(ch_name)
            if ch_type not in human2fiff:
                raise ValueError('This function cannot change to this '
                                 'channel type: %s. Accepted channel types '
                                 'are %s.' % (ch_type,
                                              ", ".join(human2unit.keys())))
            # Set sensor type
            self.info['chs'][c_ind]['kind'] = human2fiff[ch_type]
            unit_old = self.info['chs'][c_ind]['unit']
            unit_new = human2unit[ch_type]
            if unit_old != human2unit[ch_type]:
                warnings.warn("The unit for Channel %s has changed "
                              "from %s to %s." % (ch_name,
                                                  unit2human[unit_old],
                                                  unit2human[unit_new]))
            self.info['chs'][c_ind]['unit'] = human2unit[ch_type]
            if ch_type in ['eeg', 'seeg']:
                self.info['chs'][c_ind]['coil_type'] = FIFF.FIFFV_COIL_EEG
            else:
                self.info['chs'][c_ind]['coil_type'] = FIFF.FIFFV_COIL_NONE

    def rename_channels(self, mapping):
        """Rename channels.

        Parameters
        ----------
        mapping : dict
            a dictionary mapping the old channel to a new channel name
            e.g. {'EEG061' : 'EEG161'}.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        rename_channels(self.info, mapping)

    def set_montage(self, montage):
        """Set EEG sensor configuration

        Parameters
        ----------
        montage : instance of Montage or DigMontage

        Notes
        -----
        Operates in place.

        .. versionadded:: 0.9.0
        """
        from .montage import _set_montage
        _set_montage(self.info, montage)


class UpdateChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs, AverageTFR
    """
    def pick_types(self, meg=True, eeg=False, stim=False, eog=False,
                   ecg=False, emg=False, ref_meg='auto', misc=False,
                   resp=False, chpi=False, exci=False, ias=False, syst=False,
                   seeg=False, include=[], exclude='bads', selection=None,
                   copy=False):
        """Pick some channels by type and names

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
        include : list of string
            List of additional channels to include. If empty do not include
            any.
        exclude : list of string | str
            List of channels to exclude. If 'bads' (default), exclude channels
            in ``info['bads']``.
        selection : list of string
            Restrict sensor channels (MEG, EEG) to this list of channel names.
        copy : bool
            If True, returns new instance. Else, modifies in place. Defaults to
            False.

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        inst = self.copy() if copy else self
        idx = pick_types(
            self.info, meg=meg, eeg=eeg, stim=stim, eog=eog, ecg=ecg, emg=emg,
            ref_meg=ref_meg, misc=misc, resp=resp, chpi=chpi, exci=exci,
            ias=ias, syst=syst, seeg=seeg, include=include, exclude=exclude,
            selection=selection)
        inst._pick_drop_channels(idx)
        return inst

    def pick_channels(self, ch_names, copy=False):
        """Pick some channels

        Parameters
        ----------
        ch_names : list
            The list of channels to select.
        copy : bool
            If True, returns new instance. Else, modifies in place. Defaults to
            False.

        See Also
        --------
        drop_channels

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        inst = self.copy() if copy else self
        _check_excludes_includes(ch_names)

        idx = [inst.ch_names.index(c) for c in ch_names if c in inst.ch_names]
        inst._pick_drop_channels(idx)

        return inst

    def drop_channels(self, ch_names, copy=False):
        """Drop some channels

        Parameters
        ----------
        ch_names : list
            The list of channels to remove.
        copy : bool
            If True, returns new instance. Else, modifies in place. Defaults to
            False.

        See Also
        --------
        pick_channels

        Notes
        -----
        .. versionadded:: 0.9.0
        """
        inst = self.copy() if copy else self

        bad_idx = [inst.ch_names.index(c) for c in ch_names
                   if c in inst.ch_names]
        idx = np.setdiff1d(np.arange(len(inst.ch_names)), bad_idx)
        inst._pick_drop_channels(idx)

        return inst

    def _pick_drop_channels(self, idx):
        # avoid circular imports
        from ..io.base import _BaseRaw
        from ..epochs import _BaseEpochs
        from ..evoked import Evoked
        from ..time_frequency import AverageTFR

        if isinstance(self, (_BaseRaw, _BaseEpochs)):
            if not self.preload:
                raise RuntimeError('If Raw or Epochs, data must be preloaded '
                                   'to drop or pick channels')

        def inst_has(attr):
            return getattr(self, attr, None) is not None

        if inst_has('picks'):
            self.picks = self.picks[idx]

        if inst_has('_cals'):
            self._cals = self._cals[idx]

        self.info = pick_info(self.info, idx, copy=False)

        if inst_has('_projector'):
            self._projector = self._projector[idx][:, idx]

        if isinstance(self, _BaseRaw) and inst_has('_data'):
            self._data = self._data.take(idx, axis=0)
        elif isinstance(self, _BaseEpochs) and inst_has('_data'):
            self._data = self._data.take(idx, axis=1)
        elif isinstance(self, AverageTFR) and inst_has('data'):
            self.data = self.data.take(idx, axis=0)
        elif isinstance(self, Evoked):
            self.data = self.data.take(idx, axis=0)

    def add_channels(self, add_list, copy=False):
        """Append new channels to the instance.

        Parameters
        ----------
        add_list : list
            A list of objects to append to self. Must contain all the same
            type as the current object
        copy : bool
            Whether to return a new instance or modify in place

        Returns
        -------
        out : MNE object of type(self)
            An object with new channels appended (will be the same
            object if copy==False)
        """
        # avoid circular imports
        from ..io.base import _BaseRaw
        from ..epochs import _BaseEpochs
        from ..io.meas_info import _merge_info

        if not isinstance(add_list, (list, tuple)):
            raise AssertionError('Input must be a list or tuple of objs')

        # Object-specific checks
        if isinstance(self, (_BaseRaw, _BaseEpochs)):
            if not all([inst.preload for inst in add_list] + [self.preload]):
                raise AssertionError('All data must be preloaded')
            data_name = '_data'
            if isinstance(self, _BaseRaw):
                con_axis = 0
                comp_class = _BaseRaw
            elif isinstance(self, _BaseEpochs):
                con_axis = 1
                comp_class = _BaseEpochs
        else:
            data_name = 'data'
            con_axis = 0
            comp_class = type(self)
        if not all(isinstance(inst, comp_class) for inst in add_list):
            raise AssertionError('All input data must be of same type')
        data = [getattr(self, data_name)] + [getattr(inst, data_name)
                                             for inst in add_list]

        # Make sure that all dimensions other than channel axis are the same
        compare_axes = [i for i in range(data[0].ndim) if i != con_axis]
        shapes = np.array([dat.shape for dat in data])[:, compare_axes]
        if not ((shapes[0] - shapes) == 0).all():
            raise AssertionError('All dimensions except channels must match')

        # Create final data / info objects
        data = np.concatenate(data, axis=con_axis)
        infos = [self.info] + [inst.info for inst in add_list]
        new_info = _merge_info(infos)

        # Now update the attributes
        if copy is True:
            out = self.copy()
        else:
            out = self
        setattr(out, data_name, data)
        out.info = new_info
        return out


class InterpolationMixin(object):
    """Mixin class for Raw, Evoked, Epochs
    """

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
        self : mne.io.Raw, mne.Epochs or mne.Evoked
            The interpolated data.

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
    mapping : dict
        a dictionary mapping the old channel to a new channel name
        e.g. {'EEG061' : 'EEG161'}.
    """
    bads, chs = info['bads'], info['chs']
    ch_names = info['ch_names']
    new_names, new_kinds, new_bads = list(), list(), list()

    # first check and assemble clean mappings of index and name
    for ch_name, new_name in mapping.items():
        if ch_name not in ch_names:
            raise ValueError("This channel name (%s) doesn't exist in info."
                             % ch_name)

        c_ind = ch_names.index(ch_name)
        if not isinstance(new_name, string_types):
            raise ValueError('Your mapping is not configured properly. '
                             'Please see the help: mne.rename_channels?')
        new_names.append((c_ind, new_name))
        if ch_name in bads:  # check bads
            new_bads.append((bads.index(ch_name), new_name))

    # Reset ch_names and Check that all the channel names are unique.
    for key, collection in [('ch_name', new_names), ('kind', new_kinds)]:
        for c_ind, new_name in collection:
            chs[c_ind][key] = new_name

    for c_ind, new_name in new_bads:
        bads[c_ind] = new_name

    # reference magic, please don't change (with the local binding
    # it doesn't work)
    info['ch_names'] = [c['ch_name'] for c in chs]

    # assert channel names are unique
    assert len(info['ch_names']) == np.unique(info['ch_names']).shape[0]


def _recursive_flatten(cell, dtype):
    """Helper to unpack mat files in Python"""
    while not isinstance(cell[0], dtype):
        cell = [c for d in cell for c in d]
    return cell


def read_ch_connectivity(fname, picks=None):
    """Parse FieldTrip neighbors .mat file

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
    """Compute sensor connectivity matrix

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
