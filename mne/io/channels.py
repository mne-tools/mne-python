# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engmeann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np

from ..externals.six import string_types

from ..constants import FIFF
from ..utils import verbose, logger
from ..pick import channel_type, pick_info
from .tree import dir_tree_find
from .tag import find_tag


def read_bad_channels(fid, node):
    """Read bad channels

    Parameters
    ----------
    fid : file
        The file descriptor.

    node : dict
        The node of the FIF tree that contains info on the bad channels.

    Returns
    -------
    bads : list
        A list of bad channel's names.
    """
    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = []
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None and tag.data is not None:
                bads = tag.data.split(':')
    return bads


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

    valid_channel_types = ('grad mag eeg stim eog emg ecg ref_meg resp '
                           'exci ias syst misc').split()

    if ch_type not in valid_channel_types:
        msg = ('The ch_type passed ({passed}) is not valid. '
               'it must be {valid}')
        raise ValueError(msg.format(passed=ch_type,
                                    valid=' or '.join(valid_channel_types)))
    return ch_type in [channel_type(info, ii) for ii in range(info['nchan'])]


@verbose
def equalize_channels(candidates, verbose=None):
    """Equalize channel picks for a collection of MNE-Python objects

    Parameters
    ----------
    candidates : list
        list Raw | Epochs | Evoked.
    verbose : None | bool
        whether to be verbose or not.

    Note. This function operates inplace.
    """
    from . import Raw
    from .. import Epochs
    from . import Evoked

    if not all([isinstance(c, (Raw, Epochs, Evoked)) for c in candidates]):
        valid = ['Raw', 'Epochs', 'Evoked']
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


class PickDropChannelsMixin(object):
    """Mixin class for Raw, Evoked, Epochs
    """
    def pick_channels(self, ch_names, copy=False):
        """Pick some channels

        Parameters
        ----------
        ch_names : list
            The list of channels to select.
        copy : bool
            If True, returns new instance. Else, modifies in place. Defaults to
            False.
        """
        inst = self.copy() if copy else self

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
        """
        inst = self.copy() if copy else self

        bad_idx = [inst.ch_names.index(c) for c in ch_names
                   if c in inst.ch_names]
        idx = np.setdiff1d(np.arange(len(inst.ch_names)), bad_idx)
        inst._pick_drop_channels(idx)

        return inst

    def _pick_drop_channels(self, idx):

        # avoid circular imports
        from .fiff.raw import _BaseRaw
        from .. import Epochs
        from . import Evoked

        inst_has = lambda attr: getattr(self, attr, None) is not None

        if inst_has('picks'):
            self.picks = self.picks[idx]

        if inst_has('cals'):
            self.cals = self.cals[idx]

        self.info = pick_info(self.info, idx, copy=False)

        if inst_has('_projector'):
            self._projector = self._projector[idx][:, idx]

        if isinstance(self, _BaseRaw) and inst_has('_data'):
            self._data = self._data[idx, :]
        elif isinstance(self, Epochs) and inst_has('_data'):
            self._data = self._data[:, idx, :]
        elif isinstance(self, Evoked):
            self.data = self.data[idx, :]


def rename_channels(info, mapping):
    """Rename channels and optionally change the sensor type.

    Note: This only changes between the following sensor types: eeg, eog,
    emg, ecg, and misc. It also cannot change to eeg.

    Parameters
    ----------
    info : dict
        Measurement info.
    mapping : dict
        a dictionary mapping the old channel to a new channel name {'EEG061' :
        'EEG161'}. If changing the sensor type, make the new name a tuple with
        the name (str) and the new channel type (str)
        {'EEG061',('EOG061','eog')}.
    """
    human2fiff = {'eog': FIFF.FIFFV_EOG_CH,
                  'emg': FIFF.FIFFV_EMG_CH,
                  'ecg': FIFF.FIFFV_ECG_CH,
                  'misc': FIFF.FIFFV_MISC_CH}

    bads, chs = info['bads'], info['chs']
    ch_names = info['ch_names']
    new_names, new_kinds, new_bads = list(), list(), list()

    # first check and assemble clean mappings of index and name
    for ch_name, new_name in mapping.items():
        if ch_name not in ch_names:
            raise ValueError("This channel name (%s) doesn't exist in info."
                             % ch_name)

        c_ind = ch_names.index(ch_name)
        if not isinstance(new_name, (string_types, tuple)):
            raise ValueError('Your mapping is not configured properly. '
                             'Please see the help: mne.rename_channels?')

        elif isinstance(new_name, tuple):  # name and type change
            new_name, new_type = new_name  # unpack
            if new_type not in human2fiff:
                raise ValueError('This function cannot change to this '
                                 'channel type: %s.' % new_type)
            new_kinds.append((c_ind, human2fiff[new_type]))

        if new_name in ch_names:
            raise ValueError('The new name ({new}) already exists. Choose a '
                             'unique name'.format(new=new_name))

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
