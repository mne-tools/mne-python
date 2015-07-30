"""Tools for working with epoched data"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Denis Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

import copy as cp
import warnings
import json

import os.path as op
import numpy as np

from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_float_matrix, write_float,
                       write_id, write_string, _get_split_size)
from .io.meas_info import read_meas_info, write_meas_info, _merge_info
from .io.open import fiff_open, _get_next_fname
from .io.tree import dir_tree_find
from .io.tag import read_tag
from .io.constants import FIFF
from .io.pick import (pick_types, channel_indices_by_type, channel_type,
                      pick_channels)
from .io.proj import setup_proj, ProjMixin, _proj_equal
from .io.base import _BaseRaw, ToDataFrameMixin
from .evoked import EvokedArray, _aspect_rev
from .baseline import rescale
from .channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                SetChannelsMixin, InterpolationMixin)
from .filter import resample, detrend, FilterMixin
from .event import _read_events_fif
from .fixes import in1d
from .viz import (plot_epochs, plot_epochs_trellis, _drop_log_stats,
                  plot_epochs_psd, plot_epochs_psd_topomap)
from .utils import (check_fname, logger, verbose, _check_type_picks,
                    _time_mask, check_random_state, object_hash)
from .externals.six import iteritems, string_types
from .externals.six.moves import zip


def _save_split(epochs, fname, part_idx, n_parts):
    """Split epochs"""

    # insert index in filename
    path, base = op.split(fname)
    idx = base.find('.')
    if part_idx > 0:
        fname = op.join(path, '%s-%d.%s' % (base[:idx], part_idx,
                                            base[idx + 1:]))

    next_fname = None
    if part_idx < n_parts - 1:
        next_fname = op.join(path, '%s-%d.%s' % (base[:idx], part_idx + 1,
                                                 base[idx + 1:]))
        next_idx = part_idx + 1

    fid = start_file(fname)

    info = epochs.info
    meas_id = info['meas_id']

    start_block(fid, FIFF.FIFFB_MEAS)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    if info['meas_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info['meas_id'])

    # Write measurement info
    write_meas_info(fid, info)

    # One or more evoked data sets
    start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
    start_block(fid, FIFF.FIFFB_EPOCHS)

    # write events out after getting data to ensure bad events are dropped
    data = epochs.get_data()
    start_block(fid, FIFF.FIFFB_MNE_EVENTS)
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, epochs.events.T)
    mapping_ = ';'.join([k + ':' + str(v) for k, v in
                         epochs.event_id.items()])
    write_string(fid, FIFF.FIFF_DESCRIPTION, mapping_)
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)

    # First and last sample
    first = int(epochs.times[0] * info['sfreq'])
    last = first + len(epochs.times) - 1
    write_int(fid, FIFF.FIFF_FIRST_SAMPLE, first)
    write_int(fid, FIFF.FIFF_LAST_SAMPLE, last)

    # save baseline
    if epochs.baseline is not None:
        bmin, bmax = epochs.baseline
        bmin = epochs.times[0] if bmin is None else bmin
        bmax = epochs.times[-1] if bmax is None else bmax
        write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, bmin)
        write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX, bmax)

    # The epochs itself
    decal = np.empty(info['nchan'])
    for k in range(info['nchan']):
        decal[k] = 1.0 / (info['chs'][k]['cal'] *
                          info['chs'][k].get('scale', 1.0))

    data *= decal[np.newaxis, :, np.newaxis]

    write_float_matrix(fid, FIFF.FIFF_EPOCH, data)

    # undo modifications to data
    data /= decal[np.newaxis, :, np.newaxis]

    write_string(fid, FIFF.FIFFB_MNE_EPOCHS_DROP_LOG,
                 json.dumps(epochs.drop_log))

    write_int(fid, FIFF.FIFFB_MNE_EPOCHS_SELECTION,
              epochs.selection)

    # And now write the next file info in case epochs are split on disk
    if next_fname is not None and n_parts > 1:
        start_block(fid, FIFF.FIFFB_REF)
        write_int(fid, FIFF.FIFF_REF_ROLE, FIFF.FIFFV_ROLE_NEXT_FILE)
        write_string(fid, FIFF.FIFF_REF_FILE_NAME, op.basename(next_fname))
        if meas_id is not None:
            write_id(fid, FIFF.FIFF_REF_FILE_ID, meas_id)
        write_int(fid, FIFF.FIFF_REF_FILE_NUM, next_idx)
        end_block(fid, FIFF.FIFFB_REF)

    end_block(fid, FIFF.FIFFB_EPOCHS)
    end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)


class _BaseEpochs(ProjMixin, ContainsMixin, UpdateChannelsMixin,
                  SetChannelsMixin, InterpolationMixin, FilterMixin,
                  ToDataFrameMixin):
    """Abstract base class for Epochs-type classes

    This class provides basic functionality and should never be instantiated
    directly. See Epochs below for an explanation of the parameters.
    """
    def __init__(self, info, data, events, event_id, tmin, tmax,
                 baseline=(None, 0), raw=None,
                 picks=None, name='Unknown', reject=None, flat=None,
                 decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 add_eeg_ref=True, proj=True, on_missing='error',
                 preload_at_end=False, verbose=None):

        self.verbose = verbose
        self.name = name

        if on_missing not in ['error', 'warning', 'ignore']:
            raise ValueError('on_missing must be one of: error, '
                             'warning, ignore. Got: %s' % on_missing)

        # check out event_id dict
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
        elif isinstance(event_id, dict):
            if not all(isinstance(v, int) for v in event_id.values()):
                raise ValueError('Event IDs must be of type integer')
            if not all(isinstance(k, string_types) for k in event_id):
                raise ValueError('Event names must be of type str')
        elif isinstance(event_id, list):
            if not all(isinstance(v, int) for v in event_id):
                raise ValueError('Event IDs must be of type integer')
            event_id = dict(zip((str(i) for i in event_id), event_id))
        elif isinstance(event_id, int):
            event_id = {str(event_id): event_id}
        else:
            raise ValueError('event_id must be dict or int.')
        self.event_id = event_id
        del event_id

        if events is not None:  # RtEpochs can have events=None
            for key, val in self.event_id.items():
                if val not in events[:, 2]:
                    msg = ('No matching events found for %s '
                           '(event id %i)' % (key, val))
                    if on_missing == 'error':
                        raise ValueError(msg)
                    elif on_missing == 'warning':
                        logger.warning(msg)
                        warnings.warn(msg)
                    else:  # on_missing == 'ignore':
                        pass

            values = list(self.event_id.values())
            selected = in1d(events[:, 2], values)
            self.selection = np.where(selected)[0]
            self.drop_log = [list() if k in self.selection else ['IGNORED']
                             for k in range(len(events))]
            events = events[selected]
            n_events = len(events)
            if n_events > 1:
                if np.diff(events.astype(np.int64)[:, 0]).min() <= 0:
                    warnings.warn('The events passed to the Epochs '
                                  'constructor are not chronologically '
                                  'ordered.', RuntimeWarning)

            if n_events > 0:
                logger.info('%d matching events found' % n_events)
            else:
                raise ValueError('No desired events found.')
            self.events = events
            del events
        else:
            self.drop_log = list()
            self.selection = np.array([], int)
            # do not set self.events here, let subclass do it

        # check reject_tmin and reject_tmax
        if (reject_tmin is not None) and (reject_tmin < tmin):
            raise ValueError("reject_tmin needs to be None or >= tmin")
        if (reject_tmax is not None) and (reject_tmax > tmax):
            raise ValueError("reject_tmax needs to be None or <= tmax")
        if (reject_tmin is not None) and (reject_tmax is not None):
            if reject_tmin >= reject_tmax:
                raise ValueError('reject_tmin needs to be < reject_tmax')
        if detrend not in [None, 0, 1]:
            raise ValueError('detrend must be None, 0, or 1')

        # check that baseline is in available data
        if baseline is not None:
            baseline_tmin, baseline_tmax = baseline
            tstep = 1. / info['sfreq']
            if baseline_tmin is not None:
                if baseline_tmin < tmin - tstep:
                    err = ("Baseline interval (tmin = %s) is outside of epoch "
                           "data (tmin = %s)" % (baseline_tmin, tmin))
                    raise ValueError(err)
            if baseline_tmax is not None:
                if baseline_tmax > tmax + tstep:
                    err = ("Baseline interval (tmax = %s) is outside of epoch "
                           "data (tmax = %s)" % (baseline_tmax, tmax))
                    raise ValueError(err)
        if tmin >= tmax:
            raise ValueError('tmin has to be smaller than tmax')

        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        self.detrend = detrend
        self._raw = raw
        self.info = info
        del info

        if picks is None:
            picks = list(range(len(self.info['ch_names'])))
        else:
            self.info['chs'] = [self.info['chs'][k] for k in picks]
            self.info['ch_names'] = [self.info['ch_names'][k] for k in picks]
            self.info['nchan'] = len(picks)
        self.picks = _check_type_picks(picks)
        if len(picks) == 0:
            raise ValueError("Picks cannot be empty.")

        if data is None:
            self.preload = False
            self._data = None
        else:
            assert decim == 1
            if data.ndim != 3 or data.shape[2] != \
                    round((tmax - tmin) * self.info['sfreq']) + 1:
                raise RuntimeError('bad data shape')
            self.preload = True
            self._data = data
        self._offset = None

        # Handle times
        sfreq = float(self.info['sfreq'])
        start_idx = int(round(self.tmin * sfreq))
        self._raw_times = np.arange(start_idx,
                                    int(round(self.tmax * sfreq)) + 1) / sfreq
        self._decim = 1
        # this method sets the self.times property
        self.decimate(decim)

        # setup epoch rejection
        self.reject = None
        self.flat = None
        self._reject_setup(reject, flat)

        # do the rest
        if proj not in [True, 'delayed', False]:
            raise ValueError(r"'proj' must either be 'True', 'False' or "
                             "'delayed'")

        # proj is on when applied in Raw
        proj = proj or (raw is not None and raw.proj)
        if proj == 'delayed':
            if self.reject is None:
                raise RuntimeError('The delayed SSP mode was requested '
                                   'but no rejection parameters are present. '
                                   'Please add rejection parameters before '
                                   'using this option.')
            self._delayed_proj = True
            logger.info('Entering delayed SSP mode.')
        else:
            self._delayed_proj = False

        activate = False if self._delayed_proj else proj
        self._projector, self.info = setup_proj(self.info, add_eeg_ref,
                                                activate=activate)

        if preload_at_end:
            assert self._data is None
            assert self.preload is False
            self.preload_data()

    def preload_data(self):
        """Preload the data if not already preloaded

        Returns
        -------
        epochs : instance of Epochs
            The epochs object.

        Notes
        -----
        This function operates inplace.

        .. versionadded:: 0.10.0
        """
        if self.preload:
            return
        self._data = self._get_data()
        self.preload = True
        self._decim_slice = slice(None, None, None)
        self._decim = 1
        self._raw_times = self.times
        return self

    def decimate(self, decim, copy=False):
        """Decimate the epochs

        Parameters
        ----------
        decim : int
            The amount to decimate data.
        copy : bool
            If True, operate on and return a copy of the Epochs object.

        Returns
        -------
        epochs : instance of Epochs
            The decimated Epochs object.

        Notes
        -----
        Decimation can be done multiple times. For example,
        ``epochs.decimate(2).decimate(2)`` will be the same as
        ``epochs.decimate(4)``.

        .. versionadded:: 0.10.0
        """
        if decim < 1 or decim != int(decim):
            raise ValueError('decim must be an integer > 0')
        decim = int(decim)
        epochs = self.copy() if copy else self
        del self

        new_sfreq = epochs.info['sfreq'] / float(decim)
        lowpass = epochs.info['lowpass']
        if decim > 1 and lowpass is None:
            warnings.warn('The measurement information indicates data is not '
                          'low-pass filtered. The decim=%i parameter will '
                          'result in a sampling frequency of %g Hz, which can '
                          'cause aliasing artifacts.'
                          % (decim, new_sfreq))
        elif decim > 1 and new_sfreq < 2.5 * lowpass:
            warnings.warn('The measurement information indicates a low-pass '
                          'frequency of %g Hz. The decim=%i parameter will '
                          'result in a sampling frequency of %g Hz, which can '
                          'cause aliasing artifacts.'
                          % (lowpass, decim, new_sfreq))  # > 50% nyquist limit

        epochs._decim *= decim
        start_idx = int(round(epochs._raw_times[0] * (epochs.info['sfreq'] *
                                                      epochs._decim)))
        i_start = start_idx % epochs._decim
        decim_slice = slice(i_start, len(epochs._raw_times), epochs._decim)
        epochs.info['sfreq'] = new_sfreq
        if epochs.preload:
            epochs._data = epochs._data[:, :, decim_slice].copy()
            epochs._raw_times = epochs._raw_times[decim_slice].copy()
            epochs._decim_slice = slice(None, None, None)
            epochs._decim = 1
            epochs.times = epochs._raw_times
        else:
            epochs._decim_slice = decim_slice
            epochs.times = epochs._raw_times[epochs._decim_slice]
        return epochs

    def _reject_setup(self, reject, flat):
        """Sets self._reject_time and self._channel_type_idx"""
        idx = channel_indices_by_type(self.info)
        for rej, kind in zip((reject, flat), ('reject', 'flat')):
            if not isinstance(rej, (type(None), dict)):
                raise TypeError('reject and flat must be dict or None, not %s'
                                % type(rej))
            if isinstance(rej, dict):
                bads = set(rej.keys()) - set(idx.keys())
                if len(bads) > 0:
                    raise KeyError('Unknown channel types found in %s: %s'
                                   % (kind, bads))

        for key in idx.keys():
            if (reject is not None and key in reject) \
                    or (flat is not None and key in flat):
                if len(idx[key]) == 0:
                    raise ValueError("No %s channel found. Cannot reject based"
                                     " on %s." % (key.upper(), key.upper()))
            # now check to see if our rejection and flat are getting more
            # restrictive
            old_reject = self.reject if self.reject is not None else dict()
            new_reject = reject if reject is not None else dict()
            old_flat = self.flat if self.flat is not None else dict()
            new_flat = flat if flat is not None else dict()
            bad_msg = ('{kind}["{key}"] == {new} {op} {old} (old value), new '
                       '{kind} values must be at least as stringent as '
                       'previous ones')
            for key in set(new_reject.keys()).union(old_reject.keys()):
                old = old_reject.get(key, np.inf)
                new = new_reject.get(key, np.inf)
                if new > old:
                    raise ValueError(bad_msg.format(kind='reject', key=key,
                                                    new=new, old=old, op='>'))
            for key in set(new_flat.keys()).union(old_flat.keys()):
                old = old_flat.get(key, -np.inf)
                new = new_flat.get(key, -np.inf)
                if new < old:
                    raise ValueError(bad_msg.format(kind='flat', key=key,
                                                    new=new, old=old, op='<'))

        # after validation, set parameters
        self._bad_dropped = False
        self._channel_type_idx = idx
        self.reject = reject
        self.flat = flat

        if (self.reject_tmin is None) and (self.reject_tmax is None):
            self._reject_time = None
        else:
            if self.reject_tmin is None:
                reject_imin = None
            else:
                idxs = np.nonzero(self.times >= self.reject_tmin)[0]
                reject_imin = idxs[0]
            if self.reject_tmax is None:
                reject_imax = None
            else:
                idxs = np.nonzero(self.times <= self.reject_tmax)[0]
                reject_imax = idxs[-1]

            self._reject_time = slice(reject_imin, reject_imax)

    @verbose
    def _is_good_epoch(self, data, verbose=None):
        """Determine if epoch is good"""
        if data is None:
            return False, ['NO_DATA']
        n_times = len(self.times)
        if data.shape[1] < n_times:
            # epoch is too short ie at the end of the data
            return False, ['TOO_SHORT']
        if self.reject is None and self.flat is None:
            return True, None
        else:
            if self._reject_time is not None:
                data = data[:, self._reject_time]

            return _is_good(data, self.ch_names, self._channel_type_idx,
                            self.reject, self.flat, full_report=True,
                            ignore_chs=self.info['bads'])

    @verbose
    def _preprocess(self, epoch, verbose=None):
        """Aux Function: detrend, baseline correct, offset, decim

        Note: operates inplace
        """
        # Detrend
        if self.detrend is not None:
            picks = pick_types(self.info, meg=True, eeg=True, stim=False,
                               ref_meg=False, eog=False, ecg=False,
                               emg=False, exclude=[])
            epoch[picks] = detrend(epoch[picks], self.detrend, axis=1)

        # Baseline correct
        picks = pick_types(self.info, meg=True, eeg=True, stim=False,
                           ref_meg=True, eog=True, ecg=True,
                           emg=True, exclude=[])
        epoch[picks] = rescale(epoch[picks], self._raw_times, self.baseline,
                               'mean', copy=False, verbose=verbose)

        # handle offset
        if self._offset is not None:
            epoch += self._offset

        # Decimate if necessary (i.e., epoch not preloaded)
        epoch = epoch[:, self._decim_slice]
        return epoch

    def iter_evoked(self):
        """Iterate over Evoked objects with nave=1
        """
        self._current = 0

        while True:
            data, event_id = self.next(True)
            tmin = self.times[0]
            info = cp.deepcopy(self.info)

            yield EvokedArray(data, info, tmin, comment=str(event_id))

    def subtract_evoked(self, evoked=None):
        """Subtract an evoked response from each epoch

        Can be used to exclude the evoked response when analyzing induced
        activity, see e.g. [1].

        References
        ----------
        [1] David et al. "Mechanisms of evoked and induced responses in
        MEG/EEG", NeuroImage, vol. 31, no. 4, pp. 1580-1591, July 2006.

        Parameters
        ----------
        evoked : instance of Evoked | None
            The evoked response to subtract. If None, the evoked response
            is computed from Epochs itself.

        Returns
        -------
        self : instance of Epochs
            The modified instance (instance is also modified inplace).
        """
        logger.info('Subtracting Evoked from Epochs')
        if evoked is None:
            picks = pick_types(self.info, meg=True, eeg=True,
                               stim=False, eog=False, ecg=False,
                               emg=False, exclude=[])
            evoked = self.average(picks)

        # find the indices of the channels to use
        picks = pick_channels(evoked.ch_names, include=self.ch_names)

        # make sure the omitted channels are not data channels
        if len(picks) < len(self.ch_names):
            sel_ch = [evoked.ch_names[ii] for ii in picks]
            diff_ch = list(set(self.ch_names).difference(sel_ch))
            diff_idx = [self.ch_names.index(ch) for ch in diff_ch]
            diff_types = [channel_type(self.info, idx) for idx in diff_idx]
            bad_idx = [diff_types.index(t) for t in diff_types if t in
                       ['grad', 'mag', 'eeg']]
            if len(bad_idx) > 0:
                bad_str = ', '.join([diff_ch[ii] for ii in bad_idx])
                raise ValueError('The following data channels are missing '
                                 'in the evoked response: %s' % bad_str)
            logger.info('    The following channels are not included in the '
                        'subtraction: %s' % ', '.join(diff_ch))

        # make sure the times match
        if (len(self.times) != len(evoked.times) or
                np.max(np.abs(self.times - evoked.times)) >= 1e-7):
            raise ValueError('Epochs and Evoked object do not contain '
                             'the same time points.')

        # handle SSPs
        if not self.proj and evoked.proj:
            warnings.warn('Evoked has SSP applied while Epochs has not.')
        if self.proj and not evoked.proj:
            evoked = evoked.copy().apply_proj()

        # find the indices of the channels to use in Epochs
        ep_picks = [self.ch_names.index(evoked.ch_names[ii]) for ii in picks]

        # do the subtraction
        if self.preload:
            self._data[:, ep_picks, :] -= evoked.data[picks][None, :, :]
        else:
            if self._offset is None:
                self._offset = np.zeros((len(self.ch_names), len(self.times)),
                                        dtype=np.float)
            self._offset[ep_picks] -= evoked.data[picks]
        logger.info('[done]')

        return self

    def __next__(self, *args, **kwargs):
        """Wrapper for Py3k"""
        return self.next(*args, **kwargs)

    def __hash__(self):
        if not self.preload:
            raise RuntimeError('Cannot hash epochs unless preloaded')
        return object_hash(dict(info=self.info, data=self._data))

    def average(self, picks=None):
        """Compute average of epochs

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        evoked : instance of Evoked
            The averaged epochs.
        """

        return self._compute_mean_or_stderr(picks, 'ave')

    def standard_error(self, picks=None):
        """Compute standard error over epochs

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG and EEG channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        evoked : instance of Evoked
            The standard error over epochs.
        """
        return self._compute_mean_or_stderr(picks, 'stderr')

    def _compute_mean_or_stderr(self, picks, mode='ave'):
        """Compute the mean or std over epochs and return Evoked"""

        _do_std = True if mode == 'stderr' else False

        n_channels = len(self.ch_names)
        n_times = len(self.times)
        if self.preload:
            n_events = len(self.events)
            fun = np.std if _do_std else np.mean
            data = fun(self._data, axis=0)
            assert len(self.events) == len(self._data)
        else:
            data = np.zeros((n_channels, n_times))
            n_events = 0
            for e in self:
                data += e
                n_events += 1

            if n_events > 0:
                data /= n_events
            else:
                data.fill(np.nan)

            # convert to stderr if requested, could do in one pass but do in
            # two (slower) in case there are large numbers
            if _do_std:
                data_mean = cp.copy(data)
                data.fill(0.)
                for e in self:
                    data += (e - data_mean) ** 2
                data = np.sqrt(data / n_events)

        if not _do_std:
            _aspect_kind = FIFF.FIFFV_ASPECT_AVERAGE
        else:
            _aspect_kind = FIFF.FIFFV_ASPECT_STD_ERR
            data /= np.sqrt(n_events)
        kind = _aspect_rev.get(str(_aspect_kind), 'Unknown')

        info = cp.deepcopy(self.info)
        evoked = EvokedArray(data, info, tmin=self.times[0],
                             comment=self.name, nave=n_events, kind=kind,
                             verbose=self.verbose)
        # XXX: above constructor doesn't recreate the times object precisely
        evoked.times = self.times.copy()
        evoked._aspect_kind = _aspect_kind

        # pick channels
        if picks is None:
            picks = pick_types(evoked.info, meg=True, eeg=True, ref_meg=True,
                               stim=False, eog=False, ecg=False,
                               emg=False, exclude=[])

        ch_names = [evoked.ch_names[p] for p in picks]
        evoked.pick_channels(ch_names)

        if len(evoked.info['ch_names']) == 0:
            raise ValueError('No data channel found when averaging.')

        if evoked.nave < 1:
            warnings.warn('evoked object is empty (based on less '
                          'than 1 epoch)', RuntimeWarning)

        return evoked

    @property
    def ch_names(self):
        """Channel names"""
        return self.info['ch_names']

    def plot(self, epoch_idx=None, picks=None, scalings=None,
             title_str='#%003i', show=True, block=False, n_epochs=20,
             n_channels=20, title=None, trellis=True):
        """Visualize epochs.

        Bad epochs can be marked with a left click on top of the epoch. Bad
        channels can be selected by clicking the channel name on the left side
        of the main axes. Calling this function drops all the selected bad
        epochs as well as bad epochs marked beforehand with rejection
        parameters.

        Parameters
        ----------
        epoch_idx : array-like | int | None
            The epochs to visualize. If None, the first 20 epochs are shown.
            Defaults to None.
        picks : array-like of int | None
            Channels to be included. If None only good data channels are used.
            Defaults to None
        scalings : dict | None
            Scale factors for the traces. If None, defaults to
            ``dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
            emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4)``.
        title_str : None | str
            The string formatting to use for axes titles. If None, no titles
            will be shown. Defaults expand to ``#001, #002, ...``.
        show : bool
            Whether to show the figure or not.
        block : bool
            Whether to halt program execution until the figure is closed.
            Useful for rejecting bad trials on the fly by clicking on a
            sub plot.
        n_epochs : int
            The number of epochs per view.
        n_channels : int
            The number of channels per view on mne_browse_epochs. If trellis is
            True, this parameter has no effect. Defaults to 20.
        title : str | None
            The title of the window. If None, epochs name will be displayed.
            If trellis is True, this parameter has no effect.
            Defaults to None.
        trellis : bool
            Whether to use Trellis plotting. If False, plotting is similar to
            Raw plot methods, with epochs stacked horizontally.
            Defaults to True.

        Returns
        -------
        fig : Instance of matplotlib.figure.Figure
            The figure.

        Notes
        -----
        With trellis set to False, the arrow keys (up/down/left/right) can
        be used to navigate between channels and epochs and the scaling can be
        adjusted with - and + (or =) keys, but this depends on the backend
        matplotlib is configured to use (e.g., mpl.use(``TkAgg``) should work).
        Full screen mode can be to toggled with f11 key. The amount of epochs
        and channels per view can be adjusted with home/end and
        page down/page up keys. Butterfly plot can be toggled with ``b`` key.
        Right mouse click adds a vertical line to the plot.

        .. versionadded:: 0.10.0
        """
        if trellis is True:
            return plot_epochs_trellis(self, epoch_idx=epoch_idx, picks=picks,
                                       scalings=scalings, title_str=title_str,
                                       show=show, block=block,
                                       n_epochs=n_epochs)
        else:
            return plot_epochs(self, picks=picks, scalings=scalings,
                               n_epochs=n_epochs, n_channels=n_channels,
                               title=title, show=show, block=block)

    def plot_psd(self, fmin=0, fmax=np.inf, proj=False, n_fft=256,
                 picks=None, ax=None, color='black', area_mode='std',
                 area_alpha=0.33, n_overlap=0, dB=True,
                 n_jobs=1, verbose=None, show=True):
        """Plot the power spectral density across epochs

        Parameters
        ----------
        fmin : float
            Start frequency to consider.
        fmax : float
            End frequency to consider.
        proj : bool
            Apply projection.
        n_fft : int
            Number of points to use in Welch FFT calculations.
        picks : array-like of int | None
            List of channels to use.
        ax : instance of matplotlib Axes | None
            Axes to plot into. If None, axes will be created.
        color : str | tuple
            A matplotlib-compatible color to use.
        area_mode : str | None
            Mode for plotting area. If 'std', the mean +/- 1 STD (across
            channels) will be plotted. If 'range', the min and max (across
            channels) will be plotted. Bad channels will be excluded from
            these calculations. If None, no area will be plotted.
        area_alpha : float
            Alpha for the area.
        n_overlap : int
            The number of points of overlap between blocks.
        dB : bool
            If True, transform data to decibels.
        n_jobs : int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
        show : bool
            Show figure if True.

        Returns
        -------
        fig : instance of matplotlib figure
            Figure distributing one image per channel across sensor topography.
        """
        return plot_epochs_psd(self, fmin=fmin, fmax=fmax, proj=proj,
                               n_fft=n_fft, picks=picks, ax=ax,
                               color=color, area_mode=area_mode,
                               area_alpha=area_alpha,
                               n_overlap=n_overlap, dB=dB, n_jobs=n_jobs,
                               verbose=None, show=show)

    def plot_psd_topomap(self, bands=None, vmin=None, vmax=None, proj=False,
                         n_fft=256, ch_type=None,
                         n_overlap=0, layout=None, cmap='RdBu_r',
                         agg_fun=None, dB=True, n_jobs=1, normalize=False,
                         cbar_fmt='%0.3f', outlines='head', show=True,
                         verbose=None):
        """Plot the topomap of the power spectral density across epochs

        Parameters
        ----------
        bands : list of tuple | None
            The lower and upper frequency and the name for that band. If None,
            (default) expands to:

            bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                     (12, 30, 'Beta'), (30, 45, 'Gamma')]

        vmin : float | callable
            The value specfying the lower bound of the color range.
            If None, and vmax is None, -vmax is used. Else np.min(data).
            If callable, the output equals vmin(data).
        vmax : float | callable
            The value specfying the upper bound of the color range.
            If None, the maximum absolute value is used. If vmin is None,
            but vmax is not, defaults to np.min(data).
            If callable, the output equals vmax(data).
        proj : bool
            Apply projection.
        n_fft : int
            Number of points to use in Welch FFT calculations.
        ch_type : {None, 'mag', 'grad', 'planar1', 'planar2', 'eeg'}
            The channel type to plot. For 'grad', the gradiometers are
            collected in
            pairs and the RMS for each pair is plotted. If None, defaults to
            'mag' if MEG data are present and to 'eeg' if only EEG data are
            present.
        n_overlap : int
            The number of points of overlap between blocks.
        layout : None | Layout
            Layout instance specifying sensor positions (does not need to
            be specified for Neuromag data). If possible, the correct layout
            file is inferred from the data; if no appropriate layout file was
            found, the layout is automatically generated from the sensor
            locations.
        cmap : matplotlib colormap
            Colormap. For magnetometers and eeg defaults to 'RdBu_r', else
            'Reds'.
        agg_fun : callable
            The function used to aggregate over frequencies.
            Defaults to np.sum. if normalize is True, else np.mean.
        dB : bool
            If True, transform data to decibels (with ``10 * np.log10(data)``)
            following the application of `agg_fun`. Only valid if normalize
            is False.
        n_jobs : int
            Number of jobs to run in parallel.
        normalize : bool
            If True, each band will be devided by the total power. Defaults to
            False.
        cbar_fmt : str
            The colorbar format. Defaults to '%0.3f'.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', a head scheme will be drawn.
            If dict, each key refers to a tuple of x and y positions.
            The values in 'mask_pos' will serve as image mask. If None, nothing
            will be drawn. Defaults to 'head'. If dict, the 'autoshrink' (bool)
            field will trigger automated shrinking of the positions due to
            points outside the outline. Moreover, a matplotlib patch object can
            be passed for advanced masking options, either directly or as a
            function that returns patches (required for multi-axis plots).
        show : bool
            Show figure if True.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        fig : instance of matplotlib figure
            Figure distributing one image per channel across sensor topography.
        """
        return plot_epochs_psd_topomap(
            self, bands=bands, vmin=vmin, vmax=vmax, proj=proj, n_fft=n_fft,
            ch_type=ch_type, n_overlap=n_overlap, layout=layout, cmap=cmap,
            agg_fun=agg_fun, dB=dB, n_jobs=n_jobs, normalize=normalize,
            cbar_fmt=cbar_fmt, outlines=outlines, show=show, verbose=None)

    def drop_bad_epochs(self, reject='existing', flat='existing'):
        """Drop bad epochs without retaining the epochs data.

        Should be used before slicing operations.

        .. Warning:: Operation is slow since all epochs have to be read from
            disk. To avoid reading epochs from disk multiple times, initialize
            Epochs object with preload=True.

        Parameters
        ----------
        reject : dict | str | None
            Rejection parameters based on peak-to-peak amplitude.
            Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
            If reject is None then no rejection is done. If 'existing',
            then the rejection parameters set at instantiation are used.
        flat : dict | str | None
            Rejection parameters based on flatness of signal.
            Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
            are floats that set the minimum acceptable peak-to-peak amplitude.
            If flat is None then no rejection is done. If 'existing',
            then the flat parameters set at instantiation are used.

        Notes
        -----
        Dropping bad epochs can be done multiple times with different
        ``reject`` and ``flat`` parameters. However, once an epoch is
        dropped, it is dropped forever, so if more lenient thresholds may
        subsequently be applied, `epochs.copy` should be used.
        """
        if reject == 'existing':
            if flat == 'existing' and self._bad_dropped:
                return
            reject = self.reject
        if flat == 'existing':
            flat = self.flat
        if any(isinstance(rej, string_types) and rej != 'existing' for
               rej in (reject, flat)):
            raise ValueError('reject and flat, if strings, must be "existing"')
        self._reject_setup(reject, flat)
        self._get_data(out=False)

    def drop_log_stats(self, ignore=['IGNORED']):
        """Compute the channel stats based on a drop_log from Epochs.

        Parameters
        ----------
        ignore : list
            The drop reasons to ignore.

        Returns
        -------
        perc : float
            Total percentage of epochs dropped.

        See Also
        --------
        plot_drop_log
        """
        return _drop_log_stats(self.drop_log, ignore)

    def plot_drop_log(self, threshold=0, n_max_plot=20, subject='Unknown',
                      color=(0.9, 0.9, 0.9), width=0.8, ignore=['IGNORED'],
                      show=True):
        """Show the channel stats based on a drop_log from Epochs

        Parameters
        ----------
        threshold : float
            The percentage threshold to use to decide whether or not to
            plot. Default is zero (always plot).
        n_max_plot : int
            Maximum number of channels to show stats for.
        subject : str
            The subject name to use in the title of the plot.
        color : tuple | str
            Color to use for the bars.
        width : float
            Width of the bars.
        ignore : list
            The drop reasons to ignore.
        show : bool
            Show figure if True.

        Returns
        -------
        perc : float
            Total percentage of epochs dropped.
        fig : Instance of matplotlib.figure.Figure
            The figure.
        """
        if not self._bad_dropped:
            raise ValueError("You cannot use plot_drop_log since bad "
                             "epochs have not yet been dropped. "
                             "Use epochs.drop_bad_epochs().")

        from .viz import plot_drop_log
        return plot_drop_log(self.drop_log, threshold, n_max_plot, subject,
                             color=color, width=width, ignore=ignore,
                             show=show)

    @verbose
    def drop_epochs(self, indices, reason='USER', verbose=None):
        """Drop epochs based on indices or boolean mask

        Note that the indices refer to the current set of undropped epochs
        rather than the complete set of dropped and undropped epochs.
        They are therefore not necessarily consistent with any external indices
        (e.g., behavioral logs). To drop epochs based on external criteria,
        do not use the preload=True flag when constructing an Epochs object,
        and call this method before calling the drop_bad_epochs method.

        Parameters
        ----------
        indices : array of ints or bools
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are
            correspondingly modified.
        reason : str
            Reason for dropping the epochs ('ECG', 'timeout', 'blink' etc).
            Default: 'USER'.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to raw.verbose.
        """
        indices = np.atleast_1d(indices)

        if indices.ndim > 1:
            raise ValueError("indices must be a scalar or a 1-d array")

        if indices.dtype == bool:
            indices = np.where(indices)[0]

        out_of_bounds = (indices < 0) | (indices >= len(self.events))
        if out_of_bounds.any():
            first = indices[out_of_bounds][0]
            raise IndexError("Epoch index %d is out of bounds" % first)

        for ii in indices:
            self.drop_log[self.selection[ii]].append(reason)

        self.selection = np.delete(self.selection, indices)
        self.events = np.delete(self.events, indices, axis=0)
        if self.preload:
            self._data = np.delete(self._data, indices, axis=0)

        count = len(indices)
        logger.info('Dropped %d epoch%s' % (count, '' if count == 1 else 's'))

    def _get_epoch_from_raw(self, idx, verbose=None):
        """Method to get a given epoch from disk"""
        raise NotImplementedError

    def _process_epoch_raw(self, epoch_raw):
        """Helper to process a raw epoch based on the delayed param"""
        # whenever requested, the first epoch is being projected.
        if epoch_raw is None:  # can happen if t < 0
            return None
        proj = self._delayed_proj or self.proj
        if self._projector is not None and proj is True:
            epoch = self._preprocess(np.dot(self._projector, epoch_raw))
        else:
            epoch = self._preprocess(epoch_raw.copy())
        return epoch

    @verbose
    def _get_data(self, out=True, verbose=None):
        """Load all data, dropping bad epochs along the way

        Parameters
        ----------
        out : bool
            Return the data. Setting this to False is used to reject bad
            epochs without caching all the data, which saves memory.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.
        """
        n_events = len(self.events)
        # in case there are no good events
        if self.preload:
            # we will store our result in our existing array
            data = self._data
        else:
            # we start out with an empty array, allocate only if necessary
            data = np.empty((0, len(self.info['ch_names']), len(self.times)))
        if self._bad_dropped:
            if not out:
                return
            if self.preload:
                return data

            # we need to load from disk, drop, and return data
            for idx in range(n_events):
                # faster to pre-allocate memory here
                epoch_raw = self._get_epoch_from_raw(idx)
                if self._delayed_proj:
                    epoch_out = epoch_raw
                else:
                    epoch_out = self._process_epoch_raw(epoch_raw)
                if idx == 0:
                    data = np.empty((n_events, epoch_out.shape[0],
                                     epoch_out.shape[1]),
                                    dtype=epoch_raw.dtype)
                data[idx] = epoch_out
        else:
            # bads need to be dropped, this might occur after a preload
            # e.g., when calling drop_bad_epochs w/new params
            good_idx = []
            n_out = 0
            assert n_events == len(self.selection)
            for idx, sel in enumerate(self.selection):
                if self.preload:  # from memory
                    if self._delayed_proj:
                        epoch_raw = self._data[idx]
                        epoch = self._process_epoch_raw(epoch_raw)
                    else:
                        epoch_raw = None
                        epoch = self._data[idx]
                else:  # from disk
                    epoch_raw = self._get_epoch_from_raw(idx)
                    epoch = self._process_epoch_raw(epoch_raw)
                epoch_out = epoch_raw if self._delayed_proj else epoch
                is_good, offenders = self._is_good_epoch(epoch)
                if not is_good:
                    self.drop_log[sel] += offenders
                    continue
                good_idx.append(idx)

                # store the epoch if there is a reason to (output or update)
                if out or self.preload:
                    # faster to pre-allocate, then trim as necessary
                    if n_out == 0 and not self.preload:
                        data = np.empty((n_events, epoch_out.shape[0],
                                         epoch_out.shape[1]),
                                        dtype=epoch_out.dtype, order='C')
                    data[n_out] = epoch_out
                    n_out += 1

            self._bad_dropped = True
            logger.info("%d bad epochs dropped" % (n_events - len(good_idx)))

            # Now update our properties
            if len(good_idx) == 0:  # silly fix for old numpy index error
                self.selection = np.array([], int)
                self.events = np.empty((0, 3))
            else:
                self.selection = self.selection[good_idx]
                self.events = np.atleast_2d(self.events[good_idx])

            # adjust the data size if there is a reason to (output or update)
            if out or self.preload:
                data.resize((n_out,) + data.shape[1:], refcheck=False)

        return data if out else None

    def get_data(self):
        """Get all epochs as a 3D array

        Returns
        -------
        data : array of shape (n_epochs, n_channels, n_times)
            A copy of the epochs data.
        """
        data_ = self._get_data()
        if self._delayed_proj:
            data = np.zeros_like(data_)
            for ii, e in enumerate(data_):
                data[ii] = self._preprocess(e.copy())
        else:
            data = data_.copy()
        return data

    def __len__(self):
        """Number of epochs.
        """
        if not self._bad_dropped:
            raise RuntimeError('Since bad epochs have not been dropped, the '
                               'length of the Epochs is not known. Load the '
                               'Epochs with preload=True, or call '
                               'Epochs.drop_bad_epochs(). To find the number '
                               'of events in the Epochs, use '
                               'len(Epochs.events).')
        return len(self.events)

    def __iter__(self):
        """To make iteration over epochs easy.
        """
        self._current = 0
        return self

    def next(self, return_event_id=False):
        """To make iteration over epochs easy.

        Parameters
        ----------
        return_event_id : bool
            If True, return both an epoch and and event_id.

        Returns
        -------
        epoch : instance of Epochs
            The epoch.
        event_id : int
            The event id. Only returned if ``return_event_id`` is ``True``.
        """
        if self.preload:
            if self._current >= len(self._data):
                raise StopIteration
            epoch = self._data[self._current]
            if self._delayed_proj:
                epoch = self._preprocess(epoch.copy())
            self._current += 1
        else:
            is_good = False
            while not is_good:
                if self._current >= len(self.events):
                    raise StopIteration
                epoch_raw = self._get_epoch_from_raw(self._current)
                epoch = self._process_epoch_raw(epoch_raw)
                self._current += 1
                is_good, _ = self._is_good_epoch(epoch)
            # If delayed-ssp mode, pass 'virgin' data after rejection decision.
            if self._delayed_proj:
                epoch = self._preprocess(epoch_raw)

        if not return_event_id:
            return epoch
        else:
            return epoch, self.events[self._current - 1][-1]

        return epoch if not return_event_id else epoch, self.event_id

    def __repr__(self):
        """ Build string representation
        """
        s = 'n_events : %s ' % len(self.events)
        s += '(all good)' if self._bad_dropped else '(good & bad)'
        s += ', tmin : %s (s)' % self.tmin
        s += ', tmax : %s (s)' % self.tmax
        s += ', baseline : %s' % str(self.baseline)
        if len(self.event_id) > 1:
            counts = ['%r: %i' % (k, sum(self.events[:, 2] == v))
                      for k, v in sorted(self.event_id.items())]
            s += ',\n %s' % ', '.join(counts)
        class_name = self.__class__.__name__
        if class_name == '_BaseEpochs':
            class_name = 'Epochs'
        return '<%s  |  %s>' % (class_name, s)

    def _key_match(self, key):
        """Helper function for event dict use"""
        if key not in self.event_id:
            raise KeyError('Event "%s" is not in Epochs.' % key)
        return self.events[:, 2] == self.event_id[key]

    def __getitem__(self, key):
        """Return an Epochs object with a subset of epochs
        """
        data = self._data
        del self._data
        epochs = self.copy()
        self._data, epochs._data = data, data
        del self

        if isinstance(key, string_types):
            key = [key]

        if isinstance(key, (list, tuple)) and isinstance(key[0], string_types):
            if any('/' in k_i for k_i in epochs.event_id.keys()):
                if any(k_e not in epochs.event_id for k_e in key):
                    # Select a given key if the requested set of
                    # '/'-separated types are a subset of the types in that key
                    key = [k for k in epochs.event_id.keys()
                           if all(set(k_i.split('/')).issubset(k.split('/'))
                                  for k_i in key)]
                    if len(key) == 0:
                        raise KeyError('Attempting selection of events via '
                                       'multiple/partial matching, but no '
                                       'event matches all criteria.')
            select = np.any(np.atleast_2d([epochs._key_match(k)
                                           for k in key]), axis=0)
            epochs.name = '+'.join(key)
        else:
            select = key if isinstance(key, slice) else np.atleast_1d(key)

        key_selection = epochs.selection[select]
        for k in np.setdiff1d(epochs.selection, key_selection):
            epochs.drop_log[k] = ['IGNORED']
        epochs.selection = key_selection
        epochs.events = np.atleast_2d(epochs.events[select])
        if epochs.preload:
            # ensure that each Epochs instance owns its own data so we can
            # resize later if necessary
            epochs._data = np.require(epochs._data[select], requirements=['O'])
        # update event id to reflect new content of epochs
        epochs.event_id = dict((k, v) for k, v in epochs.event_id.items()
                               if v in epochs.events[:, 2])
        return epochs

    def crop(self, tmin=None, tmax=None, copy=False):
        """Crops a time interval from epochs object.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        copy : bool
            If False epochs is cropped in place.

        Returns
        -------
        epochs : Epochs instance
            The cropped epochs.

        Notes
        -----
        Unlike Python slices, MNE time intervals include both their end points;
        crop(tmin, tmax) returns the interval tmin <= t <= tmax.
        """
        # XXX this could be made to work on non-preloaded data...
        if not self.preload:
            raise RuntimeError('Modifying data of epochs is only supported '
                               'when preloading is used. Use preload=True '
                               'in the constructor.')

        if tmin is None:
            tmin = self.tmin
        elif tmin < self.tmin:
            warnings.warn("tmin is not in epochs' time interval."
                          "tmin is set to epochs.tmin")
            tmin = self.tmin

        if tmax is None:
            tmax = self.tmax
        elif tmax > self.tmax:
            warnings.warn("tmax is not in epochs' time interval."
                          "tmax is set to epochs.tmax")
            tmax = self.tmax

        tmask = _time_mask(self.times, tmin, tmax)
        tidx = np.where(tmask)[0]

        this_epochs = self if not copy else self.copy()
        this_epochs.tmin = this_epochs.times[tidx[0]]
        this_epochs.tmax = this_epochs.times[tidx[-1]]
        this_epochs.times = this_epochs.times[tmask]
        this_epochs._raw_times = this_epochs._raw_times[tmask]
        this_epochs._data = this_epochs._data[:, :, tmask]
        return this_epochs

    @verbose
    def resample(self, sfreq, npad=100, window='boxcar', n_jobs=1,
                 copy=False, verbose=None):
        """Resample preloaded data

        Parameters
        ----------
        sfreq : float
            New sample rate to use
        npad : int
            Amount to pad the start and end of the data.
        window : string or tuple
            Window to use in resampling. See scipy.signal.resample.
        n_jobs : int
            Number of jobs to run in parallel.
        copy : bool
            Whether to operate on a copy of the data (True) or modify data
            in-place (False). Defaults to False.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).
            Defaults to self.verbose.

        Returns
        -------
        epochs : instance of Epochs
            The resampled epochs object.

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        # XXX this could operate on non-preloaded data, too
        if not self.preload:
            raise RuntimeError('Can only resample preloaded data')

        inst = self.copy() if copy else self

        o_sfreq = inst.info['sfreq']
        inst._data = resample(inst._data, sfreq, o_sfreq, npad,
                              n_jobs=n_jobs)
        # adjust indirectly affected variables
        inst.info['sfreq'] = sfreq
        inst.times = (np.arange(inst._data.shape[2], dtype=np.float) /
                      sfreq + inst.times[0])

        return inst

    def copy(self):
        """Return copy of Epochs instance"""
        raw = self._raw
        del self._raw
        new = cp.deepcopy(self)
        self._raw = raw
        new._raw = raw
        return new

    def save(self, fname, split_size='2GB'):
        """Save epochs in a fif file

        Parameters
        ----------
        fname : str
            The name of the file, which should end with -epo.fif or
            -epo.fif.gz.
        split_size : string | int
            Large raw files are automatically split into multiple pieces. This
            parameter specifies the maximum size of each piece. If the
            parameter is an integer, it specifies the size in Bytes. It is
            also possible to pass a human-readable string, e.g., 100MB.
            Note: Due to FIFF file limitations, the maximum split size is 2GB.

            .. versionadded:: 0.10.0

        Notes
        -----
        Bad epochs will be dropped before saving the epochs to disk.
        """
        check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz'))
        split_size = _get_split_size(split_size)

        # to know the length accurately. The get_data() call would drop
        # bad epochs anyway
        self.drop_bad_epochs()
        total_size = self[0].get_data().nbytes * len(self)
        n_parts = int(np.ceil(total_size / float(split_size)))
        epoch_idxs = np.array_split(np.arange(len(self)), n_parts)

        for part_idx, epoch_idx in enumerate(epoch_idxs):
            this_epochs = self[epoch_idx] if n_parts > 1 else self
            # avoid missing event_ids in splits
            this_epochs.event_id = self.event_id
            _save_split(this_epochs, fname, part_idx, n_parts)

    def equalize_event_counts(self, event_ids, method='mintime', copy=True):
        """Equalize the number of trials in each condition

        It tries to make the remaining epochs occurring as close as possible in
        time. This method works based on the idea that if there happened to be
        some time-varying (like on the scale of minutes) noise characteristics
        during a recording, they could be compensated for (to some extent) in
        the equalization process. This method thus seeks to reduce any of
        those effects by minimizing the differences in the times of the events
        in the two sets of epochs. For example, if one had event times
        [1, 2, 3, 4, 120, 121] and the other one had [3.5, 4.5, 120.5, 121.5],
        it would remove events at times [1, 2] in the first epochs and not
        [20, 21].

        Parameters
        ----------
        event_ids : list
            The event types to equalize. Each entry in the list can either be
            a str (single event) or a list of str. In the case where one of
            the entries is a list of str, event_ids in that list will be
            grouped together before equalizing trial counts across conditions.
        method : str
            If 'truncate', events will be truncated from the end of each event
            list. If 'mintime', timing differences between each event list will
            be minimized.
        copy : bool
            If True, a copy of epochs will be returned. Otherwise, the
            function will operate in-place.

        Returns
        -------
        epochs : instance of Epochs
            The modified Epochs instance.
        indices : array of int
            Indices from the original events list that were dropped.

        Notes
        -----
        For example (if epochs.event_id was {'Left': 1, 'Right': 2,
        'Nonspatial':3}:

            epochs.equalize_event_counts([['Left', 'Right'], 'Nonspatial'])

        would equalize the number of trials in the 'Nonspatial' condition with
        the total number of trials in the 'Left' and 'Right' conditions.
        """
        if copy is True:
            epochs = self.copy()
        else:
            epochs = self
        if len(event_ids) == 0:
            raise ValueError('event_ids must have at least one element')
        if not epochs._bad_dropped:
            epochs.drop_bad_epochs()
        # figure out how to equalize
        eq_inds = list()
        for eq in event_ids:
            eq = np.atleast_1d(eq)
            # eq is now a list of types
            key_match = np.zeros(epochs.events.shape[0])
            for key in eq:
                key_match = np.logical_or(key_match, epochs._key_match(key))
            eq_inds.append(np.where(key_match)[0])

        event_times = [epochs.events[e, 0] for e in eq_inds]
        indices = _get_drop_indices(event_times, method)
        # need to re-index indices
        indices = np.concatenate([e[idx] for e, idx in zip(eq_inds, indices)])
        epochs.drop_epochs(indices, reason='EQUALIZED_COUNT')
        # actually remove the indices
        return epochs, indices


class Epochs(_BaseEpochs):
    """Epochs extracted from a Raw instance

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to access associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
        The baseline (a, b) includes both endpoints, i.e. all
        timepoints t such that a <= t <= b.
    picks : array-like of int | None (default)
        Indices of channels to include (if None, all channels are used).
    name : string
        Comment that describes the Epochs data created.
    preload : boolean
        Load all epochs from disk when creating the object
        or wait before accessing each epoch (more memory
        efficient but can be slower).
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )

    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    proj : bool | 'delayed'
        Apply SSP projection vectors. If proj is 'delayed' and reject is not
        None the single epochs will be projected before the rejection
        decision, but used in unprojected state if they are kept.
        This way deciding which projection vectors are good can be postponed
        to the evoked stage without resulting in lower epoch counts and
        without producing results different from early SSP application
        given comparable parameters. Note that in this case baselining,
        detrending and temporal decimation will be postponed.
        If proj is False no projections will be applied which is the
        recommended value if SSPs are not used for cleaning the data.
    decim : int
        Factor by which to downsample the data from the raw file upon import.
        Warning: This simply selects every nth sample, data is not filtered
        here. If data is not properly filtered, aliasing artifacts may occur.
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
    detrend : int | None
        If 0 or 1, the data channels (MEG and EEG) will be detrended when
        loaded. 0 is a constant (DC) detrend, 1 is a linear detrend. None
        is no detrending. Note that detrending is performed before baseline
        correction. If no DC offset is preferred (zeroth order detrending),
        either turn off baseline correction, as this may introduce a DC
        shift, or set baseline correction to use the entire time interval
        (will yield equivalent results but be slower).
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless one
        already exists).
    on_missing : str
        What to do if one or several event ids are not found in the recording.
        Valid keys are 'error' | 'warning' | 'ignore'
        Default is 'error'. If on_missing is 'warning' it will proceed but
        warn, if 'ignore' it will proceed silently. Note.
        If none of the event ids are found in the data, an error will be
        automatically generated irrespective of this parameter.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    Attributes
    ----------
    info: dict
        Measurement info.
    event_id : dict
        Names of conditions corresponding to event_ids.
    ch_names : list of string
        List of channel names.
    selection : array
        List of indices of selected events (not dropped or ignored etc.). For
        example, if the original event array had 4 events and the second event
        has been dropped, this attribute would be np.array([0, 2, 3]).
    preload : bool
        Indicates whether epochs are in memory.
    drop_log : list of lists
        A list of the same length as the event array used to initialize the
        Epochs object. If the i-th original event is still part of the
        selection, drop_log[i] will be an empty list; otherwise it will be
        a list of the reasons the event is not longer in the selection, e.g.:

        'IGNORED' if it isn't part of the current subset defined by the user;
        'NO_DATA' or 'TOO_SHORT' if epoch didn't contain enough data;
        names of channels that exceeded the amplitude threshold;
        'EQUALIZED_COUNTS' (see equalize_event_counts);
        or 'USER' for user-defined reasons (see drop_epochs).
    verbose : bool, str, int, or None
        See above.

    Notes
    -----
    For indexing and slicing:

    epochs[idx] : Epochs
        Return Epochs object with a subset of epochs (supports single
        index and python-style slicing)

    For subset selection using categorial labels:

    epochs['name'] : Epochs
        Return Epochs object with a subset of epochs corresponding to an
        experimental condition as specified by 'name'.

        If conditions are tagged by names separated by '/' (e.g. 'audio/left',
        'audio/right'), and 'name' is not in itself an event key, this selects
        every event whose condition contains the 'name' tag (e.g., 'left'
        matches 'audio/left' and 'visual/left'; but not 'audio_left'). Note
        that tags like 'auditory/left' and 'left/auditory' will be treated the
        same way when accessed using tags.

    epochs[['name_1', 'name_2', ... ]] : Epochs
        Return Epochs object with a subset of epochs corresponding to multiple
        experimental conditions as specified by 'name_1', 'name_2', ... .

        If conditions are separated by '/', selects every item containing every
        list tag (e.g. ['audio', 'left'] selects 'audio/left' and
        'audio/center/left', but not 'audio/right').

    See Also
    --------
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts
    """
    @verbose
    def __init__(self, raw, events, event_id, tmin, tmax, baseline=(None, 0),
                 picks=None, name='Unknown', preload=False, reject=None,
                 flat=None, proj=True, decim=1, reject_tmin=None,
                 reject_tmax=None, detrend=None, add_eeg_ref=True,
                 on_missing='error', verbose=None):
        if not isinstance(raw, _BaseRaw):
            raise ValueError('The first argument to `Epochs` must be an '
                             'instance of `mne.io.Raw`')
        info = cp.deepcopy(raw.info)

        # call _BaseEpochs constructor
        super(Epochs, self).__init__(info, None, events, event_id, tmin, tmax,
                                     baseline=baseline, raw=raw, picks=picks,
                                     name=name, reject=reject, flat=flat,
                                     decim=decim, reject_tmin=reject_tmin,
                                     reject_tmax=reject_tmax, detrend=detrend,
                                     add_eeg_ref=add_eeg_ref, proj=proj,
                                     on_missing=on_missing,
                                     preload_at_end=preload, verbose=verbose)

    @verbose
    def _get_epoch_from_raw(self, idx, verbose=None):
        """Load one epoch from disk"""
        if self._raw is None:
            # This should never happen, as raw=None only if preload=True
            raise ValueError('An error has occurred, no valid raw file found.'
                             ' Please report this to the mne-python '
                             'developers.')
        sfreq = self._raw.info['sfreq']
        event_samp = self.events[idx, 0]
        # Read a data segment
        first_samp = self._raw.first_samp
        start = int(round(event_samp + self.tmin * sfreq)) - first_samp
        stop = start + len(self._raw_times)
        return None if start < 0 else self._raw[self.picks, start:stop][0]


class EpochsArray(_BaseEpochs):
    """Epochs object from numpy array

    Parameters
    ----------
    data : array, shape (n_epochs, n_channels, n_times)
        The channels' time series for each epoch.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    events : array, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
    tmin : float
        Start time before event.
    event_id : int | list of int | dict | None
        The id of the event to consider. If dict,
        the keys can later be used to access associated events. Example:
        dict(auditory=1, visual=3). If int, a dict will be created with
        the id as string. If a list, all events with the IDs specified
        in the list are used. If None, all events will be used with
        and a dict is created with string integer names corresponding
        to the event id integers.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # uV (EEG channels)
                          eog=250e-6 # uV (EOG channels)
                          )

    flat : dict | None
        Rejection parameters based on flatness of signal.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg', and values
        are floats that set the minimum acceptable peak-to-peak amplitude.
        If flat is None then no rejection is done.
    reject_tmin : scalar | None
        Start of the time window used to reject epochs (with the default None,
        the window will start with tmin).
    reject_tmax : scalar | None
        End of the time window used to reject epochs (with the default None,
        the window will end with tmax).
    baseline : None or tuple of length 2 (default: None)
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal to (None, None) all the time
        interval is used.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    See Also
    --------
    io.RawArray, EvokedArray
    """

    @verbose
    def __init__(self, data, info, events, tmin=0, event_id=None,
                 reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, baseline=None, verbose=None):
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)
        if data.ndim != 3:
            raise ValueError('Data must be a 3D array of shape (n_epochs, '
                             'n_channels, n_samples)')

        if len(info['ch_names']) != data.shape[1]:
            raise ValueError('Info and data must have same number of '
                             'channels.')
        if data.shape[0] != len(events):
            raise ValueError('The number of epochs and the number of events'
                             'must match')
        tmax = (data.shape[2] - 1) / info['sfreq'] + tmin
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
        super(EpochsArray, self).__init__(info, data, events, event_id, tmin,
                                          tmax, baseline, reject=reject,
                                          flat=flat, reject_tmin=reject_tmin,
                                          reject_tmax=reject_tmax, decim=1)
        if len(events) != in1d(self.events[:, 2],
                               list(event_id.values())).sum():
            raise ValueError('The events must only contain event numbers from '
                             'event_id')
        for ii, e in enumerate(self._data):
            self._preprocess(e)
        self.drop_bad_epochs()


def combine_event_ids(epochs, old_event_ids, new_event_id, copy=True):
    """Collapse event_ids from an epochs instance into a new event_id

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to operate on.
    old_event_ids : str, or list
        Conditions to collapse together.
    new_event_id : dict, or int
        A one-element dict (or a single integer) for the new
        condition. Note that for safety, this cannot be any
        existing id (in epochs.event_id.values()).
    copy : bool
        If True, a copy of epochs will be returned. Otherwise, the
        function will operate in-place.

    Notes
    -----
    This For example (if epochs.event_id was {'Left': 1, 'Right': 2}:

        combine_event_ids(epochs, ['Left', 'Right'], {'Directional': 12})

    would create a 'Directional' entry in epochs.event_id replacing
    'Left' and 'Right' (combining their trials).
    """
    if copy:
        epochs = epochs.copy()
    old_event_ids = np.asanyarray(old_event_ids)
    if isinstance(new_event_id, int):
        new_event_id = {str(new_event_id): new_event_id}
    else:
        if not isinstance(new_event_id, dict):
            raise ValueError('new_event_id must be a dict or int')
        if not len(list(new_event_id.keys())) == 1:
            raise ValueError('new_event_id dict must have one entry')
    new_event_num = list(new_event_id.values())[0]
    if not isinstance(new_event_num, int):
        raise ValueError('new_event_id value must be an integer')
    if new_event_num in epochs.event_id.values():
        raise ValueError('new_event_id value must not already exist')
    # could use .pop() here, but if a latter one doesn't exist, we're
    # in trouble, so run them all here and pop() later
    old_event_nums = np.array([epochs.event_id[key] for key in old_event_ids])
    # find the ones to replace
    inds = np.any(epochs.events[:, 2][:, np.newaxis] ==
                  old_event_nums[np.newaxis, :], axis=1)
    # replace the event numbers in the events list
    epochs.events[inds, 2] = new_event_num
    # delete old entries
    [epochs.event_id.pop(key) for key in old_event_ids]
    # add the new entry
    epochs.event_id.update(new_event_id)
    return epochs


def equalize_epoch_counts(epochs_list, method='mintime'):
    """Equalize the number of trials in multiple Epoch instances

    It tries to make the remaining epochs occurring as close as possible in
    time. This method works based on the idea that if there happened to be some
    time-varying (like on the scale of minutes) noise characteristics during
    a recording, they could be compensated for (to some extent) in the
    equalization process. This method thus seeks to reduce any of those effects
    by minimizing the differences in the times of the events in the two sets of
    epochs. For example, if one had event times [1, 2, 3, 4, 120, 121] and the
    other one had [3.5, 4.5, 120.5, 121.5], it would remove events at times
    [1, 2] in the first epochs and not [120, 121].

    Note that this operates on the Epochs instances in-place.

    Example:

        equalize_epoch_counts(epochs1, epochs2)

    Parameters
    ----------
    epochs_list : list of Epochs instances
        The Epochs instances to equalize trial counts for.
    method : str
        If 'truncate', events will be truncated from the end of each event
        list. If 'mintime', timing differences between each event list will be
        minimized.
    """
    if not all(isinstance(e, Epochs) for e in epochs_list):
        raise ValueError('All inputs must be Epochs instances')

    # make sure bad epochs are dropped
    [e.drop_bad_epochs() if not e._bad_dropped else None for e in epochs_list]
    event_times = [e.events[:, 0] for e in epochs_list]
    indices = _get_drop_indices(event_times, method)
    for e, inds in zip(epochs_list, indices):
        e.drop_epochs(inds, reason='EQUALIZED_COUNT')


def _get_drop_indices(event_times, method):
    """Helper to get indices to drop from multiple event timing lists"""
    small_idx = np.argmin([e.shape[0] for e in event_times])
    small_e_times = event_times[small_idx]
    if method not in ['mintime', 'truncate']:
        raise ValueError('method must be either mintime or truncate, not '
                         '%s' % method)
    indices = list()
    for e in event_times:
        if method == 'mintime':
            mask = _minimize_time_diff(small_e_times, e)
        else:
            mask = np.ones(e.shape[0], dtype=bool)
            mask[small_e_times.shape[0]:] = False
        indices.append(np.where(np.logical_not(mask))[0])

    return indices


def _minimize_time_diff(t_shorter, t_longer):
    """Find a boolean mask to minimize timing differences"""
    keep = np.ones((len(t_longer)), dtype=bool)
    scores = np.ones((len(t_longer)))
    for iter in range(len(t_longer) - len(t_shorter)):
        scores.fill(np.inf)
        # Check every possible removal to see if it minimizes
        for idx in np.where(keep)[0]:
            keep[idx] = False
            scores[idx] = _area_between_times(t_shorter, t_longer[keep])
            keep[idx] = True
        keep[np.argmin(scores)] = False
    return keep


def _area_between_times(t1, t2):
    """Quantify the difference between two timing sets"""
    x1 = list(range(len(t1)))
    x2 = list(range(len(t2)))
    xs = np.concatenate((x1, x2))
    return np.sum(np.abs(np.interp(xs, x1, t1) - np.interp(xs, x2, t2)))


@verbose
def _is_good(e, ch_names, channel_type_idx, reject, flat, full_report=False,
             ignore_chs=[], verbose=None):
    """Test if data segment e is good according to the criteria
    defined in reject and flat. If full_report=True, it will give
    True/False as well as a list of all offending channels.
    """
    bad_list = list()
    has_printed = False
    checkable = np.ones(len(ch_names), dtype=bool)
    checkable[np.array([c in ignore_chs
                        for c in ch_names], dtype=bool)] = False
    for refl, f, t in zip([reject, flat], [np.greater, np.less], ['', 'flat']):
        if refl is not None:
            for key, thresh in iteritems(refl):
                idx = channel_type_idx[key]
                name = key.upper()
                if len(idx) > 0:
                    e_idx = e[idx]
                    deltas = np.max(e_idx, axis=1) - np.min(e_idx, axis=1)
                    checkable_idx = checkable[idx]
                    idx_deltas = np.where(np.logical_and(f(deltas, thresh),
                                                         checkable_idx))[0]

                    if len(idx_deltas) > 0:
                        ch_name = [ch_names[idx[i]] for i in idx_deltas]
                        if (not has_printed):
                            logger.info('    Rejecting %s epoch based on %s : '
                                        '%s' % (t, name, ch_name))
                            has_printed = True
                        if not full_report:
                            return False
                        else:
                            bad_list.extend(ch_name)

    if not full_report:
        return True
    else:
        if bad_list == []:
            return True, None
        else:
            return False, bad_list


@verbose
def _read_one_epoch_file(f, tree, fname, proj, add_eeg_ref, verbose):

    with f as fid:
        #   Read the measurement info
        info, meas = read_meas_info(fid, tree)
        info['filename'] = fname

        events, mappings = _read_events_fif(fid, tree)

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        if len(processed) == 0:
            raise ValueError('Could not find processed data')

        epochs_node = dir_tree_find(tree, FIFF.FIFFB_EPOCHS)
        if len(epochs_node) == 0:
            raise ValueError('Could not find epochs data')

        my_epochs = epochs_node[0]

        # Now find the data in the block
        name = None
        data = None
        bmin, bmax = None, None
        baseline = None
        selection = None
        drop_log = None
        for k in range(my_epochs['nent']):
            kind = my_epochs['directory'][k].kind
            pos = my_epochs['directory'][k].pos
            if kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, pos)
                first = int(tag.data)
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                tag = read_tag(fid, pos)
                last = int(tag.data)
            elif kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                name = tag.data
            elif kind == FIFF.FIFF_EPOCH:
                tag = read_tag(fid, pos)
                data = tag.data.astype(np.float)
            elif kind == FIFF.FIFF_MNE_BASELINE_MIN:
                tag = read_tag(fid, pos)
                bmin = float(tag.data)
            elif kind == FIFF.FIFF_MNE_BASELINE_MAX:
                tag = read_tag(fid, pos)
                bmax = float(tag.data)
            elif kind == FIFF.FIFFB_MNE_EPOCHS_SELECTION:
                tag = read_tag(fid, pos)
                selection = np.array(tag.data)
            elif kind == FIFF.FIFFB_MNE_EPOCHS_DROP_LOG:
                tag = read_tag(fid, pos)
                drop_log = json.loads(tag.data)

        if bmin is not None or bmax is not None:
            baseline = (bmin, bmax)

        nsamp = last - first + 1
        logger.info('    Found the data of interest:')
        logger.info('        t = %10.2f ... %10.2f ms (%s)'
                    % (1000 * first / info['sfreq'],
                       1000 * last / info['sfreq'], name))
        if info['comps'] is not None:
            logger.info('        %d CTF compensation matrices available'
                        % len(info['comps']))

        # Read the data
        if data is None:
            raise ValueError('Epochs data not found')
        if data.shape[2] != nsamp:
            raise ValueError('Incorrect number of samples (%d instead of %d)'
                             % (data.shape[2], nsamp))

        # Calibrate
        cals = np.array([info['chs'][k]['cal'] *
                         info['chs'][k].get('scale', 1.0)
                         for k in range(info['nchan'])])
        data *= cals[np.newaxis, :, np.newaxis]

        # Put it all together
        tmin = first / info['sfreq']
        tmax = last / info['sfreq']
        event_id = (dict((str(e), e) for e in np.unique(events[:, 2]))
                    if mappings is None else mappings)
        # here we ignore missing events, since users should already be
        # aware of missing events if they have saved data that way
        epochs = _BaseEpochs(info, data, events, event_id, tmin, tmax,
                             baseline, name=name, on_missing='ignore',
                             verbose=verbose)
        activate = False if epochs._delayed_proj else proj
        epochs._projector, epochs.info = setup_proj(info, add_eeg_ref,
                                                    activate=activate)

        # In case epochs didn't have a FIFF.FIFFB_MNE_EPOCHS_SELECTION tag
        # (version < 0.8):
        if selection is None:
            selection = np.arange(len(epochs))
        if drop_log is None:
            drop_log = [[] for _ in range(len(epochs))]  # noqa, analysis:ignore

        epochs.selection = selection
        epochs.drop_log = drop_log
        epochs._bad_dropped = True
    return epochs


@verbose
def read_epochs(fname, proj=True, add_eeg_ref=True, verbose=None):
    """Read epochs from a fif file

    Parameters
    ----------
    fname : str
        The name of the file, which should end with -epo.fif or -epo.fif.gz.
    proj : bool | 'delayed'
        Apply SSP projection vectors. If proj is 'delayed' and reject is not
        None the single epochs will be projected before the rejection
        decision, but used in unprojected state if they are kept.
        This way deciding which projection vectors are good can be postponed
        to the evoked stage without resulting in lower epoch counts and
        without producing results different from early SSP application
        given comparable parameters. Note that in this case baselining,
        detrending and temporal decimation will be postponed.
        If proj is False no projections will be applied which is the
        recommended value if SSPs are not used for cleaning the data.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless one
        already exists).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to raw.verbose.

    Returns
    -------
    epochs : instance of Epochs
        The epochs
    """
    check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz'))

    fnames = [fname]
    epochs = list()
    for fname in fnames:
        logger.info('Reading %s ...' % fname)
        fid, tree, _ = fiff_open(fname)
        next_fname = _get_next_fname(fid, fname, tree)
        epoch = _read_one_epoch_file(fid, tree, fname, proj, add_eeg_ref,
                                     verbose)
        epochs.append(epoch)

        if next_fname is not None:
            fnames.append(next_fname)

    epochs = _concatenate_epochs(epochs, read_file=True)
    epochs._bad_dropped = True
    return epochs


def bootstrap(epochs, random_state=None):
    """Compute epochs selected by bootstrapping

    Parameters
    ----------
    epochs : Epochs instance
        epochs data to be bootstrapped
    random_state : None | int | np.random.RandomState
        To specify the random generator state

    Returns
    -------
    epochs : Epochs instance
        The bootstrap samples
    """
    if not epochs.preload:
        raise RuntimeError('Modifying data of epochs is only supported '
                           'when preloading is used. Use preload=True '
                           'in the constructor.')

    rng = check_random_state(random_state)
    epochs_bootstrap = epochs.copy()
    n_events = len(epochs_bootstrap.events)
    idx = rng.randint(0, n_events, n_events)
    epochs_bootstrap = epochs_bootstrap[idx]
    return epochs_bootstrap


def _check_merge_epochs(epochs_list):
    """Aux function"""
    event_ids = set(tuple(epochs.event_id.items()) for epochs in epochs_list)
    if len(event_ids) == 1:
        event_id = dict(event_ids.pop())
    else:
        raise NotImplementedError("Epochs with unequal values for event_id")

    tmins = set(epochs.tmin for epochs in epochs_list)
    if len(tmins) == 1:
        tmin = tmins.pop()
    else:
        raise NotImplementedError("Epochs with unequal values for tmin")

    tmaxs = set(epochs.tmax for epochs in epochs_list)
    if len(tmaxs) == 1:
        tmax = tmaxs.pop()
    else:
        raise NotImplementedError("Epochs with unequal values for tmax")

    baselines = set(epochs.baseline for epochs in epochs_list)
    if len(baselines) == 1:
        baseline = baselines.pop()
    else:
        raise NotImplementedError("Epochs with unequal values for baseline")

    return event_id, tmin, tmax, baseline


@verbose
def add_channels_epochs(epochs_list, name='Unknown', add_eeg_ref=True,
                        verbose=None):
    """Concatenate channels, info and data from two Epochs objects

    Parameters
    ----------
    epochs_list : list of Epochs
        Epochs object to concatenate.
    name : str
        Comment that describes the Epochs data created.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless there is no
        EEG in the data).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        Defaults to True if any of the input epochs have verbose=True.

    Returns
    -------
    epochs : Epochs
        Concatenated epochs.
    """
    if not all(e.preload for e in epochs_list):
        raise ValueError('All epochs must be preloaded.')

    info = _merge_info([epochs.info for epochs in epochs_list])
    data = [epochs.get_data() for epochs in epochs_list]
    event_id, tmin, tmax, baseline = _check_merge_epochs(epochs_list)

    for d in data:
        if len(d) != len(data[0]):
            raise ValueError('all epochs must be of the same length')

    data = np.concatenate(data, axis=1)

    if len(info['chs']) != data.shape[1]:
        err = "Data shape does not match channel number in measurement info"
        raise RuntimeError(err)

    events = epochs_list[0].events.copy()
    all_same = all(np.array_equal(events, epochs.events)
                   for epochs in epochs_list[1:])
    if not all_same:
        raise ValueError('Events must be the same.')

    proj = any(e.proj for e in epochs_list) or add_eeg_ref

    if verbose is None:
        verbose = any(e.verbose for e in epochs_list)

    epochs = epochs_list[0].copy()
    epochs.info = info
    epochs.event_id = event_id
    epochs.tmin = tmin
    epochs.tmax = tmax
    epochs.baseline = baseline
    epochs.picks = None
    epochs.name = name
    epochs.verbose = verbose
    epochs.events = events
    epochs.preload = True
    epochs._bad_dropped = True
    epochs._data = data
    epochs._projector, epochs.info = setup_proj(epochs.info, add_eeg_ref,
                                                activate=proj)
    return epochs


def _compare_epochs_infos(info1, info2, ind):
    """Compare infos"""
    if not info1['nchan'] == info2['nchan']:
        raise ValueError('epochs[%d][\'info\'][\'nchan\'] must match' % ind)
    if not info1['bads'] == info2['bads']:
        raise ValueError('epochs[%d][\'info\'][\'bads\'] must match' % ind)
    if not info1['sfreq'] == info2['sfreq']:
        raise ValueError('epochs[%d][\'info\'][\'sfreq\'] must match' % ind)
    if not set(info1['ch_names']) == set(info2['ch_names']):
        raise ValueError('epochs[%d][\'info\'][\'ch_names\'] must match' % ind)
    if len(info2['projs']) != len(info1['projs']):
        raise ValueError('SSP projectors in epochs files must be the same')
    if not all(_proj_equal(p1, p2) for p1, p2 in
               zip(info2['projs'], info1['projs'])):
        raise ValueError('SSP projectors in epochs files must be the same')


def _concatenate_epochs(epochs_list, read_file=False):
    """Auxiliary function for concatenating epochs."""
    out = epochs_list[0]
    data = [out.get_data()]
    events = [out.events]
    drop_log = cp.deepcopy(out.drop_log)
    event_id = cp.deepcopy(out.event_id)
    selection = out.selection
    for ii, epochs in enumerate(epochs_list[1:]):
        _compare_epochs_infos(epochs.info, epochs_list[0].info, ii)
        if not np.array_equal(epochs.times, epochs_list[0].times):
            raise ValueError('Epochs must have same times')

        if epochs.baseline != epochs_list[0].baseline:
            raise ValueError('Baseline must be same for all epochs')

        data.append(epochs.get_data())
        events.append(epochs.events)
        selection = np.concatenate((selection, epochs.selection))
        if read_file:
            for k, (a, b) in enumerate(zip(drop_log, epochs.drop_log)):
                if a == ['IGNORED'] and b != ['IGNORED']:
                    drop_log[k] = b
        else:
            drop_log.extend(epochs.drop_log)
        event_id.update(epochs.event_id)
    events = np.concatenate(events, axis=0)
    # do not do this if epochs read from disk are being concatenated
    if read_file is False:
        events[:, 0] = np.arange(len(events))  # arbitrary after concat

    baseline = epochs_list[0].baseline
    out = _BaseEpochs(out.info, np.concatenate(data, axis=0), events, event_id,
                      out.tmin, out.tmax, baseline=baseline, add_eeg_ref=False,
                      proj=False, verbose=out.verbose, on_missing='ignore')
    # We previously only set the drop log here, but we also need to set the
    # selection, too
    if read_file is False:
        selection = np.where([len(d) == 0 for d in drop_log])[0]

    assert len(selection) == len(out.drop_log)
    out.selection = selection
    out.drop_log = drop_log
    out.drop_bad_epochs()
    return out


def concatenate_epochs(epochs_list):
    """Concatenate a list of epochs into one epochs object

    Parameters
    ----------
    epochs_list : list
        list of Epochs instances to concatenate (in order).

    Returns
    -------
    epochs : instance of Epochs
        The result of the concatenation (first Epochs instance passed in).

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    return _concatenate_epochs(epochs_list)
