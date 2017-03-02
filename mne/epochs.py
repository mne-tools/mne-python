# -*- coding: utf-8 -*-

"""Tools for working with epoched data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Denis Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from copy import deepcopy
import json
import os.path as op
from distutils.version import LooseVersion
from numbers import Integral

import numpy as np
import scipy

from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_float_matrix, write_float,
                       write_id, write_string, _get_split_size)
from .io.meas_info import read_meas_info, write_meas_info, _merge_info
from .io.open import fiff_open, _get_next_fname
from .io.tree import dir_tree_find
from .io.tag import read_tag, read_tag_info
from .io.constants import FIFF
from .io.pick import (pick_types, channel_indices_by_type, channel_type,
                      pick_channels, pick_info, _pick_data_channels,
                      _pick_aux_channels, _DATA_CH_TYPES_SPLIT)
from .io.proj import setup_proj, ProjMixin, _proj_equal
from .io.base import BaseRaw, ToDataFrameMixin, TimeMixin
from .bem import _check_origin
from .evoked import EvokedArray, _check_decim
from .baseline import rescale, _log_rescale
from .channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                SetChannelsMixin, InterpolationMixin)
from .filter import resample, detrend, FilterMixin
from .event import _read_events_fif, make_fixed_length_events
from .fixes import _get_args
from .viz import (plot_epochs, plot_epochs_psd, plot_epochs_psd_topomap,
                  plot_epochs_image, plot_topo_image_epochs, plot_drop_log)
from .utils import (check_fname, logger, verbose, _check_type_picks,
                    _time_mask, check_random_state, warn, _pl,
                    sizeof_fmt, SizeMixin, copy_function_doc_to_method_doc)
from .externals.six import iteritems, string_types
from .externals.six.moves import zip


def _save_split(epochs, fname, part_idx, n_parts):
    """Split epochs."""
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
    start_block(fid, FIFF.FIFFB_MNE_EPOCHS)

    # write events out after getting data to ensure bad events are dropped
    data = epochs.get_data()
    start_block(fid, FIFF.FIFFB_MNE_EVENTS)
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, epochs.events.T)
    mapping_ = ';'.join([k + ':' + str(v) for k, v in
                         epochs.event_id.items()])
    write_string(fid, FIFF.FIFF_DESCRIPTION, mapping_)
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)

    # First and last sample
    first = int(round(epochs.tmin * info['sfreq']))  # round just to be safe
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

    end_block(fid, FIFF.FIFFB_MNE_EPOCHS)
    end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
    end_block(fid, FIFF.FIFFB_MEAS)
    end_file(fid)


class BaseEpochs(ProjMixin, ContainsMixin, UpdateChannelsMixin,
                 SetChannelsMixin, InterpolationMixin, FilterMixin,
                 ToDataFrameMixin, TimeMixin, SizeMixin):
    """Abstract base class for Epochs-type classes.

    This class provides basic functionality and should never be instantiated
    directly. See Epochs below for an explanation of the parameters.

    Parameters
    ----------
    info : dict
        A copy of the info dict from the raw object.
    data : ndarray | None
        If ``None``, data will be read from the Raw object. If ndarray, must be
        of shape (n_epochs, n_channels, n_times).
    events : array of int, shape (n_events, 3)
        See `Epochs` docstring.
    event_id : int | list of int | dict | None
        See `Epochs` docstring.
    tmin : float
        See `Epochs` docstring.
    tmax : float
        See `Epochs` docstring.
    baseline : None or tuple of length 2 (default (None, 0))
        See `Epochs` docstring.
    raw : Raw object
        An instance of Raw.
    picks : array-like of int | None (default)
        See `Epochs` docstring.
    name : string
        See `Epochs` docstring.
    reject : dict | None
        See `Epochs` docstring.
    flat : dict | None
        See `Epochs` docstring.
    decim : int
        See `Epochs` docstring.
    reject_tmin : scalar | None
        See `Epochs` docstring.
    reject_tmax : scalar | None
        See `Epochs` docstring.
    detrend : int | None
        See `Epochs` docstring.
    add_eeg_ref : bool
        See `Epochs` docstring.
    proj : bool | 'delayed'
        See `Epochs` docstring.
    on_missing : str
        See `Epochs` docstring.
    preload_at_end : bool
        Load all epochs from disk when creating the object
        or wait before accessing each epoch (more memory
        efficient but can be slower).
    selection : iterable | None
        Iterable of indices of selected epochs. If ``None``, will be
        automatically generated, corresponding to all non-zero events.
    drop_log : list | None
        List of lists of strings indicating which epochs have been marked to be
        ignored.
    filename : str | None
        The filename (if the epochs are read from disk).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        raw.verbose.

    Notes
    -----
    The `BaseEpochs` class is public to allow for stable type-checking in user
    code (i.e., ``isinstance(my_epochs, BaseEpochs)``) but should not be used
    as a constructor for Epochs objects (use instead `mne.Epochs`).
    """

    def __init__(self, info, data, events, event_id=None, tmin=-0.2, tmax=0.5,
                 baseline=(None, 0), raw=None,
                 picks=None, name='Unknown', reject=None, flat=None,
                 decim=1, reject_tmin=None, reject_tmax=None, detrend=None,
                 add_eeg_ref=False, proj=True, on_missing='error',
                 preload_at_end=False, selection=None, drop_log=None,
                 filename=None, verbose=None):  # noqa: D102
        self.verbose = verbose
        self.name = name

        if on_missing not in ['error', 'warning', 'ignore']:
            raise ValueError('on_missing must be one of: error, '
                             'warning, ignore. Got: %s' % on_missing)

        # check out event_id dict
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
        elif isinstance(event_id, dict):
            if not all(isinstance(v, Integral) for v in event_id.values()):
                raise ValueError('Event IDs must be of type integer')
            if not all(isinstance(k, string_types) for k in event_id):
                raise ValueError('Event names must be of type str')
            event_id = deepcopy(event_id)
        elif isinstance(event_id, list):
            if not all(isinstance(v, Integral) for v in event_id):
                raise ValueError('Event IDs must be of type integer')
            event_id = dict(zip((str(i) for i in event_id), event_id))
        elif isinstance(event_id, Integral):
            event_id = {str(event_id): event_id}
        else:
            raise ValueError('event_id must be dict or int.')
        for k, v in event_id.items():
            event_id[k] = int(v)  # make sure values are of type int
        self.event_id = event_id
        del event_id

        if events is not None:  # RtEpochs can have events=None

            if events.dtype.kind not in ['i', 'u']:
                raise ValueError('events must be an array of type int')
            if events.ndim != 2 or events.shape[1] != 3:
                raise ValueError('events must be 2D with 3 columns')
            for key, val in self.event_id.items():
                if val not in events[:, 2]:
                    msg = ('No matching events found for %s '
                           '(event id %i)' % (key, val))
                    if on_missing == 'error':
                        raise ValueError(msg)
                    elif on_missing == 'warning':
                        warn(msg)
                    else:  # on_missing == 'ignore':
                        pass

            values = list(self.event_id.values())
            selected = np.in1d(events[:, 2], values)
            if selection is None:
                self.selection = np.where(selected)[0]
            else:
                self.selection = selection
            if drop_log is None:
                self.drop_log = [list() if k in self.selection else ['IGNORED']
                                 for k in range(len(events))]
            else:
                self.drop_log = drop_log
            events = events[selected]
            if len(np.unique(events[:, 0])) != len(events):
                raise RuntimeError('Event time samples were not unique')
            n_events = len(events)
            if n_events > 1:
                if np.diff(events.astype(np.int64)[:, 0]).min() <= 0:
                    warn('The events passed to the Epochs constructor are not '
                         'chronologically ordered.', RuntimeWarning)

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
        if (detrend not in [None, 0, 1]) or isinstance(detrend, bool):
            raise ValueError('detrend must be None, 0, or 1')

        # check that baseline is in available data
        if tmin > tmax:
            raise ValueError('tmin has to be less than or equal to tmax')
        _check_baseline(baseline, tmin, tmax, info['sfreq'])
        _log_rescale(baseline)
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
            self.info = pick_info(self.info, picks)
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
        start_idx = int(round(tmin * sfreq))
        self._raw_times = np.arange(start_idx,
                                    int(round(tmax * sfreq)) + 1) / sfreq
        self.times = self._raw_times.copy()
        self._decim = 1
        self.decimate(decim)

        # setup epoch rejection
        self.reject = None
        self.flat = None
        self._reject_setup(reject, flat)

        # do the rest
        valid_proj = [True, 'delayed', False]
        if proj not in valid_proj:
            raise ValueError('"proj" must be one of %s, not %s'
                             % (valid_proj, proj))
        if proj == 'delayed':
            self._do_delayed_proj = True
            logger.info('Entering delayed SSP mode.')
        else:
            self._do_delayed_proj = False
        add_eeg_ref = _dep_eeg_ref(add_eeg_ref)
        activate = False if self._do_delayed_proj else proj
        self._projector, self.info = setup_proj(self.info, add_eeg_ref,
                                                activate=activate)
        if preload_at_end:
            assert self._data is None
            assert self.preload is False
            self.load_data()  # this will do the projection
        elif proj is True and self._projector is not None and data is not None:
            # let's make sure we project if data was provided and proj
            # requested
            # we could do this with np.einsum, but iteration should be
            # more memory safe in most instances
            for ii, epoch in enumerate(self._data):
                self._data[ii] = np.dot(self._projector, epoch)
        self._filename = str(filename) if filename is not None else filename

    def load_data(self):
        """Load the data if not already preloaded.

        Returns
        -------
        epochs : instance of Epochs
            The epochs object.

        Notes
        -----
        This function operates in-place.

        .. versionadded:: 0.10.0
        """
        if self.preload:
            return self
        self._data = self._get_data()
        self.preload = True
        self._decim_slice = slice(None, None, None)
        self._decim = 1
        self._raw_times = self.times
        assert self._data.shape[-1] == len(self.times)
        return self

    @verbose
    def decimate(self, decim, offset=0, verbose=None):
        """Decimate the epochs.

        .. note:: No filtering is performed. To avoid aliasing, ensure
                  your data are properly lowpassed.

        Parameters
        ----------
        decim : int
            The amount to decimate data.
        offset : int
            Apply an offset to where the decimation starts relative to the
            sample corresponding to t=0. The offset is in samples at the
            current sampling rate.

            .. versionadded:: 0.12

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        epochs : instance of Epochs
            The decimated Epochs object.

        See Also
        --------
        mne.Evoked.decimate
        mne.Epochs.resample
        mne.io.Raw.resample

        Notes
        -----
        Decimation can be done multiple times. For example,
        ``epochs.decimate(2).decimate(2)`` will be the same as
        ``epochs.decimate(4)``.

        .. versionadded:: 0.10.0
        """
        decim, offset, new_sfreq = _check_decim(self.info, decim, offset)
        start_idx = int(round(-self._raw_times[0] * (self.info['sfreq'] *
                                                     self._decim)))
        self._decim *= decim
        i_start = start_idx % self._decim + offset
        decim_slice = slice(i_start, None, self._decim)
        self.info['sfreq'] = new_sfreq
        if self.preload:
            self._data = self._data[:, :, decim_slice].copy()
            self._raw_times = self._raw_times[decim_slice].copy()
            self._decim_slice = slice(None)
            self._decim = 1
            self.times = self._raw_times
        else:
            self._decim_slice = decim_slice
            self.times = self._raw_times[self._decim_slice]
        return self

    @verbose
    def apply_baseline(self, baseline=(None, 0), verbose=None):
        """Baseline correct epochs.

        Parameters
        ----------
        baseline : tuple of length 2
            The time interval to apply baseline correction. If None do not
            apply it. If baseline is (a, b) the interval is between "a (s)" and
            "b (s)". If a is None the beginning of the data is used and if b is
            None then b is set to the end of the interval. If baseline is equal
            to (None, None) all the time interval is used. Correction is
            applied by computing mean of the baseline period and subtracting it
            from the data. The baseline (a, b) includes both endpoints, i.e.
            all timepoints t such that a <= t <= b.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).

        Returns
        -------
        epochs : instance of Epochs
            The baseline-corrected Epochs object.

        Notes
        -----
        Baseline correction can be done multiple times.

        .. versionadded:: 0.10.0
        """
        if not self.preload:
            # Eventually we can relax this restriction, but it will require
            # more careful checking of baseline (e.g., refactor with the
            # BaseEpochs.__init__ checks)
            raise RuntimeError('Data must be loaded to apply a new baseline')
        _check_baseline(baseline, self.tmin, self.tmax, self.info['sfreq'])

        picks = _pick_data_channels(self.info, exclude=[], with_ref_meg=True)
        picks_aux = _pick_aux_channels(self.info, exclude=[])
        picks = np.sort(np.concatenate((picks, picks_aux)))

        data = self._data
        data[:, picks, :] = rescale(data[:, picks, :], self.times, baseline,
                                    copy=False)
        self.baseline = baseline

        return self

    def _reject_setup(self, reject, flat):
        """Set self._reject_time and self._channel_type_idx."""
        idx = channel_indices_by_type(self.info)
        reject = deepcopy(reject) if reject is not None else dict()
        flat = deepcopy(flat) if flat is not None else dict()
        for rej, kind in zip((reject, flat), ('reject', 'flat')):
            if not isinstance(rej, dict):
                raise TypeError('reject and flat must be dict or None, not %s'
                                % type(rej))
            bads = set(rej.keys()) - set(idx.keys())
            if len(bads) > 0:
                raise KeyError('Unknown channel types found in %s: %s'
                               % (kind, bads))

        for key in idx.keys():
            # don't throw an error if rejection/flat would do nothing
            if len(idx[key]) == 0 and (np.isfinite(reject.get(key, np.inf)) or
                                       flat.get(key, -1) >= 0):
                # This is where we could eventually add e.g.
                # self.allow_missing_reject_keys check to allow users to
                # provide keys that don't exist in data
                raise ValueError("No %s channel found. Cannot reject based on "
                                 "%s." % (key.upper(), key.upper()))

        # check for invalid values
        for rej, kind in zip((reject, flat), ('Rejection', 'Flat')):
            for key, val in rej.items():
                if val is None or val < 0:
                    raise ValueError('%s value must be a number >= 0, not "%s"'
                                     % (kind, val))

        # now check to see if our rejection and flat are getting more
        # restrictive
        old_reject = self.reject if self.reject is not None else dict()
        old_flat = self.flat if self.flat is not None else dict()
        bad_msg = ('{kind}["{key}"] == {new} {op} {old} (old value), new '
                   '{kind} values must be at least as stringent as '
                   'previous ones')
        for key in set(reject.keys()).union(old_reject.keys()):
            old = old_reject.get(key, np.inf)
            new = reject.get(key, np.inf)
            if new > old:
                raise ValueError(bad_msg.format(kind='reject', key=key,
                                                new=new, old=old, op='>'))
        for key in set(flat.keys()).union(old_flat.keys()):
            old = old_flat.get(key, -np.inf)
            new = flat.get(key, -np.inf)
            if new < old:
                raise ValueError(bad_msg.format(kind='flat', key=key,
                                                new=new, old=old, op='<'))

        # after validation, set parameters
        self._bad_dropped = False
        self._channel_type_idx = idx
        self.reject = reject if len(reject) > 0 else None
        self.flat = flat if len(flat) > 0 else None

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
        """Determine if epoch is good."""
        if isinstance(data, string_types):
            return False, [data]
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
    def _detrend_offset_decim(self, epoch, verbose=None):
        """Aux Function: detrend, baseline correct, offset, decim.

        Note: operates inplace
        """
        if (epoch is None) or isinstance(epoch, string_types):
            return epoch

        # Detrend
        if self.detrend is not None:
            picks = _pick_data_channels(self.info, exclude=[])
            epoch[picks] = detrend(epoch[picks], self.detrend, axis=1)

        # Baseline correct
        picks = pick_types(self.info, meg=True, eeg=True, stim=False,
                           ref_meg=True, eog=True, ecg=True, seeg=True,
                           emg=True, bio=True, ecog=True, fnirs=True,
                           exclude=[])
        epoch[picks] = rescale(epoch[picks], self._raw_times, self.baseline,
                               copy=False, verbose=False)

        # handle offset
        if self._offset is not None:
            epoch += self._offset

        # Decimate if necessary (i.e., epoch not preloaded)
        epoch = epoch[:, self._decim_slice]
        return epoch

    def iter_evoked(self):
        """Iterate over epochs as a sequence of Evoked objects.

        The Evoked objects yielded will each contain a single epoch (i.e., no
        averaging is performed).
        """
        self._current = 0

        while True:
            out = self.next(True)
            if out is None:
                return  # properly signal the end of iteration
            data, event_id = out
            tmin = self.times[0]
            info = deepcopy(self.info)

            yield EvokedArray(data, info, tmin, comment=str(event_id))

    def subtract_evoked(self, evoked=None):
        """Subtract an evoked response from each epoch.

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
            picks = _pick_data_channels(self.info, exclude=[])
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
                       _DATA_CH_TYPES_SPLIT]
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
            warn('Evoked has SSP applied while Epochs has not.')
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
        """Wrapper for Py3k."""
        return self.next(*args, **kwargs)

    def average(self, picks=None):
        """Compute average of epochs.

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG, EEG, SEEG, ECoG, and fNIRS channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        evoked : instance of Evoked
            The averaged epochs.

        Notes
        -----
        Computes an average of all epochs in the instance, even if
        they correspond to different conditions. To average by condition,
        do ``epochs[condition].average()`` for each condition separately.

        When picks is None and epochs contain only ICA channels, no channels
        are selected, resulting in an error. This is because ICA channels
        are not considered data channels (they are of misc type) and only data
        channels are selected when picks is None.
        """
        return self._compute_mean_or_stderr(picks, 'ave')

    def standard_error(self, picks=None):
        """Compute standard error over epochs.

        Parameters
        ----------
        picks : array-like of int | None
            If None only MEG, EEG, SEEG, ECoG, and fNIRS channels are kept
            otherwise the channels indices in picks are kept.

        Returns
        -------
        evoked : instance of Evoked
            The standard error over epochs.
        """
        return self._compute_mean_or_stderr(picks, 'stderr')

    def _compute_mean_or_stderr(self, picks, mode='ave'):
        """Compute the mean or std over epochs and return Evoked."""
        _do_std = True if mode == 'stderr' else False

        # if instance contains ICA channels they won't be included unless picks
        # is specified
        if picks is None:
            check_ICA = [x.startswith('ICA') for x in self.ch_names]
            if np.all(check_ICA):
                raise TypeError('picks must be specified (i.e. not None) for '
                                'ICA channel data')
            elif np.any(check_ICA):
                warn('ICA channels will not be included unless explicitly '
                     'selected in picks')

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
                data_mean = data.copy()
                data.fill(0.)
                for e in self:
                    data += (e - data_mean) ** 2
                data = np.sqrt(data / n_events)

        if not _do_std:
            kind = 'average'
        else:
            kind = 'standard_error'
            data /= np.sqrt(n_events)
        return self._evoked_from_epoch_data(data, self.info, picks, n_events,
                                            kind)

    def _evoked_from_epoch_data(self, data, info, picks, n_events, kind):
        """Create an evoked object from epoch data."""
        info = deepcopy(info)
        evoked = EvokedArray(data, info, tmin=self.times[0],
                             comment=self.name, nave=n_events, kind=kind,
                             verbose=self.verbose)
        # XXX: above constructor doesn't recreate the times object precisely
        evoked.times = self.times.copy()

        # pick channels
        if picks is None:
            picks = _pick_data_channels(evoked.info, exclude=[])

        ch_names = [evoked.ch_names[p] for p in picks]
        evoked.pick_channels(ch_names)

        if len(evoked.info['ch_names']) == 0:
            raise ValueError('No data channel found when averaging.')

        if evoked.nave < 1:
            warn('evoked object is empty (based on less than 1 epoch)')

        return evoked

    @property
    def ch_names(self):
        """Channel names."""
        return self.info['ch_names']

    @copy_function_doc_to_method_doc(plot_epochs)
    def plot(self, picks=None, scalings=None, n_epochs=20, n_channels=20,
             title=None, events=None, event_colors=None, show=True,
             block=False):
        return plot_epochs(self, picks=picks, scalings=scalings,
                           n_epochs=n_epochs, n_channels=n_channels,
                           title=title, events=events,
                           event_colors=event_colors, show=show, block=block)

    @copy_function_doc_to_method_doc(plot_epochs_psd)
    def plot_psd(self, fmin=0, fmax=np.inf, tmin=None, tmax=None, proj=False,
                 bandwidth=None, adaptive=False, low_bias=True,
                 normalization='length', picks=None, ax=None, color='black',
                 area_mode='std', area_alpha=0.33, dB=True, n_jobs=1,
                 show=True, verbose=None):
        return plot_epochs_psd(self, fmin=fmin, fmax=fmax, tmin=tmin,
                               tmax=tmax, proj=proj, bandwidth=bandwidth,
                               adaptive=adaptive, low_bias=low_bias,
                               normalization=normalization, picks=picks, ax=ax,
                               color=color, area_mode=area_mode,
                               area_alpha=area_alpha, dB=dB, n_jobs=n_jobs,
                               show=show, verbose=verbose)

    @copy_function_doc_to_method_doc(plot_epochs_psd_topomap)
    def plot_psd_topomap(self, bands=None, vmin=None, vmax=None, tmin=None,
                         tmax=None, proj=False, bandwidth=None, adaptive=False,
                         low_bias=True, normalization='length', ch_type=None,
                         layout=None, cmap='RdBu_r', agg_fun=None, dB=True,
                         n_jobs=1, normalize=False, cbar_fmt='%0.3f',
                         outlines='head', axes=None, show=True, verbose=None):
        return plot_epochs_psd_topomap(
            self, bands=bands, vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax,
            proj=proj, bandwidth=bandwidth, adaptive=adaptive,
            low_bias=low_bias, normalization=normalization, ch_type=ch_type,
            layout=layout, cmap=cmap, agg_fun=agg_fun, dB=dB, n_jobs=n_jobs,
            normalize=normalize, cbar_fmt=cbar_fmt, outlines=outlines,
            axes=axes, show=show, verbose=verbose)

    @copy_function_doc_to_method_doc(plot_topo_image_epochs)
    def plot_topo_image(self, layout=None, sigma=0., vmin=None, vmax=None,
                        colorbar=True, order=None, cmap='RdBu_r',
                        layout_scale=.95, title=None, scalings=None,
                        border='none', fig_facecolor='k', fig_background=None,
                        font_color='w', show=True):
        return plot_topo_image_epochs(
            self, layout=layout, sigma=sigma, vmin=vmin, vmax=vmax,
            colorbar=colorbar, order=order, cmap=cmap,
            layout_scale=layout_scale, title=title, scalings=scalings,
            border=border, fig_facecolor=fig_facecolor,
            fig_background=fig_background, font_color=font_color, show=show)

    @verbose
    def drop_bad(self, reject='existing', flat='existing', verbose=None):
        """Drop bad epochs without retaining the epochs data.

        Should be used before slicing operations.

        .. warning:: This operation is slow since all epochs have to be read
                     from disk. To avoid reading epochs from disk multiple
                     times, use :func:`mne.Epochs.load_data()`.

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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        epochs : instance of Epochs
            The epochs with bad epochs dropped. Operates in-place.

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
        return self

    def drop_log_stats(self, ignore=('IGNORED',)):
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

    @copy_function_doc_to_method_doc(plot_drop_log)
    def plot_drop_log(self, threshold=0, n_max_plot=20, subject='Unknown',
                      color=(0.9, 0.9, 0.9), width=0.8, ignore=('IGNORED',),
                      show=True):
        if not self._bad_dropped:
            raise ValueError("You cannot use plot_drop_log since bad "
                             "epochs have not yet been dropped. "
                             "Use epochs.drop_bad().")
        return plot_drop_log(self.drop_log, threshold, n_max_plot, subject,
                             color=color, width=width, ignore=ignore,
                             show=show)

    @copy_function_doc_to_method_doc(plot_epochs_image)
    def plot_image(self, picks=None, sigma=0., vmin=None,
                   vmax=None, colorbar=True, order=None, show=True,
                   units=None, scalings=None, cmap='RdBu_r',
                   fig=None, axes=None, overlay_times=None):
        return plot_epochs_image(self, picks=picks, sigma=sigma, vmin=vmin,
                                 vmax=vmax, colorbar=colorbar, order=order,
                                 show=show, units=units, scalings=scalings,
                                 cmap=cmap, fig=fig, axes=axes,
                                 overlay_times=overlay_times)

    @verbose
    def drop(self, indices, reason='USER', verbose=None):
        """Drop epochs based on indices or boolean mask.

        .. note:: The indices refer to the current set of undropped epochs
                  rather than the complete set of dropped and undropped epochs.
                  They are therefore not necessarily consistent with any
                  external indices (e.g., behavioral logs). To drop epochs
                  based on external criteria, do not use the ``preload=True``
                  flag when constructing an Epochs object, and call this
                  method before calling the :func:`mne.Epochs.drop_bad` or
                  :func:`mne.Epochs.load_data` methods.

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
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        epochs : instance of Epochs
            The epochs with indices dropped. Operates in-place.
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
        logger.info('Dropped %d epoch%s' % (count, _pl(count)))
        return self

    def _get_epoch_from_raw(self, idx, verbose=None):
        """Method to get a given epoch from disk."""
        raise NotImplementedError

    def _project_epoch(self, epoch):
        """Process a raw epoch based on the delayed param."""
        # whenever requested, the first epoch is being projected.
        if (epoch is None) or isinstance(epoch, string_types):
            # can happen if t < 0 or reject based on annotations
            return epoch
        proj = self._do_delayed_proj or self.proj
        if self._projector is not None and proj is True:
            epoch = np.dot(self._projector, epoch)
        return epoch

    @verbose
    def _get_data(self, out=True, verbose=None):
        """Load all data, dropping bad epochs along the way.

        Parameters
        ----------
        out : bool
            Return the data. Setting this to False is used to reject bad
            epochs without caching all the data, which saves memory.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.
        """
        n_events = len(self.events)
        # in case there are no good events
        if self.preload:
            # we will store our result in our existing array
            data = self._data
        else:
            # we start out with an empty array, allocate only if necessary
            data = np.empty((0, len(self.info['ch_names']), len(self.times)))
            logger.info('Loading data for %s events and %s original time '
                        'points ...' % (n_events, len(self._raw_times)))
        if self._bad_dropped:
            if not out:
                return
            if self.preload:
                return data

            # we need to load from disk, drop, and return data
            for idx in range(n_events):
                # faster to pre-allocate memory here
                epoch_noproj = self._get_epoch_from_raw(idx)
                epoch_noproj = self._detrend_offset_decim(epoch_noproj)
                if self._do_delayed_proj:
                    epoch_out = epoch_noproj
                else:
                    epoch_out = self._project_epoch(epoch_noproj)
                if idx == 0:
                    data = np.empty((n_events, len(self.ch_names),
                                     len(self.times)), dtype=epoch_out.dtype)
                data[idx] = epoch_out
        else:
            # bads need to be dropped, this might occur after a preload
            # e.g., when calling drop_bad w/new params
            good_idx = []
            n_out = 0
            assert n_events == len(self.selection)
            for idx, sel in enumerate(self.selection):
                if self.preload:  # from memory
                    if self._do_delayed_proj:
                        epoch_noproj = self._data[idx]
                        epoch = self._project_epoch(epoch_noproj)
                    else:
                        epoch_noproj = None
                        epoch = self._data[idx]
                else:  # from disk
                    epoch_noproj = self._get_epoch_from_raw(idx)
                    epoch_noproj = self._detrend_offset_decim(epoch_noproj)
                    epoch = self._project_epoch(epoch_noproj)

                epoch_out = epoch_noproj if self._do_delayed_proj else epoch
                is_good, offending_reason = self._is_good_epoch(epoch)
                if not is_good:
                    self.drop_log[sel] += offending_reason
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
        """Get all epochs as a 3D array.

        Returns
        -------
        data : array of shape (n_epochs, n_channels, n_times)
            A view on epochs data.
        """
        return self._get_data()

    def __len__(self):
        """The number of epochs.

        Returns
        -------
        n_epochs : int
            The number of remaining epochs.

        Notes
        -----
        This function only works if bad epochs have been dropped.

        Examples
        --------
        This can be used as::

            >>> epochs.drop_bad()  # doctest: +SKIP
            >>> len(epochs)  # doctest: +SKIP
            43
            >>> len(epochs.events)  # doctest: +SKIP
            43

        """
        if not self._bad_dropped:
            raise RuntimeError('Since bad epochs have not been dropped, the '
                               'length of the Epochs is not known. Load the '
                               'Epochs with preload=True, or call '
                               'Epochs.drop_bad(). To find the number '
                               'of events in the Epochs, use '
                               'len(Epochs.events).')
        return len(self.events)

    def __iter__(self):
        """Function to make iteration over epochs easy.

        Notes
        -----
        This enables the use of this Python pattern::

            >>> for epoch in epochs:  # doctest: +SKIP
            >>>     print(epoch)  # doctest: +SKIP

        Where ``epoch`` is given by successive outputs of
        :func:`mne.Epochs.next`.
        """
        self._current = 0
        while True:
            x = self.next()
            if x is None:
                return
            yield x

    def next(self, return_event_id=False):
        """Iterate over epoch data.

        Parameters
        ----------
        return_event_id : bool
            If True, return both the epoch data and an event_id.

        Returns
        -------
        epoch : array of shape (n_channels, n_times)
            The epoch data.
        event_id : int
            The event id. Only returned if ``return_event_id`` is ``True``.
        """
        if self.preload:
            if self._current >= len(self._data):
                return  # signal the end
            epoch = self._data[self._current]
            self._current += 1
        else:
            is_good = False
            while not is_good:
                if self._current >= len(self.events):
                    return  # signal the end properly
                epoch_noproj = self._get_epoch_from_raw(self._current)
                epoch_noproj = self._detrend_offset_decim(epoch_noproj)
                epoch = self._project_epoch(epoch_noproj)
                self._current += 1
                is_good, _ = self._is_good_epoch(epoch)
            # If delayed-ssp mode, pass 'virgin' data after rejection decision.
            if self._do_delayed_proj:
                epoch = epoch_noproj

        if not return_event_id:
            return epoch
        else:
            return epoch, self.events[self._current - 1][-1]

        return epoch if not return_event_id else epoch, self.event_id

    @property
    def tmin(self):
        """First time point."""
        return self.times[0]

    @property
    def filename(self):
        """The filename."""
        return self._filename

    @property
    def tmax(self):
        """Last time point."""
        return self.times[-1]

    def __repr__(self):
        """Build string representation."""
        s = 'n_events : %s ' % len(self.events)
        s += '(all good)' if self._bad_dropped else '(good & bad)'
        s += ', tmin : %s (s)' % self.tmin
        s += ', tmax : %s (s)' % self.tmax
        s += ', baseline : %s' % str(self.baseline)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        s += ', data%s loaded' % ('' if self.preload else ' not')
        if len(self.event_id) > 1:
            counts = ['%r: %i' % (k, sum(self.events[:, 2] == v))
                      for k, v in sorted(self.event_id.items())]
            s += ',\n %s' % ', '.join(counts)
        class_name = self.__class__.__name__
        class_name = 'Epochs' if class_name == 'BaseEpochs' else class_name
        return '<%s  |  %s>' % (class_name, s)

    def _keys_to_idx(self, keys):
        """Find entries in event dict."""
        return np.array([self.events[:, 2] == self.event_id[k]
                         for k in _hid_match(self.event_id, keys)]).any(axis=0)

    def __getitem__(self, item):
        """Return an Epochs object with a copied subset of epochs.

        Parameters
        ----------
        item : slice, array-like, str, or list
            See below for use cases.

        Returns
        -------
        epochs : instance of Epochs
            See below for use cases.

        Notes
        -----
        Epochs can be accessed as ``epochs[...]`` in several ways:

            1. ``epochs[idx]``: Return ``Epochs`` object with a subset of
               epochs (supports single index and python-style slicing).

            2. ``epochs['name']``: Return ``Epochs`` object with a copy of the
               subset of epochs corresponding to an experimental condition as
               specified by 'name'.

               If conditions are tagged by names separated by '/' (e.g.
               'audio/left', 'audio/right'), and 'name' is not in itself an
               event key, this selects every event whose condition contains
               the 'name' tag (e.g., 'left' matches 'audio/left' and
               'visual/left'; but not 'audio_left'). Note that tags like
               'auditory/left' and 'left/auditory' will be treated the
               same way when accessed using tags.

            3. ``epochs[['name_1', 'name_2', ... ]]``: Return ``Epochs`` object
               with a copy of the subset of epochs corresponding to multiple
               experimental conditions as specified by
               ``'name_1', 'name_2', ...`` .

               If conditions are separated by '/', selects every item
               containing every list tag (e.g. ['audio', 'left'] selects
               'audio/left' and 'audio/center/left', but not 'audio/right').

        """
        data = self._data
        del self._data
        epochs = self.copy()
        self._data, epochs._data = data, data
        del self

        if isinstance(item, string_types):
            item = [item]

        if isinstance(item, (list, tuple)) and \
                isinstance(item[0], string_types):
            select = epochs._keys_to_idx(item)
            epochs.name = '+'.join(item)
        else:
            select = item if isinstance(item, slice) else np.atleast_1d(item)

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

    def crop(self, tmin=None, tmax=None):
        """Crop a time interval from epochs object.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.

        Returns
        -------
        epochs : instance of Epochs
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
            warn('tmin is not in epochs time interval. tmin is set to '
                 'epochs.tmin')
            tmin = self.tmin

        if tmax is None:
            tmax = self.tmax
        elif tmax > self.tmax:
            warn('tmax is not in epochs time interval. tmax is set to '
                 'epochs.tmax')
            tmax = self.tmax

        tmask = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'])
        self.times = self.times[tmask]
        self._raw_times = self._raw_times[tmask]
        self._data = self._data[:, :, tmask]
        return self

    @verbose
    def resample(self, sfreq, npad='auto', window='boxcar', n_jobs=1,
                 verbose=None):
        """Resample data.

        .. note:: Data must be loaded.

        Parameters
        ----------
        sfreq : float
            New sample rate to use
        npad : int | str
            Amount to pad the start and end of the data.
            Can also be "auto" to use a padding that will result in
            a power-of-two size (can be much faster).
        window : string or tuple
            Window to use in resampling. See :func:`scipy.signal.resample`.
        n_jobs : int
            Number of jobs to run in parallel.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` :ref:`Logging documentation <tut_logging>` for
            more). Defaults to self.verbose.

        Returns
        -------
        epochs : instance of Epochs
            The resampled epochs object.

        See Also
        --------
        mne.Epochs.savgol_filter
        mne.io.Raw.resample

        Notes
        -----
        For some data, it may be more accurate to use npad=0 to reduce
        artifacts. This is dataset dependent -- check your data!
        """
        # XXX this could operate on non-preloaded data, too
        if not self.preload:
            raise RuntimeError('Can only resample preloaded data')
        o_sfreq = self.info['sfreq']
        self._data = resample(self._data, sfreq, o_sfreq, npad, window=window,
                              n_jobs=n_jobs)
        # adjust indirectly affected variables
        self.info['sfreq'] = float(sfreq)
        self.times = (np.arange(self._data.shape[2], dtype=np.float) /
                      sfreq + self.times[0])
        self._raw_times = self.times
        return self

    def copy(self):
        """Return copy of Epochs instance."""
        raw = self._raw
        del self._raw
        new = deepcopy(self)
        self._raw = raw
        new._raw = raw
        return new

    def save(self, fname, split_size='2GB'):
        """Save epochs in a fif file.

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
        self.drop_bad()
        total_size = self[0].get_data().nbytes * len(self)
        n_parts = int(np.ceil(total_size / float(split_size)))
        epoch_idxs = np.array_split(np.arange(len(self)), n_parts)

        for part_idx, epoch_idx in enumerate(epoch_idxs):
            this_epochs = self[epoch_idx] if n_parts > 1 else self
            # avoid missing event_ids in splits
            this_epochs.event_id = self.event_id
            _save_split(this_epochs, fname, part_idx, n_parts)

    def equalize_event_counts(self, event_ids, method='mintime'):
        """Equalize the number of trials in each condition.

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
            In the case where partial matching is used (using '/' in
            `event_ids`), `event_ids` will be matched according to the
            provided tags, that is, processing works as if the event_ids
            matched by the provided tags had been supplied instead.
            The event_ids must identify nonoverlapping subsets of the epochs.
        method : str
            If 'truncate', events will be truncated from the end of each event
            list. If 'mintime', timing differences between each event list
            will be minimized.

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

        If multiple indices are provided (e.g. 'Left' and 'Right' in the
        example above), it is not guaranteed that after equalization, the
        conditions will contribute evenly. E.g., it is possible to end up
        with 70 'Nonspatial' trials, 69 'Left' and 1 'Right'.
        """
        if len(event_ids) == 0:
            raise ValueError('event_ids must have at least one element')
        if not self._bad_dropped:
            self.drop_bad()
        # figure out how to equalize
        eq_inds = list()

        # deal with hierarchical tags
        ids = self.event_id
        orig_ids = list(event_ids)
        tagging = False
        if "/" in "".join(ids):
            # make string inputs a list of length 1
            event_ids = [[x] if isinstance(x, string_types) else x
                         for x in event_ids]
            for ids_ in event_ids:  # check if tagging is attempted
                if any([id_ not in ids for id_ in ids_]):
                    tagging = True
            # 1. treat everything that's not in event_id as a tag
            # 2a. for tags, find all the event_ids matched by the tags
            # 2b. for non-tag ids, just pass them directly
            # 3. do this for every input
            event_ids = [[k for k in ids if all((tag in k.split("/")
                         for tag in id_))]  # find ids matching all tags
                         if all(id__ not in ids for id__ in id_)
                         else id_  # straight pass for non-tag inputs
                         for id_ in event_ids]
            for ii, id_ in enumerate(event_ids):
                if len(id_) == 0:
                    raise KeyError(orig_ids[ii] + "not found in the "
                                   "epoch object's event_id.")
                elif len(set([sub_id in ids for sub_id in id_])) != 1:
                    err = ("Don't mix hierarchical and regular event_ids"
                           " like in \'%s\'." % ", ".join(id_))
                    raise ValueError(err)

            # raise for non-orthogonal tags
            if tagging is True:
                events_ = [set(self[x].events[:, 0]) for x in event_ids]
                doubles = events_[0].intersection(events_[1])
                if len(doubles):
                    raise ValueError("The two sets of epochs are "
                                     "overlapping. Provide an "
                                     "orthogonal selection.")

        for eq in event_ids:
            eq_inds.append(np.where(self._keys_to_idx(eq))[0])

        event_times = [self.events[e, 0] for e in eq_inds]
        indices = _get_drop_indices(event_times, method)
        # need to re-index indices
        indices = np.concatenate([e[idx] for e, idx in zip(eq_inds, indices)])
        self.drop(indices, reason='EQUALIZED_COUNT')
        # actually remove the indices
        return self, indices


def _hid_match(event_id, keys):
    """Match event IDs using HID selection.

    Parameters
    ----------
    event_id : dict
        The event ID dictionary.
    keys : list | str
        The event ID or subset (for HID), or list of such items.

    Returns
    -------
    use_keys : list
        The full keys that fit the selection criteria.
    """
    # form the hierarchical event ID mapping
    keys = [keys] if not isinstance(keys, (list, tuple)) else keys
    use_keys = []
    for key in keys:
        if not isinstance(key, string_types):
            raise KeyError('keys must be strings, got %s (%s)'
                           % (type(key), key))
        use_keys.extend(k for k in event_id.keys()
                        if set(key.split('/')).issubset(k.split('/')))
    if len(use_keys) == 0:
        raise KeyError('Event "%s" is not in Epochs.' % key)
    use_keys = list(set(use_keys))  # deduplicate if necessary
    return use_keys


def _check_baseline(baseline, tmin, tmax, sfreq):
    """Check for a valid baseline."""
    if baseline is not None:
        if not isinstance(baseline, tuple) or len(baseline) != 2:
            raise ValueError('`baseline=%s` is an invalid argument.'
                             % str(baseline))
        baseline_tmin, baseline_tmax = baseline
        tstep = 1. / float(sfreq)
        if baseline_tmin is None:
            baseline_tmin = tmin
        baseline_tmin = float(baseline_tmin)
        if baseline_tmax is None:
            baseline_tmax = tmax
        baseline_tmax = float(baseline_tmax)
        if baseline_tmin < tmin - tstep:
            raise ValueError(
                "Baseline interval (tmin = %s) is outside of epoch "
                "data (tmin = %s)" % (baseline_tmin, tmin))
        if baseline_tmax > tmax + tstep:
            raise ValueError(
                "Baseline interval (tmax = %s) is outside of epoch "
                "data (tmax = %s)" % (baseline_tmax, tmax))
        if baseline_tmin > baseline_tmax:
            raise ValueError(
                "Baseline min (%s) must be less than baseline max (%s)"
                % (baseline_tmin, baseline_tmax))
        del baseline_tmin, baseline_tmax


def _drop_log_stats(drop_log, ignore=('IGNORED',)):
    """Compute drop log stats.

    Parameters
    ----------
    drop_log : list of lists
        Epoch drop log from Epochs.drop_log.
    ignore : list
        The drop reasons to ignore.

    Returns
    -------
    perc : float
        Total percentage of epochs dropped.
    """
    if not isinstance(drop_log, list) or not isinstance(drop_log[0], list):
        raise ValueError('drop_log must be a list of lists')
    perc = 100 * np.mean([len(d) > 0 for d in drop_log
                          if not any(r in ignore for r in d)])
    return perc


def _dep_eeg_ref(add_eeg_ref):
    """Helper for deprecation add_eeg_ref -> False."""
    # current_default is False
    add_eeg_ref = bool(add_eeg_ref)
    if add_eeg_ref:
        warn('add_eeg_ref will be removed in 0.15, use set_eeg_reference()'
             ' instead', DeprecationWarning)
    return add_eeg_ref


class Epochs(BaseEpochs):
    """Epochs extracted from a Raw instance.

    Parameters
    ----------
    raw : Raw object
        An instance of Raw.
    events : array of int, shape (n_events, 3)
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
        Start time before event. If nothing is provided, defaults to -0.2
    tmax : float
        End time after event. If nothing is provided, defaults to 0.5
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
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
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
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
        already exists). The default value of True in 0.13 will change to
        False in 0.14, and the parameter will be removed in 0.15. Use
        :func:`mne.set_eeg_reference` instead.
    on_missing : str
        What to do if one or several event ids are not found in the recording.
        Valid keys are 'error' | 'warning' | 'ignore'
        Default is 'error'. If on_missing is 'warning' it will proceed but
        warn, if 'ignore' it will proceed silently. Note.
        If none of the event ids are found in the data, an error will be
        automatically generated irrespective of this parameter.
    reject_by_annotation : bool
        Whether to reject based on annotations. If True (default), epochs
        overlapping with segments whose description begins with ``'bad'`` are
        rejected. If False, no rejection based on annotations is performed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        raw.verbose.

    Attributes
    ----------
    info : instance of Info
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
        or 'USER' for user-defined reasons (see drop method).
    filename : str
        The filename of the object.
    verbose : bool, str, int, or None
        See above.

    See Also
    --------
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts

    Notes
    -----
    When accessing data, Epochs are detrended, baseline-corrected, and
    decimated, then projectors are (optionally) applied.

    For indexing and slicing using ``epochs[...]``, see
    :func:`mne.Epochs.__getitem__`.
    """

    @verbose
    def __init__(self, raw, events, event_id=None, tmin=-0.2, tmax=0.5,
                 baseline=(None, 0), picks=None, name='Unknown', preload=False,
                 reject=None, flat=None, proj=True, decim=1, reject_tmin=None,
                 reject_tmax=None, detrend=None, add_eeg_ref=None,
                 on_missing='error', reject_by_annotation=True,
                 verbose=None):  # noqa: D102
        if not isinstance(raw, BaseRaw):
            raise ValueError('The first argument to `Epochs` must be an '
                             'instance of `mne.io.Raw`')
        info = deepcopy(raw.info)

        # proj is on when applied in Raw
        proj = proj or raw.proj

        self.reject_by_annotation = reject_by_annotation
        # call BaseEpochs constructor
        super(Epochs, self).__init__(
            info, None, events, event_id, tmin, tmax, baseline=baseline,
            raw=raw, picks=picks, name=name, reject=reject, flat=flat,
            decim=decim, reject_tmin=reject_tmin, reject_tmax=reject_tmax,
            detrend=detrend, add_eeg_ref=add_eeg_ref, proj=proj,
            on_missing=on_missing, preload_at_end=preload, verbose=verbose)

    @verbose
    def _get_epoch_from_raw(self, idx, verbose=None):
        """Load one epoch from disk.

        Returns
        -------
        data : array | str | None
            If string it's details on rejection reason.
            If None it means no data.
        """
        if self._raw is None:
            # This should never happen, as raw=None only if preload=True
            raise ValueError('An error has occurred, no valid raw file found.'
                             ' Please report this to the mne-python '
                             'developers.')
        sfreq = self._raw.info['sfreq']
        event_samp = self.events[idx, 0]
        # Read a data segment
        first_samp = self._raw.first_samp
        start = int(round(event_samp + self._raw_times[0] * sfreq))
        start -= first_samp
        stop = start + len(self._raw_times)
        data = self._raw._check_bad_segment(start, stop, self.picks,
                                            self.reject_by_annotation)
        return data


class EpochsArray(BaseEpochs):
    """Epochs object from numpy array.

    Parameters
    ----------
    data : array, shape (n_epochs, n_channels, n_times)
        The channels' time series for each epoch. See notes for proper units of
        measure.
    info : instance of Info
        Info dictionary. Consider using ``create_info`` to populate
        this structure.
    events : None | array of int, shape (n_events, 3)
        The events typically returned by the read_events function.
        If some events don't match the events of interest as specified
        by event_id, they will be marked as 'IGNORED' in the drop log.
        If None (default), all event values are set to 1 and event time-samples
        are set to range(n_epochs).
    tmin : float
        Start time before event. If nothing provided, defaults to -0.2.
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
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
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
    baseline : None or tuple of length 2 (default None)
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    proj : bool | 'delayed'
        Apply SSP projection vectors. See :class:`mne.Epochs` for details.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Notes
    -----
    Proper units of measure:
    * V: eeg, eog, seeg, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc

    See Also
    --------
    io.RawArray, EvokedArray, create_info
    """

    @verbose
    def __init__(self, data, info, events=None, tmin=0, event_id=None,
                 reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, baseline=None, proj=True,
                 verbose=None):  # noqa: D102
        dtype = np.complex128 if np.any(np.iscomplex(data)) else np.float64
        data = np.asanyarray(data, dtype=dtype)
        if data.ndim != 3:
            raise ValueError('Data must be a 3D array of shape (n_epochs, '
                             'n_channels, n_samples)')

        if len(info['ch_names']) != data.shape[1]:
            raise ValueError('Info and data must have same number of '
                             'channels.')
        if events is None:
            n_epochs = len(data)
            events = np.c_[np.arange(n_epochs), np.zeros(n_epochs, int),
                           np.ones(n_epochs, int)]
        if data.shape[0] != len(events):
            raise ValueError('The number of epochs and the number of events'
                             'must match')
        info = info.copy()  # do not modify original info
        tmax = (data.shape[2] - 1) / info['sfreq'] + tmin
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = dict((str(e), int(e)) for e in np.unique(events[:, 2]))
        super(EpochsArray, self).__init__(info, data, events, event_id, tmin,
                                          tmax, baseline, reject=reject,
                                          flat=flat, reject_tmin=reject_tmin,
                                          reject_tmax=reject_tmax, decim=1,
                                          proj=proj)
        if len(events) != np.in1d(self.events[:, 2],
                                  list(self.event_id.values())).sum():
            raise ValueError('The events must only contain event numbers from '
                             'event_id')
        for ii, e in enumerate(self._data):
            # This is safe without assignment b/c there is no decim
            self._detrend_offset_decim(e)
        self.drop_bad()


def combine_event_ids(epochs, old_event_ids, new_event_id, copy=True):
    """Collapse event_ids from an epochs instance into a new event_id.

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
        Whether to return a new instance or modify in place.

    Notes
    -----
    This For example (if epochs.event_id was {'Left': 1, 'Right': 2}:

        combine_event_ids(epochs, ['Left', 'Right'], {'Directional': 12})

    would create a 'Directional' entry in epochs.event_id replacing
    'Left' and 'Right' (combining their trials).
    """
    epochs = epochs.copy() if copy else epochs
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
    for key in old_event_ids:
        epochs.event_id.pop(key)
    # add the new entry
    epochs.event_id.update(new_event_id)
    return epochs


def equalize_epoch_counts(epochs_list, method='mintime'):
    """Equalize the number of trials in multiple Epoch instances.

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
    if not all(isinstance(e, BaseEpochs) for e in epochs_list):
        raise ValueError('All inputs must be Epochs instances')

    # make sure bad epochs are dropped
    for e in epochs_list:
        if not e._bad_dropped:
            e.drop_bad()
    event_times = [e.events[:, 0] for e in epochs_list]
    indices = _get_drop_indices(event_times, method)
    for e, inds in zip(epochs_list, indices):
        e.drop(inds, reason='EQUALIZED_COUNT')


def _get_drop_indices(event_times, method):
    """Get indices to drop from multiple event timing lists."""
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


def _fix_fill(fill):
    """Helper to fix bug on old scipy."""
    if LooseVersion(scipy.__version__) < LooseVersion('0.12'):
        fill = fill[:, np.newaxis]
    return fill


def _minimize_time_diff(t_shorter, t_longer):
    """Find a boolean mask to minimize timing differences."""
    from scipy.interpolate import interp1d
    keep = np.ones((len(t_longer)), dtype=bool)
    if len(t_shorter) == 0:
        keep.fill(False)
        return keep
    scores = np.ones((len(t_longer)))
    x1 = np.arange(len(t_shorter))
    # The first set of keep masks to test
    kwargs = dict(copy=False, bounds_error=False)
    # this is a speed tweak, only exists for certain versions of scipy
    if 'assume_sorted' in _get_args(interp1d.__init__):
        kwargs['assume_sorted'] = True
    shorter_interp = interp1d(x1, t_shorter, fill_value=t_shorter[-1],
                              **kwargs)
    for ii in range(len(t_longer) - len(t_shorter)):
        scores.fill(np.inf)
        # set up the keep masks to test, eliminating any rows that are already
        # gone
        keep_mask = ~np.eye(len(t_longer), dtype=bool)[keep]
        keep_mask[:, ~keep] = False
        # Check every possible removal to see if it minimizes
        x2 = np.arange(len(t_longer) - ii - 1)
        t_keeps = np.array([t_longer[km] for km in keep_mask])
        longer_interp = interp1d(x2, t_keeps, axis=1,
                                 fill_value=_fix_fill(t_keeps[:, -1]),
                                 **kwargs)
        d1 = longer_interp(x1) - t_shorter
        d2 = shorter_interp(x2) - t_keeps
        scores[keep] = np.abs(d1, d1).sum(axis=1) + np.abs(d2, d2).sum(axis=1)
        keep[np.argmin(scores)] = False
    return keep


@verbose
def _is_good(e, ch_names, channel_type_idx, reject, flat, full_report=False,
             ignore_chs=[], verbose=None):
    """Test if data segment e is good according to reject and flat.

    If full_report=True, it will give True/False as well as a list of all
    offending channels.
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


def _read_one_epoch_file(f, tree, preload):
    """Read a single FIF file."""
    with f as fid:
        #   Read the measurement info
        info, meas = read_meas_info(fid, tree, clean_bads=True)

        events, mappings = _read_events_fif(fid, tree)

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        if len(processed) == 0:
            raise ValueError('Could not find processed data')

        epochs_node = dir_tree_find(tree, FIFF.FIFFB_MNE_EPOCHS)
        if len(epochs_node) == 0:
            # before version 0.11 we errantly saved with this tag instead of
            # an MNE tag
            epochs_node = dir_tree_find(tree, FIFF.FIFFB_MNE_EPOCHS)
            if len(epochs_node) == 0:
                epochs_node = dir_tree_find(tree, 122)  # 122 used before v0.11
                if len(epochs_node) == 0:
                    raise ValueError('Could not find epochs data')

        my_epochs = epochs_node[0]

        # Now find the data in the block
        name = None
        data = None
        data_tag = None
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
                # delay reading until later
                fid.seek(pos, 0)
                data_tag = read_tag_info(fid)
                data_tag.pos = pos
            elif kind in [FIFF.FIFF_MNE_BASELINE_MIN, 304]:
                # Constant 304 was used before v0.11
                tag = read_tag(fid, pos)
                bmin = float(tag.data)
            elif kind in [FIFF.FIFF_MNE_BASELINE_MAX, 305]:
                # Constant 305 was used before v0.11
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

        n_samp = last - first + 1
        logger.info('    Found the data of interest:')
        logger.info('        t = %10.2f ... %10.2f ms (%s)'
                    % (1000 * first / info['sfreq'],
                       1000 * last / info['sfreq'], name))
        if info['comps'] is not None:
            logger.info('        %d CTF compensation matrices available'
                        % len(info['comps']))

        # Inspect the data
        if data_tag is None:
            raise ValueError('Epochs data not found')
        epoch_shape = (len(info['ch_names']), n_samp)
        expected = len(events) * np.prod(epoch_shape)
        if data_tag.size // 4 - 4 != expected:  # 32-bit floats stored
            raise ValueError('Incorrect number of samples (%d instead of %d)'
                             % (data_tag.size // 4, expected))

        # Calibration factors
        cals = np.array([[info['chs'][k]['cal'] *
                          info['chs'][k].get('scale', 1.0)]
                         for k in range(info['nchan'])], np.float64)

        # Read the data
        if preload:
            data = read_tag(fid, data_tag.pos).data.astype(np.float64)
            data *= cals[np.newaxis, :, :]

        # Put it all together
        tmin = first / info['sfreq']
        tmax = last / info['sfreq']
        event_id = (dict((str(e), e) for e in np.unique(events[:, 2]))
                    if mappings is None else mappings)
        # In case epochs didn't have a FIFF.FIFFB_MNE_EPOCHS_SELECTION tag
        # (version < 0.8):
        if selection is None:
            selection = np.arange(len(events))
        if drop_log is None:
            drop_log = [[] for _ in range(len(events))]

    return (info, data, data_tag, events, event_id, tmin, tmax, baseline, name,
            selection, drop_log, epoch_shape, cals)


@verbose
def read_epochs(fname, proj=True, preload=True, verbose=None):
    """Read epochs from a fif file.

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
    preload : bool
        If True, read all epochs from disk immediately. If False, epochs will
        be read on demand.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    epochs : instance of Epochs
        The epochs
    """
    return EpochsFIF(fname, proj, False, preload, verbose)


class _RawContainer(object):
    """Helper for a raw data container."""

    def __init__(self, fid, data_tag, event_samps, epoch_shape,
                 cals):  # noqa: D102
        self.fid = fid
        self.data_tag = data_tag
        self.event_samps = event_samps
        self.epoch_shape = epoch_shape
        self.cals = cals
        self.proj = False

    def __del__(self):  # noqa: D105
        self.fid.close()


class EpochsFIF(BaseEpochs):
    """Epochs read from disk.

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
        already exists). This parameter will be removed in 0.15. Use
        :func:`mne.set_eeg_reference` instead.
    preload : bool
        If True, read all epochs from disk immediately. If False, epochs will
        be read on demand.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        raw.verbose.

    See Also
    --------
    mne.Epochs
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts
    """

    @verbose
    def __init__(self, fname, proj=True, add_eeg_ref=False, preload=True,
                 verbose=None):  # noqa: D102
        check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz'))
        fnames = [fname]
        ep_list = list()
        raw = list()
        for fname in fnames:
            logger.info('Reading %s ...' % fname)
            fid, tree, _ = fiff_open(fname)
            next_fname = _get_next_fname(fid, fname, tree)
            (info, data, data_tag, events, event_id, tmin, tmax, baseline,
             name, selection, drop_log, epoch_shape, cals) = \
                _read_one_epoch_file(fid, tree, preload)
            # here we ignore missing events, since users should already be
            # aware of missing events if they have saved data that way
            epoch = BaseEpochs(
                info, data, events, event_id, tmin, tmax, baseline,
                on_missing='ignore', selection=selection, drop_log=drop_log,
                proj=False, verbose=False)
            ep_list.append(epoch)
            if not preload:
                # store everything we need to index back to the original data
                raw.append(_RawContainer(fiff_open(fname)[0], data_tag,
                                         events[:, 0].copy(), epoch_shape,
                                         cals))

            if next_fname is not None:
                fnames.append(next_fname)

        (info, data, events, event_id, tmin, tmax, baseline, selection,
         drop_log, _) = _concatenate_epochs(ep_list, with_data=preload)
        # we need this uniqueness for non-preloaded data to work properly
        if len(np.unique(events[:, 0])) != len(events):
            raise RuntimeError('Event time samples were not unique')

        # correct the drop log
        assert len(drop_log) % len(fnames) == 0
        step = len(drop_log) // len(fnames)
        offsets = np.arange(step, len(drop_log) + 1, step)
        for i1, i2 in zip(offsets[:-1], offsets[1:]):
            other_log = drop_log[i1:i2]
            for k, (a, b) in enumerate(zip(drop_log, other_log)):
                    if a == ['IGNORED'] and b != ['IGNORED']:
                        drop_log[k] = b
        drop_log = drop_log[:step]

        # call BaseEpochs constructor
        super(EpochsFIF, self).__init__(
            info, data, events, event_id, tmin, tmax, baseline, raw=raw,
            name=name, proj=proj, add_eeg_ref=add_eeg_ref,
            preload_at_end=False, on_missing='ignore', selection=selection,
            drop_log=drop_log, filename=fname, verbose=verbose)
        # use the private property instead of drop_bad so that epochs
        # are not all read from disk for preload=False
        self._bad_dropped = True

    @verbose
    def _get_epoch_from_raw(self, idx, verbose=None):
        """Load one epoch from disk."""
        # Find the right file and offset to use
        event_samp = self.events[idx, 0]
        for raw in self._raw:
            idx = np.where(raw.event_samps == event_samp)[0]
            if len(idx) == 1:
                idx = idx[0]
                size = np.prod(raw.epoch_shape) * 4
                offset = idx * size
                break
        else:
            # read the correct subset of the data
            raise RuntimeError('Correct epoch could not be found, please '
                               'contact mne-python developers')
        # the following is equivalent to this, but faster:
        #
        # >>> data = read_tag(raw.fid, raw.data_tag.pos).data.astype(float)
        # >>> data *= raw.cals[np.newaxis, :, :]
        # >>> data = data[idx]
        #
        # Eventually this could be refactored in io/tag.py if other functions
        # could make use of it

        raw.fid.seek(raw.data_tag.pos + offset + 16, 0)  # 16 = Tag header
        data = np.fromstring(raw.fid.read(size), '>f4').astype(np.float64)
        data.shape = raw.epoch_shape
        data *= raw.cals
        return data


def bootstrap(epochs, random_state=None):
    """Compute epochs selected by bootstrapping.

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
    """Aux function."""
    if len(set(tuple(epochs.event_id.items()) for epochs in epochs_list)) != 1:
        raise NotImplementedError("Epochs with unequal values for event_id")
    if len(set(epochs.tmin for epochs in epochs_list)) != 1:
        raise NotImplementedError("Epochs with unequal values for tmin")
    if len(set(epochs.tmax for epochs in epochs_list)) != 1:
        raise NotImplementedError("Epochs with unequal values for tmax")
    if len(set(epochs.baseline for epochs in epochs_list)) != 1:
        raise NotImplementedError("Epochs with unequal values for baseline")


@verbose
def add_channels_epochs(epochs_list, name='Unknown', add_eeg_ref=False,
                        verbose=None):
    """Concatenate channels, info and data from two Epochs objects.

    Parameters
    ----------
    epochs_list : list of Epochs
        Epochs object to concatenate.
    name : str
        Comment that describes the Epochs data created.
    add_eeg_ref : bool
        If True, an EEG average reference will be added (unless there is
        no EEG in the data). This parameter will be removed in 0.15. Use
        :func:`mne.set_eeg_reference` instead.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more). Defaults to
        True if any of the input epochs have verbose=True.

    Returns
    -------
    epochs : instance of Epochs
        Concatenated epochs.
    """
    add_eeg_ref = _dep_eeg_ref(add_eeg_ref)
    if not all(e.preload for e in epochs_list):
        raise ValueError('All epochs must be preloaded.')

    info = _merge_info([epochs.info for epochs in epochs_list])
    data = [epochs.get_data() for epochs in epochs_list]
    _check_merge_epochs(epochs_list)
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
    """Compare infos."""
    info1._check_consistency()
    info2._check_consistency()
    if info1['nchan'] != info2['nchan']:
        raise ValueError('epochs[%d][\'info\'][\'nchan\'] must match' % ind)
    if info1['bads'] != info2['bads']:
        raise ValueError('epochs[%d][\'info\'][\'bads\'] must match' % ind)
    if info1['sfreq'] != info2['sfreq']:
        raise ValueError('epochs[%d][\'info\'][\'sfreq\'] must match' % ind)
    if set(info1['ch_names']) != set(info2['ch_names']):
        raise ValueError('epochs[%d][\'info\'][\'ch_names\'] must match' % ind)
    if len(info2['projs']) != len(info1['projs']):
        raise ValueError('SSP projectors in epochs files must be the same')
    if any(not _proj_equal(p1, p2) for p1, p2 in
           zip(info2['projs'], info1['projs'])):
        raise ValueError('SSP projectors in epochs files must be the same')
    if (info1['dev_head_t'] is None) != (info2['dev_head_t'] is None) or \
            (info1['dev_head_t'] is not None and not
             np.allclose(info1['dev_head_t']['trans'],
                         info2['dev_head_t']['trans'], rtol=1e-6)):
        raise ValueError('epochs[%d][\'info\'][\'dev_head_t\'] must match. '
                         'The epochs probably come from different runs, and '
                         'are therefore associated with different head '
                         'positions. Manually change info[\'dev_head_t\'] to '
                         'avoid this message but beware that this means the '
                         'MEG sensors will not be properly spatially aligned. '
                         'See mne.preprocessing.maxwell_filter to realign the '
                         'runs to a common head position.' % ind)


def _concatenate_epochs(epochs_list, with_data=True):
    """Auxiliary function for concatenating epochs."""
    if not isinstance(epochs_list, (list, tuple)):
        raise TypeError('epochs_list must be a list or tuple, got %s'
                        % (type(epochs_list),))
    for ei, epochs in enumerate(epochs_list):
        if not isinstance(epochs, BaseEpochs):
            raise TypeError('epochs_list[%d] must be an instance of Epochs, '
                            'got %s' % (ei, type(epochs)))
    out = epochs_list[0]
    data = [out.get_data()] if with_data else None
    events = [out.events]
    baseline, tmin, tmax = out.baseline, out.tmin, out.tmax
    info = deepcopy(out.info)
    verbose = out.verbose
    drop_log = deepcopy(out.drop_log)
    event_id = deepcopy(out.event_id)
    selection = out.selection
    for ii, epochs in enumerate(epochs_list[1:]):
        _compare_epochs_infos(epochs.info, info, ii)
        if not np.allclose(epochs.times, epochs_list[0].times):
            raise ValueError('Epochs must have same times')

        if epochs.baseline != baseline:
            raise ValueError('Baseline must be same for all epochs')

        # compare event_id
        common_keys = list(set(event_id).intersection(set(epochs.event_id)))
        for key in common_keys:
            if not event_id[key] == epochs.event_id[key]:
                msg = ('event_id values must be the same for identical keys '
                       'for all concatenated epochs. Key "{}" maps to {} in '
                       'some epochs and to {} in others.')
                raise ValueError(msg.format(key, event_id[key],
                                            epochs.event_id[key]))

        if with_data:
            data.append(epochs.get_data())
        events.append(epochs.events)
        selection = np.concatenate((selection, epochs.selection))
        drop_log.extend(epochs.drop_log)
        event_id.update(epochs.event_id)
    events = np.concatenate(events, axis=0)
    if with_data:
        data = np.concatenate(data, axis=0)
    return (info, data, events, event_id, tmin, tmax, baseline, selection,
            drop_log, verbose)


def _finish_concat(info, data, events, event_id, tmin, tmax, baseline,
                   selection, drop_log, verbose):
    """Helper to finish concatenation for epochs not read from disk."""
    events[:, 0] = np.arange(len(events))  # arbitrary after concat
    selection = np.where([len(d) == 0 for d in drop_log])[0]
    out = BaseEpochs(
        info, data, events, event_id, tmin, tmax, baseline=baseline,
        selection=selection, drop_log=drop_log, proj=False,
        on_missing='ignore', verbose=verbose)
    out.drop_bad()
    return out


def concatenate_epochs(epochs_list):
    """Concatenate a list of epochs into one epochs object.

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
    return _finish_concat(*_concatenate_epochs(epochs_list))


@verbose
def average_movements(epochs, head_pos=None, orig_sfreq=None, picks=None,
                      origin='auto', weight_all=True, int_order=8, ext_order=3,
                      destination=None, ignore_ref=False, return_mapping=False,
                      mag_scale=100., verbose=None):
    u"""Average data using Maxwell filtering, transforming using head positions.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs to operate on.
    head_pos : array | tuple | None
        The array should be of shape ``(N, 10)``, holding the position
        parameters as returned by e.g. `read_head_pos`. For backward
        compatibility, this can also be a tuple of ``(trans, rot t)``
        as returned by `head_pos_to_trans_rot_t`.
    orig_sfreq : float | None
        The original sample frequency of the data (that matches the
        event sample numbers in ``epochs.events``). Can be ``None``
        if data have not been decimated or resampled.
    picks : array-like of int | None
        If None only MEG, EEG, SEEG, ECoG, and fNIRS channels are kept
        otherwise the channels indices in picks are kept.
    origin : array-like, shape (3,) | str
        Origin of internal and external multipolar moment space in head
        coords and in meters. The default is ``'auto'``, which means
        a head-digitization-based origin fit.
    weight_all : bool
        If True, all channels are weighted by the SSS basis weights.
        If False, only MEG channels are weighted, other channels
        receive uniform weight per epoch.
    int_order : int
        Order of internal component of spherical expansion.
    ext_order : int
        Order of external component of spherical expansion.
    regularize : str | None
        Basis regularization type, must be "in" or None.
        See :func:`mne.preprocessing.maxwell_filter` for details.
        Regularization is chosen based only on the destination position.
    destination : str | array-like, shape (3,) | None
        The destination location for the head. Can be ``None``, which
        will not change the head position, or a string path to a FIF file
        containing a MEG device<->head transformation, or a 3-element array
        giving the coordinates to translate to (with no rotations).
        For example, ``destination=(0, 0, 0.04)`` would translate the bases
        as ``--trans default`` would in MaxFilter™ (i.e., to the default
        head location).

        .. versionadded:: 0.12

    ignore_ref : bool
        If True, do not include reference channels in compensation. This
        option should be True for KIT files, since Maxwell filtering
        with reference channels is not currently supported.
    return_mapping : bool
        If True, return the mapping matrix.
    mag_scale : float | str
        The magenetometer scale-factor used to bring the magnetometers
        to approximately the same order of magnitude as the gradiometers
        (default 100.), as they have different units (T vs T/m).
        Can be ``'auto'`` to use the reciprocal of the physical distance
        between the gradiometer pickup loops (e.g., 0.0168 m yields
        59.5 for VectorView).

        .. versionadded:: 0.13

    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    evoked : instance of Evoked
        The averaged epochs.

    See Also
    --------
    mne.preprocessing.maxwell_filter
    mne.chpi.read_head_pos

    Notes
    -----
    The Maxwell filtering version of this algorithm is described in [1]_,
    in section V.B "Virtual signals and movement correction", equations
    40-44. For additional validation, see [2]_.

    Regularization has not been added because in testing it appears to
    decrease dipole localization accuracy relative to using all components.
    Fine calibration and cross-talk cancellation, however, could be added
    to this algorithm based on user demand.

    .. versionadded:: 0.11

    References
    ----------
    .. [1] Taulu S. and Kajola M. "Presentation of electromagnetic
           multichannel data: The signal space separation method,"
           Journal of Applied Physics, vol. 97, pp. 124905 1-10, 2005.

    .. [2] Wehner DT, Hämäläinen MS, Mody M, Ahlfors SP. "Head movements
           of children in MEG: Quantification, effects on source
           estimation, and compensation. NeuroImage 40:541–550, 2008.
    """  # noqa: E501
    from .preprocessing.maxwell import (_trans_sss_basis, _reset_meg_bads,
                                        _check_usable, _col_norm_pinv,
                                        _get_n_moments, _get_mf_picks,
                                        _prep_mf_coils, _check_destination,
                                        _remove_meg_projs, _get_coil_scale)
    if head_pos is None:
        raise TypeError('head_pos must be provided and cannot be None')
    from .chpi import head_pos_to_trans_rot_t
    if not isinstance(epochs, BaseEpochs):
        raise TypeError('epochs must be an instance of Epochs, not %s'
                        % (type(epochs),))
    orig_sfreq = epochs.info['sfreq'] if orig_sfreq is None else orig_sfreq
    orig_sfreq = float(orig_sfreq)
    if isinstance(head_pos, np.ndarray):
        head_pos = head_pos_to_trans_rot_t(head_pos)
    trn, rot, t = head_pos
    del head_pos
    _check_usable(epochs)
    origin = _check_origin(origin, epochs.info, 'head')
    recon_trans = _check_destination(destination, epochs.info, True)

    logger.info('Aligning and averaging up to %s epochs'
                % (len(epochs.events)))
    if not np.array_equal(epochs.events[:, 0], np.unique(epochs.events[:, 0])):
        raise RuntimeError('Epochs must have monotonically increasing events')
    meg_picks, mag_picks, grad_picks, good_picks, _ = \
        _get_mf_picks(epochs.info, int_order, ext_order, ignore_ref)
    coil_scale, mag_scale = _get_coil_scale(
        meg_picks, mag_picks, grad_picks, mag_scale, epochs.info)
    n_channels, n_times = len(epochs.ch_names), len(epochs.times)
    other_picks = np.setdiff1d(np.arange(n_channels), meg_picks)
    data = np.zeros((n_channels, n_times))
    count = 0
    # keep only MEG w/bad channels marked in "info_from"
    info_from = pick_info(epochs.info, good_picks, copy=True)
    all_coils_recon = _prep_mf_coils(epochs.info, ignore_ref=ignore_ref)
    all_coils = _prep_mf_coils(info_from, ignore_ref=ignore_ref)
    # remove MEG bads in "to" info
    info_to = deepcopy(epochs.info)
    _reset_meg_bads(info_to)
    # set up variables
    w_sum = 0.
    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_decomp = 0.  # this will end up being a weighted average
    last_trans = None
    decomp_coil_scale = coil_scale[good_picks]
    exp = dict(int_order=int_order, ext_order=ext_order, head_frame=True,
               origin=origin)
    for ei, epoch in enumerate(epochs):
        event_time = epochs.events[epochs._current - 1, 0] / orig_sfreq
        use_idx = np.where(t <= event_time)[0]
        if len(use_idx) == 0:
            trans = epochs.info['dev_head_t']['trans']
        else:
            use_idx = use_idx[-1]
            trans = np.vstack([np.hstack([rot[use_idx], trn[[use_idx]].T]),
                               [[0., 0., 0., 1.]]])
        loc_str = ', '.join('%0.1f' % tr for tr in (trans[:3, 3] * 1000))
        if last_trans is None or not np.allclose(last_trans, trans):
            logger.info('    Processing epoch %s (device location: %s mm)'
                        % (ei + 1, loc_str))
            reuse = False
            last_trans = trans
        else:
            logger.info('    Processing epoch %s (device location: same)'
                        % (ei + 1,))
            reuse = True
        epoch = epoch.copy()  # because we operate inplace
        if not reuse:
            S = _trans_sss_basis(exp, all_coils, trans,
                                 coil_scale=decomp_coil_scale)
            # Get the weight from the un-regularized version
            weight = np.sqrt(np.sum(S * S))  # frobenius norm (eq. 44)
            # XXX Eventually we could do cross-talk and fine-cal here
            S *= weight
        S_decomp += S  # eq. 41
        epoch[slice(None) if weight_all else meg_picks] *= weight
        data += epoch  # eq. 42
        w_sum += weight
        count += 1
    del info_from
    mapping = None
    if count == 0:
        data.fill(np.nan)
    else:
        data[meg_picks] /= w_sum
        data[other_picks] /= w_sum if weight_all else count
        # Finalize weighted average decomp matrix
        S_decomp /= w_sum
        # Get recon matrix
        # (We would need to include external here for regularization to work)
        exp['ext_order'] = 0
        S_recon = _trans_sss_basis(exp, all_coils_recon, recon_trans)
        exp['ext_order'] = ext_order
        # We could determine regularization on basis of destination basis
        # matrix, restricted to good channels, as regularizing individual
        # matrices within the loop above does not seem to work. But in
        # testing this seemed to decrease localization quality in most cases,
        # so we do not provide the option here.
        S_recon /= coil_scale
        # Invert
        pS_ave = _col_norm_pinv(S_decomp)[0][:n_in]
        pS_ave *= decomp_coil_scale.T
        # Get mapping matrix
        mapping = np.dot(S_recon, pS_ave)
        # Apply mapping
        data[meg_picks] = np.dot(mapping, data[good_picks])
    info_to['dev_head_t'] = recon_trans  # set the reconstruction transform
    evoked = epochs._evoked_from_epoch_data(data, info_to, picks,
                                            n_events=count, kind='average')
    _remove_meg_projs(evoked)  # remove MEG projectors, they won't apply now
    logger.info('Created Evoked dataset from %s epochs' % (count,))
    return (evoked, mapping) if return_mapping else evoked


@verbose
def _segment_raw(raw, segment_length=1., verbose=None, **kwargs):
    """Divide continuous raw data into equal-sized consecutive epochs.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to divide into segments.
    segment_length : float
        Length of each segment in seconds. Defaults to 1.
    verbose: bool
        Whether to report what is being done by printing text.
    **kwargs
        Any additional keyword arguments are passed to ``Epochs`` constructor.

    Returns
    -------
    epochs : instance of ``Epochs``
        Segmented data.
    """
    events = make_fixed_length_events(raw, 1, duration=segment_length)
    return Epochs(raw, events, event_id=[1], tmin=0., tmax=segment_length,
                  verbose=verbose, baseline=None, add_eeg_ref=False, **kwargs)
