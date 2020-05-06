# -*- coding: utf-8 -*-

"""Tools for working with epoched data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#          Denis Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

from collections import Counter
from copy import deepcopy
import json
import operator
import os.path as op
import warnings
from distutils.version import LooseVersion

import numpy as np
import scipy

from .io.write import (start_file, start_block, end_file, end_block,
                       write_int, write_float, write_float_matrix,
                       write_double_matrix, write_complex_float_matrix,
                       write_complex_double_matrix, write_id, write_string,
                       _get_split_size, _NEXT_FILE_BUFFER)
from .io.meas_info import read_meas_info, write_meas_info, _merge_info
from .io.open import fiff_open, _get_next_fname
from .io.tree import dir_tree_find
from .io.tag import read_tag, read_tag_info
from .io.constants import FIFF
from .io.fiff.raw import _get_fname_rep
from .io.pick import (pick_types, channel_indices_by_type, channel_type,
                      pick_channels, pick_info, _pick_data_channels,
                      _pick_aux_channels, _DATA_CH_TYPES_SPLIT,
                      _picks_to_idx)
from .io.proj import setup_proj, ProjMixin, _proj_equal
from .io.base import BaseRaw, TimeMixin
from .bem import _check_origin
from .evoked import EvokedArray, _check_decim
from .baseline import rescale, _log_rescale
from .channels.channels import (ContainsMixin, UpdateChannelsMixin,
                                SetChannelsMixin, InterpolationMixin)
from .filter import detrend, FilterMixin
from .event import _read_events_fif, make_fixed_length_events
from .fixes import _get_args, rng_uniform
from .viz import (plot_epochs, plot_epochs_psd, plot_epochs_psd_topomap,
                  plot_epochs_image, plot_topo_image_epochs, plot_drop_log)
from .utils import (_check_fname, check_fname, logger, verbose,
                    _time_mask, check_random_state, warn, _pl,
                    sizeof_fmt, SizeMixin, copy_function_doc_to_method_doc,
                    _check_pandas_installed, _check_preload, GetEpochsMixin,
                    _prepare_read_metadata, _prepare_write_metadata,
                    _check_event_id, _gen_events, _check_option,
                    _check_combine, ShiftTimeMixin, _build_data_frame,
                    _check_pandas_index_arguments, _convert_times,
                    _scale_dataframe_data, _check_time_format, object_size,
                    _check_scaling_time)
from .utils.docs import fill_doc


def _pack_reject_params(epochs):
    reject_params = dict()
    for key in ('reject', 'flat', 'reject_tmin', 'reject_tmax'):
        val = getattr(epochs, key, None)
        if val is not None:
            reject_params[key] = val
    return reject_params


def _save_split(epochs, fname, part_idx, n_parts, fmt):
    """Split epochs.

    Anything new added to this function also needs to be added to
    BaseEpochs.save to account for new file sizes.
    """
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

    _check_option('fmt', fmt, ['single', 'double'])

    if np.iscomplexobj(data):
        if fmt == 'single':
            write_function = write_complex_float_matrix
        elif fmt == 'double':
            write_function = write_complex_double_matrix
    else:
        if fmt == 'single':
            write_function = write_float_matrix
        elif fmt == 'double':
            write_function = write_double_matrix

    start_block(fid, FIFF.FIFFB_MNE_EVENTS)
    write_int(fid, FIFF.FIFF_MNE_EVENT_LIST, epochs.events.T)
    write_string(fid, FIFF.FIFF_DESCRIPTION, _event_id_string(epochs.event_id))
    end_block(fid, FIFF.FIFFB_MNE_EVENTS)

    # Metadata
    if epochs.metadata is not None:
        start_block(fid, FIFF.FIFFB_MNE_METADATA)
        metadata = _prepare_write_metadata(epochs.metadata)
        write_string(fid, FIFF.FIFF_DESCRIPTION, metadata)
        end_block(fid, FIFF.FIFFB_MNE_METADATA)

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

    write_function(fid, FIFF.FIFF_EPOCH, data)

    # undo modifications to data
    data /= decal[np.newaxis, :, np.newaxis]

    write_string(fid, FIFF.FIFF_MNE_EPOCHS_DROP_LOG,
                 json.dumps(epochs.drop_log))

    reject_params = _pack_reject_params(epochs)
    if reject_params:
        write_string(fid, FIFF.FIFF_MNE_EPOCHS_REJECT_FLAT,
                     json.dumps(reject_params))

    write_int(fid, FIFF.FIFF_MNE_EPOCHS_SELECTION,
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


def _event_id_string(event_id):
    return ';'.join([k + ':' + str(v) for k, v in event_id.items()])


def _merge_events(events, event_id, selection):
    """Merge repeated events."""
    event_id = event_id.copy()
    new_events = events.copy()
    event_idxs_to_delete = list()
    unique_events, counts = np.unique(events[:, 0], return_counts=True)
    for ev in unique_events[counts > 1]:

        # indices at which the non-unique events happened
        idxs = (events[:, 0] == ev).nonzero()[0]

        # Figure out new value for events[:, 1]. Set to 0, if mixed vals exist
        unique_priors = np.unique(events[idxs, 1])
        new_prior = unique_priors[0] if len(unique_priors) == 1 else 0

        # If duplicate time samples have same event val, "merge" == "drop"
        # and no new event_id key will be created
        ev_vals = events[idxs, 2]
        if len(np.unique(ev_vals)) <= 1:
            new_event_val = ev_vals[0]

        # Else, make a new event_id for the merged event
        else:

            # Find all event_id keys involved in duplicated events. These
            # keys will be merged to become a new entry in "event_id"
            event_id_keys = list(event_id.keys())
            event_id_vals = list(event_id.values())
            new_key_comps = [event_id_keys[event_id_vals.index(value)]
                             for value in ev_vals]

            # Check if we already have an entry for merged keys of duplicate
            # events ... if yes, reuse it
            for key in event_id:
                if set(key.split('/')) == set(new_key_comps):
                    new_event_val = event_id[key]
                    break

            # Else, find an unused value for the new key and make an entry into
            # the event_id dict
            else:
                ev_vals = np.concatenate((list(event_id.values()),
                                          events[:, 1:].flatten()),
                                         axis=0)
                new_event_val = np.setdiff1d(np.arange(1, 9999999),
                                             ev_vals).min()
                new_event_id_key = '/'.join(sorted(new_key_comps))
                event_id[new_event_id_key] = int(new_event_val)

        # Replace duplicate event times with merged event and remember which
        # duplicate indices to delete later
        new_events[idxs[0], 1] = new_prior
        new_events[idxs[0], 2] = new_event_val
        event_idxs_to_delete.extend(idxs[1:])

    # Delete duplicate event idxs
    new_events = np.delete(new_events, event_idxs_to_delete, 0)
    new_selection = np.delete(selection, event_idxs_to_delete, 0)

    return new_events, event_id, new_selection


def _handle_event_repeated(events, event_id, event_repeated, selection,
                           drop_log):
    """Handle repeated events.

    Note that drop_log will be modified inplace
    """
    assert len(events) == len(selection)
    selection = np.asarray(selection)

    unique_events, u_ev_idxs = np.unique(events[:, 0], return_index=True)

    # Return early if no duplicates
    if len(unique_events) == len(events):
        return events, event_id, selection, drop_log

    # Else, we have duplicates. Triage ...
    _check_option('event_repeated', event_repeated, ['error', 'drop', 'merge'])
    if event_repeated == 'error':
        raise RuntimeError('Event time samples were not unique. Consider '
                           'setting the `event_repeated` parameter."')

    elif event_repeated == 'drop':
        logger.info('Multiple event values for single event times found. '
                    'Keeping the first occurrence and dropping all others.')
        new_events = events[u_ev_idxs]
        new_selection = selection[u_ev_idxs]
        drop_ev_idxs = np.setdiff1d(selection, new_selection)
        for idx in drop_ev_idxs:
            drop_log[idx].append('DROP DUPLICATE')
        selection = new_selection
    elif event_repeated == 'merge':
        logger.info('Multiple event values for single event times found. '
                    'Creating new event value to reflect simultaneous events.')
        new_events, event_id, new_selection = \
            _merge_events(events, event_id, selection)
        drop_ev_idxs = np.setdiff1d(selection, new_selection)
        for idx in drop_ev_idxs:
            drop_log[idx].append('MERGE DUPLICATE')
        selection = new_selection

    # Remove obsolete kv-pairs from event_id after handling
    keys = new_events[:, 1:].flatten()
    event_id = {k: v for k, v in event_id.items() if v in keys}

    return new_events, event_id, selection, drop_log


@fill_doc
class BaseEpochs(ProjMixin, ContainsMixin, UpdateChannelsMixin, ShiftTimeMixin,
                 SetChannelsMixin, InterpolationMixin, FilterMixin,
                 TimeMixin, SizeMixin, GetEpochsMixin):
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
    %(picks_header)s
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
    metadata : instance of pandas.DataFrame | None
        See :class:`mne.Epochs` docstring.

        .. versionadded:: 0.16
    event_repeated : str
        See :class:`mne.Epochs` docstring.

        .. versionadded:: 0.19
    %(verbose)s

    Notes
    -----
    The ``BaseEpochs`` class is public to allow for stable type-checking in
    user code (i.e., ``isinstance(my_epochs, BaseEpochs)``) but should not be
    used as a constructor for Epochs objects (use instead :class:`mne.Epochs`).
    """

    @verbose
    def __init__(self, info, data, events, event_id=None, tmin=-0.2, tmax=0.5,
                 baseline=(None, 0), raw=None, picks=None, reject=None,
                 flat=None, decim=1, reject_tmin=None, reject_tmax=None,
                 detrend=None, proj=True, on_missing='error',
                 preload_at_end=False, selection=None, drop_log=None,
                 filename=None, metadata=None, event_repeated='error',
                 verbose=None):  # noqa: D102
        self.verbose = verbose

        _check_option('on_missing', on_missing, ['error', 'warning', 'ignore'])

        if events is not None:  # RtEpochs can have events=None
            events_type = type(events)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('ignore')  # deprecation for object array
                events = np.asarray(events)
            if not np.issubdtype(events.dtype, np.integer):
                raise TypeError('events should be a NumPy array of integers, '
                                'got {}'.format(events_type))
        event_id = _check_event_id(event_id, events)
        self.event_id = event_id
        del event_id

        if events is not None:  # RtEpochs can have events=None
            if events.ndim != 2 or events.shape[1] != 3:
                raise ValueError('events must be of shape (N, 3), got %s'
                                 % (events.shape,))
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
            selected = np.where(np.in1d(events[:, 2], values))[0]
            if selection is None:
                selection = selected
            else:
                selection = np.array(selection, int)
            if selection.shape != (len(selected),):
                raise ValueError('selection must be shape %s got shape %s'
                                 % (selected.shape, selection.shape))
            self.selection = selection
            if drop_log is None:
                self.drop_log = [list() if k in self.selection else ['IGNORED']
                                 for k in range(max(len(events),
                                                    max(self.selection) + 1))]
            else:
                self.drop_log = drop_log

            events = events[selected]

            events, self.event_id, self.selection, self.drop_log = \
                _handle_event_repeated(events, self.event_id, event_repeated,
                                       self.selection, self.drop_log)

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
        logger.info(_log_rescale(baseline))
        self.baseline = baseline
        self.reject_tmin = reject_tmin
        self.reject_tmax = reject_tmax
        self.detrend = detrend
        self._raw = raw
        info._check_consistency()
        self.picks = _picks_to_idx(info, picks, none='all', exclude=(),
                                   allow_empty=False)
        self.info = pick_info(info, self.picks)
        del info
        self.metadata = metadata
        self._current = 0

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
        self._set_times(self._raw_times)
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
        activate = False if self._do_delayed_proj else proj
        self._projector, self.info = setup_proj(self.info, False,
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
        self._check_consistency()

    def _check_consistency(self):
        """Check invariants of epochs object."""
        assert len(self.selection) == len(self.events)
        assert len(self.selection) == sum(
            (len(dl) == 0 for dl in self.drop_log))
        assert len(self.drop_log) >= len(self.events)
        assert hasattr(self, '_times_readonly')
        assert not self.times.flags['WRITEABLE']

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
        self._raw = None  # shouldn't need it anymore
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
        %(verbose_meth)s

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
        If `decim` is 1, this method does not copy the underlying data.

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
            if decim != 1:
                self._data = self._data[:, :, decim_slice].copy()
                self._raw_times = self._raw_times[decim_slice].copy()
            else:
                self._data = np.ascontiguousarray(self._data)
            self._decim_slice = slice(None)
            self._decim = 1
        else:
            self._decim_slice = decim_slice
        self._set_times(self._raw_times[self._decim_slice])
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
        %(verbose_meth)s

        Returns
        -------
        epochs : instance of Epochs
            The baseline-corrected Epochs object.

        Notes
        -----
        Baseline correction can be done multiple times.

        .. versionadded:: 0.10.0
        """
        _check_baseline(baseline, self.tmin, self.tmax, self.info['sfreq'])
        if self.preload:
            picks = _pick_data_channels(self.info, exclude=[],
                                        with_ref_meg=True)
            picks_aux = _pick_aux_channels(self.info, exclude=[])
            picks = np.sort(np.concatenate((picks, picks_aux)))
            rescale(self._data, self.times, baseline, copy=False, picks=picks)
        else:  # logging happens in "rescale" in "if" branch
            logger.info(_log_rescale(baseline))
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
        if isinstance(data, str):
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
        if (epoch is None) or isinstance(epoch, str):
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

    def iter_evoked(self, copy=False):
        """Iterate over epochs as a sequence of Evoked objects.

        The Evoked objects yielded will each contain a single epoch (i.e., no
        averaging is performed).

        This method resets the object iteration state to the first epoch.

        Parameters
        ----------
        copy : bool
            If False copies of data and measurement info will be omitted
            to save time.
        """
        self._current = 0

        while True:
            try:
                out = self.__next__(True)
            except StopIteration:
                break
            data, event_id = out
            tmin = self.times[0]
            info = self.info
            if copy:
                info = deepcopy(self.info)
                data = data.copy()

            yield EvokedArray(data, info, tmin, comment=str(event_id))

    def subtract_evoked(self, evoked=None):
        """Subtract an evoked response from each epoch.

        Can be used to exclude the evoked response when analyzing induced
        activity, see e.g. [1]_.

        Parameters
        ----------
        evoked : instance of Evoked | None
            The evoked response to subtract. If None, the evoked response
            is computed from Epochs itself.

        Returns
        -------
        self : instance of Epochs
            The modified instance (instance is also modified inplace).

        References
        ----------
        .. [1] David et al. "Mechanisms of evoked and induced responses in
               MEG/EEG", NeuroImage, vol. 31, no. 4, pp. 1580-1591, July 2006.
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

    @fill_doc
    def average(self, picks=None, method="mean"):
        """Compute an average over epochs.

        Parameters
        ----------
        %(picks_all_data)s
        method : str | callable
            How to combine the data. If "mean"/"median", the mean/median
            are returned.
            Otherwise, must be a callable which, when passed an array of shape
            (n_epochs, n_channels, n_time) returns an array of shape
            (n_channels, n_time).
            Note that due to file type limitations, the kind for all
            these will be "average".

        Returns
        -------
        evoked : instance of Evoked | dict of Evoked
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

        The `method` parameter allows e.g. robust averaging.
        For example, one could do:

            >>> from scipy.stats import trim_mean  # doctest:+SKIP
            >>> trim = lambda x: trim_mean(x, 0.1, axis=0)  # doctest:+SKIP
            >>> epochs.average(method=trim)  # doctest:+SKIP

        This would compute the trimmed mean.
        """
        return self._compute_aggregate(picks=picks, mode=method)

    @fill_doc
    def standard_error(self, picks=None):
        """Compute standard error over epochs.

        Parameters
        ----------
        %(picks_all_data)s

        Returns
        -------
        evoked : instance of Evoked
            The standard error over epochs.
        """
        return self._compute_aggregate(picks, "std")

    def _compute_aggregate(self, picks, mode='mean'):
        """Compute the mean or std over epochs and return Evoked."""
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
            fun = _check_combine(mode, valid=('mean', 'median', 'std'))
            data = fun(self._data)
            assert len(self.events) == len(self._data)
            if data.shape != self._data.shape[1:]:
                raise RuntimeError(
                    'You passed a function that resulted n data of shape {}, '
                    'but it should be {}.'.format(
                        data.shape, self._data.shape[1:]))
        else:
            if mode not in {"mean", "std"}:
                raise ValueError("If data are not preloaded, can only compute "
                                 "mean or standard deviation.")
            data = np.zeros((n_channels, n_times))
            n_events = 0
            for e in self:
                if np.iscomplexobj(e):
                    data = data.astype(np.complex128)
                data += e
                n_events += 1

            if n_events > 0:
                data /= n_events
            else:
                data.fill(np.nan)

            # convert to stderr if requested, could do in one pass but do in
            # two (slower) in case there are large numbers
            if mode == "std":
                data_mean = data.copy()
                data.fill(0.)
                for e in self:
                    data += (e - data_mean) ** 2
                data = np.sqrt(data / n_events)

        if mode == "std":
            kind = 'standard_error'
            data /= np.sqrt(n_events)
        else:
            kind = "average"

        return self._evoked_from_epoch_data(data, self.info, picks, n_events,
                                            kind, self._name)

    @property
    def _name(self):
        """Give a nice string representation based on event ids."""
        if len(self.event_id) == 1:
            comment = next(iter(self.event_id.keys()))
        else:
            count = Counter(self.events[:, 2])
            comments = list()
            for key, value in self.event_id.items():
                comments.append('%.2f * %s' % (
                    float(count[value]) / len(self.events), key))
            comment = ' + '.join(comments)
        return comment

    def _evoked_from_epoch_data(self, data, info, picks, n_events, kind,
                                comment):
        """Create an evoked object from epoch data."""
        info = deepcopy(info)
        evoked = EvokedArray(data, info, tmin=self.times[0], comment=comment,
                             nave=n_events, kind=kind, verbose=self.verbose)
        # XXX: above constructor doesn't recreate the times object precisely
        evoked.times = self.times.copy()

        # pick channels
        picks = _picks_to_idx(self.info, picks, 'data_or_ica', ())
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
             title=None, events=None, event_colors=None, order=None,
             show=True, block=False, decim='auto', noise_cov=None,
             butterfly=False, show_scrollbars=True, epoch_colors=None,
             event_id=None):
        return plot_epochs(self, picks=picks, scalings=scalings,
                           n_epochs=n_epochs, n_channels=n_channels,
                           title=title, events=events,
                           event_colors=event_colors, order=order,
                           show=show, block=block, decim=decim,
                           noise_cov=noise_cov, butterfly=butterfly,
                           show_scrollbars=show_scrollbars,
                           epoch_colors=epoch_colors, event_id=event_id)

    @copy_function_doc_to_method_doc(plot_epochs_psd)
    def plot_psd(self, fmin=0, fmax=np.inf, tmin=None, tmax=None,
                 proj=False, bandwidth=None, adaptive=False, low_bias=True,
                 normalization='length', picks=None, ax=None, color='black',
                 xscale='linear', area_mode='std', area_alpha=0.33,
                 dB=True, estimate='auto', show=True, n_jobs=1,
                 average=False, line_alpha=None, spatial_colors=True,
                 sphere=None, verbose=None):
        return plot_epochs_psd(self, fmin=fmin, fmax=fmax, tmin=tmin,
                               tmax=tmax, proj=proj, bandwidth=bandwidth,
                               adaptive=adaptive, low_bias=low_bias,
                               normalization=normalization, picks=picks, ax=ax,
                               color=color, xscale=xscale, area_mode=area_mode,
                               area_alpha=area_alpha, dB=dB, estimate=estimate,
                               show=show, n_jobs=n_jobs, average=average,
                               line_alpha=line_alpha,
                               spatial_colors=spatial_colors, sphere=sphere,
                               verbose=verbose)

    @copy_function_doc_to_method_doc(plot_epochs_psd_topomap)
    def plot_psd_topomap(self, bands=None, vmin=None, vmax=None, tmin=None,
                         tmax=None, proj=False, bandwidth=None, adaptive=False,
                         low_bias=True, normalization='length', ch_type=None,
                         layout=None, cmap='RdBu_r', agg_fun=None, dB=True,
                         n_jobs=1, normalize=False, cbar_fmt='%0.3f',
                         outlines='head', axes=None, show=True,
                         sphere=None, verbose=None):
        return plot_epochs_psd_topomap(
            self, bands=bands, vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax,
            proj=proj, bandwidth=bandwidth, adaptive=adaptive,
            low_bias=low_bias, normalization=normalization, ch_type=ch_type,
            layout=layout, cmap=cmap, agg_fun=agg_fun, dB=dB, n_jobs=n_jobs,
            normalize=normalize, cbar_fmt=cbar_fmt, outlines=outlines,
            axes=axes, show=show, sphere=sphere, verbose=verbose)

    @copy_function_doc_to_method_doc(plot_topo_image_epochs)
    def plot_topo_image(self, layout=None, sigma=0., vmin=None, vmax=None,
                        colorbar=None, order=None, cmap='RdBu_r',
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
                     times, use :meth:`mne.Epochs.load_data()`.

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
        %(verbose_meth)s

        Returns
        -------
        epochs : instance of Epochs
            The epochs with bad epochs dropped. Operates in-place.

        Notes
        -----
        Dropping bad epochs can be done multiple times with different
        ``reject`` and ``flat`` parameters. However, once an epoch is
        dropped, it is dropped forever, so if more lenient thresholds may
        subsequently be applied, `epochs.copy <mne.Epochs.copy>` should be
        used.
        """
        if reject == 'existing':
            if flat == 'existing' and self._bad_dropped:
                return
            reject = self.reject
        if flat == 'existing':
            flat = self.flat
        if any(isinstance(rej, str) and rej != 'existing' for
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
    def plot_image(self, picks=None, sigma=0., vmin=None, vmax=None,
                   colorbar=True, order=None, show=True, units=None,
                   scalings=None, cmap=None, fig=None, axes=None,
                   overlay_times=None, combine=None, group_by=None,
                   evoked=True, ts_args=None, title=None, clear=False):
        return plot_epochs_image(self, picks=picks, sigma=sigma, vmin=vmin,
                                 vmax=vmax, colorbar=colorbar, order=order,
                                 show=show, units=units, scalings=scalings,
                                 cmap=cmap, fig=fig, axes=axes,
                                 overlay_times=overlay_times, combine=combine,
                                 group_by=group_by, evoked=evoked,
                                 ts_args=ts_args, title=title, clear=clear)

    @verbose
    def drop(self, indices, reason='USER', verbose=None):
        """Drop epochs based on indices or boolean mask.

        .. note:: The indices refer to the current set of undropped epochs
                  rather than the complete set of dropped and undropped epochs.
                  They are therefore not necessarily consistent with any
                  external indices (e.g., behavioral logs). To drop epochs
                  based on external criteria, do not use the ``preload=True``
                  flag when constructing an Epochs object, and call this
                  method before calling the :meth:`mne.Epochs.drop_bad` or
                  :meth:`mne.Epochs.load_data` methods.

        Parameters
        ----------
        indices : array of int or bool
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are
            correspondingly modified.
        reason : str
            Reason for dropping the epochs ('ECG', 'timeout', 'blink' etc).
            Default: 'USER'.
        %(verbose_meth)s

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
        try_idx = np.where(indices < 0, indices + len(self.events), indices)

        out_of_bounds = (try_idx < 0) | (try_idx >= len(self.events))
        if out_of_bounds.any():
            first = indices[out_of_bounds][0]
            raise IndexError("Epoch index %d is out of bounds" % first)
        keep = np.setdiff1d(np.arange(len(self.events)), try_idx)
        self._getitem(keep, reason, copy=False, drop_event_id=False)
        count = len(try_idx)
        logger.info('Dropped %d epoch%s: %s' %
                    (count, _pl(count), ', '.join(map(str, np.sort(try_idx)))))

        return self

    def _get_epoch_from_raw(self, idx, verbose=None):
        """Get a given epoch from disk."""
        raise NotImplementedError

    def _project_epoch(self, epoch):
        """Process a raw epoch based on the delayed param."""
        # whenever requested, the first epoch is being projected.
        if (epoch is None) or isinstance(epoch, str):
            # can happen if t < 0 or reject based on annotations
            return epoch
        proj = self._do_delayed_proj or self.proj
        if self._projector is not None and proj is True:
            epoch = np.dot(self._projector, epoch)
        return epoch

    @verbose
    def _get_data(self, out=True, picks=None, item=None, verbose=None):
        """Load all data, dropping bad epochs along the way.

        Parameters
        ----------
        out : bool
            Return the data. Setting this to False is used to reject bad
            epochs without caching all the data, which saves memory.
        %(picks_all)s
        %(verbose_meth)s
        """
        if item is None:
            item = slice(None)
        elif not self._bad_dropped:
            raise ValueError(
                'item must be None in epochs.get_data() unless bads have been '
                'dropped. Consider using epochs.drop_bad().')
        select = self._item_to_select(item)  # indices or slice
        use_idx = np.arange(len(self.events))[select]
        n_events = len(use_idx)
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
                data = data[select]
                if picks is None:
                    return data
                else:
                    picks = _picks_to_idx(self.info, picks)
                    return data[:, picks]

            # we need to load from disk, drop, and return data
            for ii, idx in enumerate(use_idx):
                # faster to pre-allocate memory here
                epoch_noproj = self._get_epoch_from_raw(idx)
                epoch_noproj = self._detrend_offset_decim(epoch_noproj)
                if self._do_delayed_proj:
                    epoch_out = epoch_noproj
                else:
                    epoch_out = self._project_epoch(epoch_noproj)
                if ii == 0:
                    data = np.empty((n_events, len(self.ch_names),
                                     len(self.times)), dtype=epoch_out.dtype)
                data[ii] = epoch_out
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

            # adjust the data size if there is a reason to (output or update)
            if out or self.preload:
                if data.flags['OWNDATA'] and data.flags['C_CONTIGUOUS']:
                    data.resize((n_out,) + data.shape[1:], refcheck=False)
                else:
                    data = data[:n_out]
                    if self.preload:
                        self._data = data

            # Now update our properties (excepd data, which is already fixed)
            self._getitem(good_idx, None, copy=False, drop_event_id=False,
                          select_data=False)

        if out:
            if picks is None:
                return data
            else:
                picks = _picks_to_idx(self.info, picks)
                return data[:, picks]
        else:
            return None

    @fill_doc
    def get_data(self, picks=None, item=None):
        """Get all epochs as a 3D array.

        Parameters
        ----------
        %(picks_all)s
        item : slice | array-like | str | list | None
            The items to get. See :meth:`mne.Epochs.__getitem__` for
            a description of valid options. This can be substantially faster
            for obtaining an ndarray than :meth:`~mne.Epochs.__getitem__`
            for repeated access on large Epochs objects.
            None (default) is an alias for ``slice(None)``.

            .. versionadded:: 0.20

        Returns
        -------
        data : array of shape (n_epochs, n_channels, n_times)
            A view on epochs data.
        """
        return self._get_data(picks=picks, item=item)

    @property
    def times(self):
        """Time vector in seconds."""
        return self._times_readonly

    def _set_times(self, times):
        """Set self._times_readonly (and make it read only)."""
        # naming used to indicate that it shouldn't be
        # changed directly, but rather via this method
        self._times_readonly = times.copy()
        self._times_readonly.flags['WRITEABLE'] = False

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
        s = ' %s events ' % len(self.events)
        s += '(all good)' if self._bad_dropped else '(good & bad)'
        s += ', %g - %g sec' % (self.tmin, self.tmax)
        s += ', baseline '
        if self.baseline is None:
            s += 'off'
        else:
            s += '[%s, %s]' % tuple(['None' if b is None else ('%g' % b)
                                     for b in self.baseline])
        s += ', ~%s' % (sizeof_fmt(self._size),)
        s += ', data%s loaded' % ('' if self.preload else ' not')
        s += ', with metadata' if self.metadata is not None else ''
        counts = ['%r: %i' % (k, sum(self.events[:, 2] == v))
                  for k, v in sorted(self.event_id.items())]
        if len(self.event_id) > 0:
            s += ',' + '\n '.join([''] + counts)
        class_name = self.__class__.__name__
        class_name = 'Epochs' if class_name == 'BaseEpochs' else class_name
        return '<%s  |  %s>' % (class_name, s)

    @fill_doc
    def crop(self, tmin=None, tmax=None, include_tmax=True):
        """Crop a time interval from the epochs.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        %(include_tmax)s

        Returns
        -------
        epochs : instance of Epochs
            The cropped epochs.

        Notes
        -----
        Unlike Python slices, MNE time intervals include both their end points;
        crop(tmin, tmax) returns the interval tmin <= t <= tmax.

        Note that the object is modified in place.
        """
        # XXX this could be made to work on non-preloaded data...
        _check_preload(self, 'Modifying data of epochs')

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

        tmask = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'],
                           include_tmax=include_tmax)
        self._set_times(self.times[tmask])
        self._raw_times = self._raw_times[tmask]
        self._data = self._data[:, :, tmask]
        try:
            _check_baseline(self.baseline, tmin, tmax, self.info['sfreq'])
        except ValueError:  # in no longer applies, wipe it out
            warn('Cropping removes baseline period, setting '
                 'epochs.baseline = None')
            self.baseline = None
        return self

    def copy(self):
        """Return copy of Epochs instance.

        Returns
        -------
        epochs : instance of Epochs
            A copy of the object.
        """
        raw = self._raw
        del self._raw
        new = deepcopy(self)
        self._raw = raw
        new._raw = raw
        new._set_times(new.times)  # sets RO
        return new

    @verbose
    def save(self, fname, split_size='2GB', fmt='single', overwrite=False,
             verbose=True):
        """Save epochs in a fif file.

        Parameters
        ----------
        fname : str
            The name of the file, which should end with -epo.fif or
            -epo.fif.gz.
        split_size : str | int
            Large raw files are automatically split into multiple pieces. This
            parameter specifies the maximum size of each piece. If the
            parameter is an integer, it specifies the size in Bytes. It is
            also possible to pass a human-readable string, e.g., 100MB.
            Note: Due to FIFF file limitations, the maximum split size is 2GB.

            .. versionadded:: 0.10.0
        fmt : str
            Format to save data. Valid options are 'double' or
            'single' for 64- or 32-bit float, or for 128- or
            64-bit complex numbers respectively. Note: Data are processed with
            double precision. Choosing single-precision, the saved data
            will slightly differ due to the reduction in precision.

            .. versionadded:: 0.17
        overwrite : bool
            If True, the destination file (if it exists) will be overwritten.
            If False (default), an error will be raised if the file exists.
            To overwrite original file (the same one that was loaded),
            data must be preloaded upon reading. This defaults to True in 0.18
            but will change to False in 0.19.

            .. versionadded:: 0.18
        %(verbose_meth)s

        Notes
        -----
        Bad epochs will be dropped before saving the epochs to disk.
        """
        check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz',
                                      '_epo.fif', '_epo.fif.gz'))

        # check for file existence
        _check_fname(fname, overwrite)

        split_size = _get_split_size(split_size)

        _check_option('fmt', fmt, ['single', 'double'])

        # to know the length accurately. The get_data() call would drop
        # bad epochs anyway
        self.drop_bad()
        if len(self) == 0:
            warn('Saving epochs with no data')
            total_size = 0
        else:
            d = self[0].get_data()
            # this should be guaranteed by subclasses
            assert d.dtype in ('>f8', '<f8', '>c16', '<c16')
            total_size = d.nbytes * len(self)
        self._check_consistency()
        if fmt == "single":
            total_size //= 2  # 64bit data converted to 32bit before writing.
        total_size += 32  # FIF tags
        # Account for all the other things we write, too
        # 1. meas_id block plus main epochs block
        total_size += 132
        # 2. measurement info (likely slight overestimate, but okay)
        total_size += object_size(self.info)
        # 3. events and event_id in its own block
        total_size += (self.events.size * 4 +
                       len(_event_id_string(self.event_id)) + 72)
        # 4. Metadata in a block of its own
        if self.metadata is not None:
            total_size += len(_prepare_write_metadata(self.metadata)) + 56
        # 5. first sample, last sample, baseline
        total_size += 40 + 40 * (self.baseline is not None)
        # 6. drop log
        total_size += len(json.dumps(self.drop_log)) + 16
        # 7. reject params
        reject_params = _pack_reject_params(self)
        if reject_params:
            total_size += len(json.dumps(reject_params)) + 16
        # 8. selection
        total_size += self.selection.size * 4 + 16
        # 9. end of file tags
        total_size += _NEXT_FILE_BUFFER

        # This is like max(int(ceil(total_size / split_size)), 1) but cleaner
        n_parts = (total_size - 1) // split_size + 1
        assert n_parts >= 1
        epoch_idxs = np.array_split(np.arange(len(self)), n_parts)

        for part_idx, epoch_idx in enumerate(epoch_idxs):
            this_epochs = self[epoch_idx] if n_parts > 1 else self
            # avoid missing event_ids in splits
            this_epochs.event_id = self.event_id
            _save_split(this_epochs, fname, part_idx, n_parts, fmt)

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
            event_ids = [[x] if isinstance(x, str) else x
                         for x in event_ids]
            for ids_ in event_ids:  # check if tagging is attempted
                if any([id_ not in ids for id_ in ids_]):
                    tagging = True
            # 1. treat everything that's not in event_id as a tag
            # 2a. for tags, find all the event_ids matched by the tags
            # 2b. for non-tag ids, just pass them directly
            # 3. do this for every input
            event_ids = [[k for k in ids
                          if all((tag in k.split("/")
                                  for tag in id_))]  # ids matching all tags
                         if all(id__ not in ids for id__ in id_)
                         else id_  # straight pass for non-tag inputs
                         for id_ in event_ids]
            for ii, id_ in enumerate(event_ids):
                if len(id_) == 0:
                    raise KeyError(orig_ids[ii] + "not found in the "
                                   "epoch object's event_id.")
                elif len({sub_id in ids for sub_id in id_}) != 1:
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
            eq_inds.append(self._keys_to_idx(eq))

        event_times = [self.events[e, 0] for e in eq_inds]
        indices = _get_drop_indices(event_times, method)
        # need to re-index indices
        indices = np.concatenate([e[idx] for e, idx in zip(eq_inds, indices)])
        self.drop(indices, reason='EQUALIZED_COUNT')
        # actually remove the indices
        return self, indices

    @fill_doc
    def to_data_frame(self, picks=None, index=None, scaling_time=None,
                      scalings=None, copy=True, long_format=False,
                      time_format='ms'):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        additional columns "time", "epoch" (epoch number), and "condition"
        (epoch event description) are added, unless ``index`` is not ``None``
        (in which case the columns specified in ``index`` will be used to form
        the DataFrame's index instead).

        Parameters
        ----------
        %(picks_all)s
        %(df_index_epo)s
            Valid string values are 'time', 'epoch', and 'condition'.
            Defaults to ``None``.
        %(df_scaling_time_deprecated)s
        %(df_scalings)s
        %(df_copy)s
        %(df_longform_epo)s
        %(df_time_format)s

            .. versionadded:: 0.20

        Returns
        -------
        %(df_return)s
        """
        # check deprecation
        _check_scaling_time(scaling_time)
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ['time', 'epoch', 'condition']
        valid_time_formats = ['ms', 'timedelta']
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        data = self.get_data()[:, picks, :]
        times = self.times
        n_epochs, n_picks, n_times = data.shape
        data = np.hstack(data).T  # (time*epochs) x signals
        if copy:
            data = data.copy()
        data = _scale_dataframe_data(self, data, picks, scalings)
        # prepare extra columns / multiindex
        mindex = list()
        times = np.tile(times, n_epochs)
        times = _convert_times(self, times, time_format)
        mindex.append(('time', times))
        rev_event_id = {v: k for k, v in self.event_id.items()}
        conditions = [rev_event_id[k] for k in self.events[:, 2]]
        mindex.append(('condition', np.repeat(conditions, n_times)))
        mindex.append(('epoch', np.repeat(self.selection, n_times)))
        assert all(len(mdx) == len(mindex[0]) for mdx in mindex)
        # build DataFrame
        df = _build_data_frame(self, data, picks, long_format, mindex, index,
                               default_index=['condition', 'epoch', 'time'])
        return df

    def as_type(self, ch_type='grad', mode='fast'):
        """Compute virtual epochs using interpolated fields.

        .. Warning:: Using virtual epochs to compute inverse can yield
            unexpected results. The virtual channels have `'_v'` appended
            at the end of the names to emphasize that the data contained in
            them are interpolated.

        Parameters
        ----------
        ch_type : str
            The destination channel type. It can be 'mag' or 'grad'.
        mode : str
            Either `'accurate'` or `'fast'`, determines the quality of the
            Legendre polynomial expansion used. `'fast'` should be sufficient
            for most applications.

        Returns
        -------
        epochs : instance of mne.EpochsArray
            The transformed epochs object containing only virtual channels.

        Notes
        -----
        This method returns a copy and does not modify the data it
        operates on. It also returns an EpochsArraw instance.

        .. versionadded:: 0.20.0
        """
        from .forward import _as_meg_type_inst
        return _as_meg_type_inst(self, ch_type=ch_type, mode=mode)


def _check_baseline(baseline, tmin, tmax, sfreq):
    """Check for a valid baseline."""
    if baseline is not None:
        if not isinstance(baseline, tuple) or len(baseline) != 2:
            raise ValueError('`baseline=%s` is an invalid argument, must be '
                             'a tuple of length 2 or None' % str(baseline))
        # check default value of baseline and `tmin=0`
        if baseline == (None, 0) and tmin == 0:
            raise ValueError('Baseline interval is only one sample. Use '
                             '`baseline=(0, 0)` if this is desired.')

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


def _drop_log_stats(drop_log, ignore=('IGNORED',)):
    """Compute drop log stats.

    Parameters
    ----------
    drop_log : list of list
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


@fill_doc
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
        Start time before event. If nothing is provided, defaults to -0.2.
    tmax : float
        End time after event. If nothing is provided, defaults to 0.5.
    baseline : None or tuple of length 2 (default (None, 0))
        The time interval to apply baseline correction. If None do not apply
        it. If baseline is (a, b) the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used and if b is None then b
        is set to the end of the interval. If baseline is equal to (None, None)
        all the time interval is used. Correction is applied by computing mean
        of the baseline period and subtracting it from the data. The baseline
        (a, b) includes both endpoints, i.e. all timepoints t such that
        a <= t <= b.
    %(picks_all)s
    preload : bool
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
    metadata : instance of pandas.DataFrame | None
        A :class:`pandas.DataFrame` specifying metadata about each epoch.
        If given, ``len(metadata)`` must equal ``len(events)``. The DataFrame
        may only contain values of type (str | int | float | bool).
        If metadata is given, then pandas-style queries may be used to select
        subsets of data, see :meth:`mne.Epochs.__getitem__`.
        When a subset of the epochs is created in this (or any other
        supported) manner, the metadata object is subsetted accordingly, and
        the row indices will be modified to match ``epochs.selection``.

        .. versionadded:: 0.16
    event_repeated : str
        How to handle duplicates in `events[:, 0]`. Can be 'error' (default),
        to raise an error, 'drop' to only retain the row occurring first in the
        `events`, or 'merge' to combine the coinciding events (=duplicates)
        into a new event (see Notes for details).

        .. versionadded:: 0.19
    %(verbose)s

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
    drop_log : list of list
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
    times :  ndarray
        Time vector in seconds. Goes from `tmin` to `tmax`. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.
    %(verbose)s

    See Also
    --------
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts

    Notes
    -----
    When accessing data, Epochs are detrended, baseline-corrected, and
    decimated, then projectors are (optionally) applied.

    For indexing and slicing using ``epochs[...]``, see
    :meth:`mne.Epochs.__getitem__`.

    All methods for iteration over objects (using :meth:`mne.Epochs.__iter__`,
    :meth:`mne.Epochs.iter_evoked` or :meth:`mne.Epochs.next`) use the same
    internal state.

    If `event_repeated` is set to "merge", the coinciding events (duplicates)
    will be merged into a single event_id and assigned a new id_number as
    follows: `event_id['{event_id_1}/{event_id_2}/...'] = new_id_number`.
    For example with the event_id {'aud': 1, 'vis': 2} and the events
    [[0, 0, 1], [0, 0, 2]], the "merge" behavior will update both event_id and
    events to be: {'aud/vis': 3} and [[0, 0, 3], ] respectively.
    """

    @verbose
    def __init__(self, raw, events, event_id=None, tmin=-0.2, tmax=0.5,
                 baseline=(None, 0), picks=None, preload=False, reject=None,
                 flat=None, proj=True, decim=1, reject_tmin=None,
                 reject_tmax=None, detrend=None, on_missing='error',
                 reject_by_annotation=True, metadata=None,
                 event_repeated='error', verbose=None):  # noqa: D102
        if not isinstance(raw, BaseRaw):
            raise ValueError('The first argument to `Epochs` must be an '
                             'instance of mne.io.BaseRaw')
        info = deepcopy(raw.info)

        # proj is on when applied in Raw
        proj = proj or raw.proj

        self.reject_by_annotation = reject_by_annotation
        # call BaseEpochs constructor
        super(Epochs, self).__init__(
            info, None, events, event_id, tmin, tmax, metadata=metadata,
            baseline=baseline, raw=raw, picks=picks, reject=reject,
            flat=flat, decim=decim, reject_tmin=reject_tmin,
            reject_tmax=reject_tmax, detrend=detrend,
            proj=proj, on_missing=on_missing, preload_at_end=preload,
            event_repeated=event_repeated, verbose=verbose)

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
        logger.debug('    Getting epoch for %d-%d' % (start, stop))
        data = self._raw._check_bad_segment(start, stop, self.picks,
                                            self.reject_by_annotation)
        return data


@fill_doc
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
        Start time before event. If nothing provided, defaults to 0.
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
    on_missing : str
        See :class:`mne.Epochs` docstring for details.
    metadata : instance of pandas.DataFrame | None
        See :class:`mne.Epochs` docstring for details.

        .. versionadded:: 0.16
    selection : ndarray | None
        The selection compared to the original set of epochs.
        Can be None to use ``np.arange(len(events))``.

        .. versionadded:: 0.16
    %(verbose)s

    See Also
    --------
    create_info
    EvokedArray
    io.RawArray

    Notes
    -----
    Proper units of measure:

    * V: eeg, eog, seeg, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc
    """

    @verbose
    def __init__(self, data, info, events=None, tmin=0, event_id=None,
                 reject=None, flat=None, reject_tmin=None,
                 reject_tmax=None, baseline=None, proj=True,
                 on_missing='error', metadata=None, selection=None,
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
            events = _gen_events(n_epochs)
        if data.shape[0] != len(events):
            raise ValueError('The number of epochs and the number of events'
                             'must match')
        info = info.copy()  # do not modify original info
        tmax = (data.shape[2] - 1) / info['sfreq'] + tmin
        if event_id is None:  # convert to int to make typing-checks happy
            event_id = {str(e): int(e) for e in np.unique(events[:, 2])}
        super(EpochsArray, self).__init__(
            info, data, events, event_id, tmin, tmax, baseline, reject=reject,
            flat=flat, reject_tmin=reject_tmin, reject_tmax=reject_tmax,
            decim=1, metadata=metadata, selection=selection, proj=proj,
            on_missing=on_missing)
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
    new_event_num = operator.index(new_event_num)
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

    Parameters
    ----------
    epochs_list : list of Epochs instances
        The Epochs instances to equalize trial counts for.
    method : str
        If 'truncate', events will be truncated from the end of each event
        list. If 'mintime', timing differences between each event list will be
        minimized.

    Notes
    -----
    This tries to make the remaining epochs occurring as close as possible in
    time. This method works based on the idea that if there happened to be some
    time-varying (like on the scale of minutes) noise characteristics during
    a recording, they could be compensated for (to some extent) in the
    equalization process. This method thus seeks to reduce any of those effects
    by minimizing the differences in the times of the events in the two sets of
    epochs. For example, if one had event times [1, 2, 3, 4, 120, 121] and the
    other one had [3.5, 4.5, 120.5, 121.5], it would remove events at times
    [1, 2] in the first epochs and not [120, 121].

    Examples
    --------
    >>> equalize_epoch_counts([epochs1, epochs2])  # doctest: +SKIP
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
    _check_option('method', method, ['mintime', 'truncate'])
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
    """Fix bug on old scipy."""
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
            for key, thresh in refl.items():
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

        #   Metadata
        metadata = None
        metadata_tree = dir_tree_find(tree, FIFF.FIFFB_MNE_METADATA)
        if len(metadata_tree) > 0:
            for dd in metadata_tree[0]['directory']:
                kind = dd.kind
                pos = dd.pos
                if kind == FIFF.FIFF_DESCRIPTION:
                    metadata = read_tag(fid, pos).data
                    metadata = _prepare_read_metadata(metadata)
                    break

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        del meas
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
        data = None
        data_tag = None
        bmin, bmax = None, None
        baseline = None
        selection = None
        drop_log = None
        reject_params = {}
        for k in range(my_epochs['nent']):
            kind = my_epochs['directory'][k].kind
            pos = my_epochs['directory'][k].pos
            if kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, pos)
                first = int(tag.data)
            elif kind == FIFF.FIFF_LAST_SAMPLE:
                tag = read_tag(fid, pos)
                last = int(tag.data)
            elif kind == FIFF.FIFF_EPOCH:
                # delay reading until later
                fid.seek(pos, 0)
                data_tag = read_tag_info(fid)
                data_tag.pos = pos
                data_tag.type = data_tag.type ^ (1 << 30)
            elif kind in [FIFF.FIFF_MNE_BASELINE_MIN, 304]:
                # Constant 304 was used before v0.11
                tag = read_tag(fid, pos)
                bmin = float(tag.data)
            elif kind in [FIFF.FIFF_MNE_BASELINE_MAX, 305]:
                # Constant 305 was used before v0.11
                tag = read_tag(fid, pos)
                bmax = float(tag.data)
            elif kind == FIFF.FIFF_MNE_EPOCHS_SELECTION:
                tag = read_tag(fid, pos)
                selection = np.array(tag.data)
            elif kind == FIFF.FIFF_MNE_EPOCHS_DROP_LOG:
                tag = read_tag(fid, pos)
                drop_log = json.loads(tag.data)
            elif kind == FIFF.FIFF_MNE_EPOCHS_REJECT_FLAT:
                tag = read_tag(fid, pos)
                reject_params = json.loads(tag.data)

        if bmin is not None or bmax is not None:
            baseline = (bmin, bmax)

        n_samp = last - first + 1
        logger.info('    Found the data of interest:')
        logger.info('        t = %10.2f ... %10.2f ms'
                    % (1000 * first / info['sfreq'],
                       1000 * last / info['sfreq']))
        if info['comps'] is not None:
            logger.info('        %d CTF compensation matrices available'
                        % len(info['comps']))

        # Inspect the data
        if data_tag is None:
            raise ValueError('Epochs data not found')
        epoch_shape = (len(info['ch_names']), n_samp)
        size_expected = len(events) * np.prod(epoch_shape)
        # on read double-precision is always used
        if data_tag.type == FIFF.FIFFT_FLOAT:
            datatype = np.float64
            fmt = '>f4'
        elif data_tag.type == FIFF.FIFFT_DOUBLE:
            datatype = np.float64
            fmt = '>f8'
        elif data_tag.type == FIFF.FIFFT_COMPLEX_FLOAT:
            datatype = np.complex128
            fmt = '>c8'
        elif data_tag.type == FIFF.FIFFT_COMPLEX_DOUBLE:
            datatype = np.complex128
            fmt = '>c16'
        fmt_itemsize = np.dtype(fmt).itemsize
        assert fmt_itemsize in (4, 8, 16)
        size_actual = data_tag.size // fmt_itemsize - 16 // fmt_itemsize

        if not size_actual == size_expected:
            raise ValueError('Incorrect number of samples (%d instead of %d)'
                             % (size_actual, size_expected))

        # Calibration factors
        cals = np.array([[info['chs'][k]['cal'] *
                          info['chs'][k].get('scale', 1.0)]
                         for k in range(info['nchan'])], np.float64)

        # Read the data
        if preload:
            data = read_tag(fid, data_tag.pos).data.astype(datatype)
            data *= cals

        # Put it all together
        tmin = first / info['sfreq']
        tmax = last / info['sfreq']
        event_id = ({str(e): e for e in np.unique(events[:, 2])}
                    if mappings is None else mappings)
        # In case epochs didn't have a FIFF.FIFF_MNE_EPOCHS_SELECTION tag
        # (version < 0.8):
        if selection is None:
            selection = np.arange(len(events))
        if drop_log is None:
            drop_log = [[] for _ in range(len(events))]

    return (info, data, data_tag, events, event_id, metadata, tmin, tmax,
            baseline, selection, drop_log, epoch_shape, cals, reject_params,
            fmt)


@verbose
def read_epochs(fname, proj=True, preload=True, verbose=None):
    """Read epochs from a fif file.

    Parameters
    ----------
    fname : str | file-like
        The epochs filename to load. Filename should end with -epo.fif or
        -epo.fif.gz. If a file-like object is provided, preloading must be
        used.
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
    %(verbose)s

    Returns
    -------
    epochs : instance of Epochs
        The epochs.
    """
    return EpochsFIF(fname, proj, preload, verbose)


class _RawContainer(object):
    """Helper for a raw data container."""

    def __init__(self, fid, data_tag, event_samps, epoch_shape,
                 cals, fmt):  # noqa: D102
        self.fid = fid
        self.data_tag = data_tag
        self.event_samps = event_samps
        self.epoch_shape = epoch_shape
        self.cals = cals
        self.proj = False
        self.fmt = fmt

    def __del__(self):  # noqa: D105
        self.fid.close()


@fill_doc
class EpochsFIF(BaseEpochs):
    """Epochs read from disk.

    Parameters
    ----------
    fname : str | file-like
        The name of the file, which should end with -epo.fif or -epo.fif.gz. If
        a file-like object is provided, preloading must be used.
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
    %(verbose)s

    See Also
    --------
    mne.Epochs
    mne.epochs.combine_event_ids
    mne.Epochs.equalize_event_counts
    """

    @verbose
    def __init__(self, fname, proj=True, preload=True,
                 verbose=None):  # noqa: D102
        if isinstance(fname, str):
            check_fname(fname, 'epochs', ('-epo.fif', '-epo.fif.gz',
                                          '_epo.fif', '_epo.fif.gz'))
        elif not preload:
            raise ValueError('preload must be used with file-like objects')

        fnames = [fname]
        ep_list = list()
        raw = list()
        for fname in fnames:
            fname_rep = _get_fname_rep(fname)
            logger.info('Reading %s ...' % fname_rep)
            fid, tree, _ = fiff_open(fname, preload=preload)
            next_fname = _get_next_fname(fid, fname, tree)
            (info, data, data_tag, events, event_id, metadata, tmin, tmax,
             baseline, selection, drop_log, epoch_shape, cals,
             reject_params, fmt) = \
                _read_one_epoch_file(fid, tree, preload)

            # here we ignore missing events, since users should already be
            # aware of missing events if they have saved data that way
            epoch = BaseEpochs(
                info, data, events, event_id, tmin, tmax, baseline,
                metadata=metadata, on_missing='ignore',
                selection=selection, drop_log=drop_log,
                proj=False, verbose=False)
            ep_list.append(epoch)
            if not preload:
                # store everything we need to index back to the original data
                raw.append(_RawContainer(fiff_open(fname)[0], data_tag,
                                         events[:, 0].copy(), epoch_shape,
                                         cals, fmt))

            if next_fname is not None:
                fnames.append(next_fname)

        (info, data, events, event_id, tmin, tmax, metadata, baseline,
         selection, drop_log, _) = \
            _concatenate_epochs(ep_list, with_data=preload, add_offset=False)
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
            proj=proj, preload_at_end=False, on_missing='ignore',
            selection=selection, drop_log=drop_log, filename=fname_rep,
            metadata=metadata, verbose=verbose, **reject_params)
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
                fmt = raw.fmt
                idx = idx[0]
                size = np.prod(raw.epoch_shape) * np.dtype(fmt).itemsize
                offset = idx * size + 16  # 16 = Tag header
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
        raw.fid.seek(raw.data_tag.pos + offset, 0)
        if fmt == '>c8':
            read_fmt = '>f4'
        elif fmt == '>c16':
            read_fmt = '>f8'
        else:
            read_fmt = fmt
        data = np.frombuffer(raw.fid.read(size), read_fmt)
        if read_fmt != fmt:
            data = data.view(fmt)
            data = data.astype(np.complex128)
        else:
            data = data.astype(np.float64)

        data.shape = raw.epoch_shape
        data *= raw.cals
        return data


@fill_doc
def bootstrap(epochs, random_state=None):
    """Compute epochs selected by bootstrapping.

    Parameters
    ----------
    epochs : Epochs instance
        epochs data to be bootstrapped
    %(random_state)s

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
    idx = rng_uniform(rng)(0, n_events, n_events)
    epochs_bootstrap = epochs_bootstrap[idx]
    return epochs_bootstrap


def _check_merge_epochs(epochs_list):
    """Aux function."""
    if len({tuple(epochs.event_id.items()) for epochs in epochs_list}) != 1:
        raise NotImplementedError("Epochs with unequal values for event_id")
    if len({epochs.tmin for epochs in epochs_list}) != 1:
        raise NotImplementedError("Epochs with unequal values for tmin")
    if len({epochs.tmax for epochs in epochs_list}) != 1:
        raise NotImplementedError("Epochs with unequal values for tmax")
    if len({epochs.baseline for epochs in epochs_list}) != 1:
        raise NotImplementedError("Epochs with unequal values for baseline")


@verbose
def add_channels_epochs(epochs_list, verbose=None):
    """Concatenate channels, info and data from two Epochs objects.

    Parameters
    ----------
    epochs_list : list of Epochs
        Epochs object to concatenate.
    %(verbose)s Defaults to True if any of the input epochs have verbose=True.

    Returns
    -------
    epochs : instance of Epochs
        Concatenated epochs.
    """
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

    proj = any(e.proj for e in epochs_list)

    if verbose is None:
        verbose = any(e.verbose for e in epochs_list)

    epochs = epochs_list[0].copy()
    epochs.info = info
    epochs.picks = None
    epochs.verbose = verbose
    epochs.events = events
    epochs.preload = True
    epochs._bad_dropped = True
    epochs._data = data
    epochs._projector, epochs.info = setup_proj(epochs.info, False,
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


def _concatenate_epochs(epochs_list, with_data=True, add_offset=True):
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
    metadata = [out.metadata]
    baseline, tmin, tmax = out.baseline, out.tmin, out.tmax
    info = deepcopy(out.info)
    verbose = out.verbose
    drop_log = deepcopy(out.drop_log)
    event_id = deepcopy(out.event_id)
    selection = out.selection
    # offset is the last epoch + tmax + 10 second
    events_offset = (np.max(out.events[:, 0]) +
                     int((10 + tmax) * epochs.info['sfreq']))
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
        evs = epochs.events.copy()
        # add offset
        if add_offset:
            evs[:, 0] += events_offset
        # Update offset for the next iteration.
        # offset is the last epoch + tmax + 10 second
        events_offset += (np.max(epochs.events[:, 0]) +
                          int((10 + tmax) * epochs.info['sfreq']))
        events.append(evs)
        selection = np.concatenate((selection, epochs.selection))
        drop_log.extend(epochs.drop_log)
        event_id.update(epochs.event_id)
        metadata.append(epochs.metadata)
    events = np.concatenate(events, axis=0)

    # Create metadata object (or make it None)
    n_have = sum(this_meta is not None for this_meta in metadata)
    if n_have == 0:
        metadata = None
    elif n_have != len(metadata):
        raise ValueError('%d of %d epochs instances have metadata, either '
                         'all or none must have metadata'
                         % (n_have, len(metadata)))
    else:
        pd = _check_pandas_installed(strict=False)
        if pd is not False:
            metadata = pd.concat(metadata)
        else:  # dict of dicts
            metadata = sum(metadata, list())
    if with_data:
        data = np.concatenate(data, axis=0)
    return (info, data, events, event_id, tmin, tmax, metadata, baseline,
            selection, drop_log, verbose)


def _finish_concat(info, data, events, event_id, tmin, tmax, metadata,
                   baseline, selection, drop_log, verbose):
    """Finish concatenation for epochs not read from disk."""
    selection = np.where([len(d) == 0 for d in drop_log])[0]
    out = BaseEpochs(
        info, data, events, event_id, tmin, tmax, baseline=baseline,
        selection=selection, drop_log=drop_log, proj=False,
        on_missing='ignore', metadata=metadata, verbose=verbose)
    out.drop_bad()
    return out


def concatenate_epochs(epochs_list, add_offset=True):
    """Concatenate a list of epochs into one epochs object.

    Parameters
    ----------
    epochs_list : list
        List of Epochs instances to concatenate (in order).
    add_offset : bool
        If True, a fixed offset is added to the event times from different
        Epochs sets, such that they are easy to distinguish after the
        concatenation.
        If False, the event times are unaltered during the concatenation.

    Returns
    -------
    epochs : instance of Epochs
        The result of the concatenation (first Epochs instance passed in).

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    return _finish_concat(*_concatenate_epochs(epochs_list,
                                               add_offset=add_offset))


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
    %(picks_all_data)s
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

    %(verbose)s

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
    meg_picks, mag_picks, grad_picks, good_mask, _ = \
        _get_mf_picks(epochs.info, int_order, ext_order, ignore_ref)
    coil_scale, mag_scale = _get_coil_scale(
        meg_picks, mag_picks, grad_picks, mag_scale, epochs.info)
    n_channels, n_times = len(epochs.ch_names), len(epochs.times)
    other_picks = np.setdiff1d(np.arange(n_channels), meg_picks)
    data = np.zeros((n_channels, n_times))
    count = 0
    # keep only MEG w/bad channels marked in "info_from"
    info_from = pick_info(epochs.info, meg_picks[good_mask], copy=True)
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
    decomp_coil_scale = coil_scale[good_mask]
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
        data[meg_picks] = np.dot(mapping, data[meg_picks[good_mask]])
    info_to['dev_head_t'] = recon_trans  # set the reconstruction transform
    evoked = epochs._evoked_from_epoch_data(data, info_to, picks,
                                            n_events=count, kind='average',
                                            comment=epochs._name)
    _remove_meg_projs(evoked)  # remove MEG projectors, they won't apply now
    logger.info('Created Evoked dataset from %s epochs' % (count,))
    return (evoked, mapping) if return_mapping else evoked


@verbose
def make_fixed_length_epochs(raw, duration=1.,
                             preload=False, verbose=None):
    """Divide continuous raw data into equal-sized consecutive epochs.

    Parameters
    ----------
    raw : instance of Raw
        Raw data to divide into segments.
    duration : float
        Duration of each epoch in seconds. Defaults to 1.
    %(preload)s
    %(verbose)s

    Returns
    -------
    epochs : instance of Epochs
        Segmented data.

    Notes
    -----
    .. versionadded:: 0.20
    """
    events = make_fixed_length_events(raw, 1, duration=duration)
    delta = 1. / raw.info['sfreq']
    return Epochs(raw, events, event_id=[1], tmin=0.,
                  tmax=duration - delta,
                  verbose=verbose, baseline=None)
