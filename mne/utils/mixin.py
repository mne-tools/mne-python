# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

from collections import OrderedDict
from copy import deepcopy
import logging
import json

import numpy as np

from .check import _check_pandas_installed, _check_preload, _validate_type
from ._logging import warn, verbose
from .numerics import object_size, object_hash, _time_mask


logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)


class SizeMixin(object):
    """Estimate MNE object sizes."""

    def __eq__(self, other):
        """Compare self to other.

        Parameters
        ----------
        other : object
            The object to compare to.

        Returns
        -------
        eq : bool
            True if the two objects are equal.
        """
        return isinstance(other, type(self)) and hash(self) == hash(other)

    @property
    def _size(self):
        """Estimate the object size."""
        try:
            size = object_size(self.info)
        except Exception:
            warn('Could not get size for self.info')
            return -1
        if hasattr(self, 'data'):
            size += object_size(self.data)
        elif hasattr(self, '_data'):
            size += object_size(self._data)
        return size

    def __hash__(self):
        """Hash the object.

        Returns
        -------
        hash : int
            The hash
        """
        from ..evoked import Evoked
        from ..epochs import BaseEpochs
        from ..io.base import BaseRaw
        if isinstance(self, Evoked):
            return object_hash(dict(info=self.info, data=self.data))
        elif isinstance(self, (BaseEpochs, BaseRaw)):
            _check_preload(self, "Hashing ")
            return object_hash(dict(info=self.info, data=self._data))
        else:
            raise RuntimeError('Hashing unknown object type: %s' % type(self))


class GetEpochsMixin(object):
    """Class to add epoch selection and metadata to certain classes."""

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

        1. **Integer or slice:** ``epochs[idx]`` will return an `~mne.Epochs`
           object with a subset of epochs chosen by index (supports single
           index and Python-style slicing).

        2. **String:** ``epochs['name']`` will return an `~mne.Epochs` object
           comprising only the epochs labeled ``'name'`` (i.e., epochs created
           around events with the label ``'name'``).

           If there are no epochs labeled ``'name'`` but there are epochs
           labeled with /-separated tags (e.g. ``'name/left'``,
           ``'name/right'``), then ``epochs['name']`` will select the epochs
           with labels that contain that tag (e.g., ``epochs['left']`` selects
           epochs labeled ``'audio/left'`` and ``'visual/left'``, but not
           ``'audio_left'``).

           If multiple tags are provided *as a single string* (e.g.,
           ``epochs['name_1/name_2']``), this selects epochs containing *all*
           provided tags. For example, ``epochs['audio/left']`` selects
           ``'audio/left'`` and ``'audio/quiet/left'``, but not
           ``'audio/right'``. Note that tag-based selection is insensitive to
           order: tags like ``'audio/left'`` and ``'left/audio'`` will be
           treated the same way when selecting via tag.

        3. **List of strings:** ``epochs[['name_1', 'name_2', ... ]]`` will
           return an `~mne.Epochs` object comprising epochs that match *any* of
           the provided names (i.e., the list of names is treated as an
           inclusive-or condition). If *none* of the provided names match any
           epoch labels, a ``KeyError`` will be raised.

           If epoch labels are /-separated tags, then providing multiple tags
           *as separate list entries* will likewise act as an inclusive-or
           filter. For example, ``epochs[['audio', 'left']]`` would select
           ``'audio/left'``, ``'audio/right'``, and ``'visual/left'``, but not
           ``'visual/right'``.

        4. **Pandas query:** ``epochs['pandas query']`` will return an
           `~mne.Epochs` object with a subset of epochs (and matching
           metadata) selected by the query called with
           ``self.metadata.eval``, e.g.::

               epochs["col_a > 2 and col_b == 'foo'"]

           would return all epochs whose associated ``col_a`` metadata was
           greater than two, and whose ``col_b`` metadata was the string 'foo'.
           Query-based indexing only works if Pandas is installed and
           ``self.metadata`` is a :class:`pandas.DataFrame`.

           .. versionadded:: 0.16
        """
        return self._getitem(item)

    def _item_to_select(self, item):
        if isinstance(item, str):
            item = [item]

        # Convert string to indices
        if isinstance(item, (list, tuple)) and len(item) > 0 and \
                isinstance(item[0], str):
            select = self._keys_to_idx(item)
        elif isinstance(item, slice):
            select = item
        else:
            select = np.atleast_1d(item)
            if len(select) == 0:
                select = np.array([], int)
        return select

    def _getitem(self, item, reason='IGNORED', copy=True, drop_event_id=True,
                 select_data=True, return_indices=False):
        """
        Select epochs from current object.

        Parameters
        ----------
        item: slice, array-like, str, or list
            see `__getitem__` for details.
        reason: str
            entry in `drop_log` for unselected epochs
        copy: bool
            return a copy of the current object
        drop_event_id: bool
            remove non-existing event-ids after selection
        select_data: bool
            apply selection to data
            (use `select_data=False` if subclasses do not have a
             valid `_data` field, or data has already been subselected)
        return_indices: bool
            return the indices of selected epochs from the original object
            in addition to the new `Epochs` objects
        Returns
        -------
        `Epochs` or tuple(Epochs, np.ndarray) if `return_indices` is True
            subset of epochs (and optionally array with kept epoch indices)
        """
        data = self._data
        self._data = None
        inst = self.copy() if copy else self
        self._data = inst._data = data
        del self

        select = inst._item_to_select(item)
        has_selection = hasattr(inst, 'selection')
        if has_selection:
            key_selection = inst.selection[select]
            drop_log = list(inst.drop_log)
            if reason is not None:
                for k in np.setdiff1d(inst.selection, key_selection):
                    drop_log[k] = (reason,)
            inst.drop_log = tuple(drop_log)
            inst.selection = key_selection
            del drop_log

        inst.events = np.atleast_2d(inst.events[select])
        if inst.metadata is not None:
            pd = _check_pandas_installed(strict=False)
            if pd:
                metadata = inst.metadata.iloc[select]
                if has_selection:
                    metadata.index = inst.selection
            else:
                metadata = np.array(inst.metadata, 'object')[select].tolist()

            # will reset the index for us
            GetEpochsMixin.metadata.fset(inst, metadata, verbose=False)
        if inst.preload and select_data:
            # ensure that each Epochs instance owns its own data so we can
            # resize later if necessary
            inst._data = np.require(inst._data[select], requirements=['O'])
        if drop_event_id:
            # update event id to reflect new content of inst
            inst.event_id = {k: v for k, v in inst.event_id.items()
                             if v in inst.events[:, 2]}

        if return_indices:
            return inst, select
        else:
            return inst

    def _keys_to_idx(self, keys):
        """Find entries in event dict."""
        from ..event import match_event_names  # avoid circular import

        keys = keys if isinstance(keys, (list, tuple)) else [keys]
        try:
            # Assume it's a condition name
            return np.where(np.any(
                np.array([self.events[:, 2] == self.event_id[k]
                          for k in match_event_names(self.event_id, keys)]),
                axis=0))[0]
        except KeyError as err:
            # Could we in principle use metadata with these Epochs and keys?
            if (len(keys) != 1 or self.metadata is None):
                # If not, raise original error
                raise
            msg = str(err.args[0])  # message for KeyError
            pd = _check_pandas_installed(strict=False)
            # See if the query can be done
            if pd:
                md = self.metadata if hasattr(self, '_metadata') else None
                self._check_metadata(metadata=md)
                try:
                    # Try metadata
                    vals = self.metadata.reset_index().query(
                        keys[0],
                        engine='python'
                    ).index.values
                except Exception as exp:
                    msg += (' The epochs.metadata Pandas query did not '
                            'yield any results: %s' % (exp.args[0],))
                else:
                    return vals
            else:
                # If not, warn this might be a problem
                msg += (' The epochs.metadata Pandas query could not '
                        'be performed, consider installing Pandas.')
            raise KeyError(msg)

    def __len__(self):
        """Return the number of epochs.

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
        from ..epochs import BaseEpochs
        if isinstance(self, BaseEpochs) and not self._bad_dropped:
            raise RuntimeError('Since bad epochs have not been dropped, the '
                               'length of the Epochs is not known. Load the '
                               'Epochs with preload=True, or call '
                               'Epochs.drop_bad(). To find the number '
                               'of events in the Epochs, use '
                               'len(Epochs.events).')
        return len(self.events)

    def __iter__(self):
        """Facilitate iteration over epochs.

        This method resets the object iteration state to the first epoch.

        Notes
        -----
        This enables the use of this Python pattern::

            >>> for epoch in epochs:  # doctest: +SKIP
            >>>     print(epoch)  # doctest: +SKIP

        Where ``epoch`` is given by successive outputs of
        :meth:`mne.Epochs.next`.
        """
        self._current = 0
        self._current_detrend_picks = self._detrend_picks
        return self

    def __next__(self, return_event_id=False):
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
        if not hasattr(self, '_current_detrend_picks'):
            self.__iter__()  # ensure we're ready to iterate
        if self.preload:
            if self._current >= len(self._data):
                self._stop_iter()
            epoch = self._data[self._current]
            self._current += 1
        else:
            is_good = False
            while not is_good:
                if self._current >= len(self.events):
                    self._stop_iter()
                epoch_noproj = self._get_epoch_from_raw(self._current)
                epoch_noproj = self._detrend_offset_decim(
                    epoch_noproj, self._current_detrend_picks)
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

    def _stop_iter(self):
        del self._current
        del self._current_detrend_picks
        raise StopIteration  # signal the end

    next = __next__  # originally for Python2, now b/c public

    def _check_metadata(self, metadata=None, reset_index=False):
        """Check metadata consistency."""
        # reset_index=False will not copy!
        if metadata is None:
            return
        else:
            pd = _check_pandas_installed(strict=False)
            if pd:
                _validate_type(metadata, types=pd.DataFrame,
                               item_name='metadata')
                if len(metadata) != len(self.events):
                    raise ValueError('metadata must have the same number of '
                                     'rows (%d) as events (%d)'
                                     % (len(metadata), len(self.events)))
                if reset_index:
                    if hasattr(self, 'selection'):
                        # makes a copy
                        metadata = metadata.reset_index(drop=True)
                        metadata.index = self.selection
                    else:
                        metadata = deepcopy(metadata)
            else:
                _validate_type(metadata, types=list,
                               item_name='metadata')
                if reset_index:
                    metadata = deepcopy(metadata)
        return metadata

    @property
    def metadata(self):
        """Get the metadata."""
        return self._metadata

    @metadata.setter
    @verbose
    def metadata(self, metadata, verbose=None):
        metadata = self._check_metadata(metadata, reset_index=True)
        if metadata is not None:
            if _check_pandas_installed(strict=False):
                n_col = metadata.shape[1]
            else:
                n_col = len(metadata[0])
            n_col = ' with %d columns' % n_col
        else:
            n_col = ''
        if hasattr(self, '_metadata') and self._metadata is not None:
            action = 'Removing' if metadata is None else 'Replacing'
            action += ' existing'
        else:
            action = 'Not setting' if metadata is None else 'Adding'
        logger.info('%s metadata%s' % (action, n_col))
        self._metadata = metadata


def _check_decim(info, decim, offset, check_filter=True):
    """Check decimation parameters."""
    if decim < 1 or decim != int(decim):
        raise ValueError('decim must be an integer > 0')
    decim = int(decim)
    new_sfreq = info['sfreq'] / float(decim)
    offset = int(offset)
    if not 0 <= offset < decim:
        raise ValueError(f'decim must be at least 0 and less than {decim}, '
                         f'got {offset}')
    if check_filter:
        lowpass = info['lowpass']
        if decim > 1 and lowpass is None:
            warn('The measurement information indicates data is not low-pass '
                 f'filtered. The decim={decim} parameter will result in a '
                 f'sampling frequency of {new_sfreq} Hz, which can cause '
                 'aliasing artifacts.')
        elif decim > 1 and new_sfreq < 3 * lowpass:
            warn('The measurement information indicates a low-pass frequency '
                 f'of {lowpass} Hz. The decim={decim} parameter will result '
                 f'in a sampling frequency of {new_sfreq} Hz, which can '
                 'cause aliasing artifacts.')  # > 50% nyquist lim
    return decim, offset, new_sfreq


class TimeMixin(object):
    """Class to handle operations on time for MNE objects."""

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
    def tmax(self):
        """Last time point."""
        return self.times[-1]

    @verbose
    def crop(self, tmin=None, tmax=None, include_tmax=True, verbose=None):
        """Crop data to a given time interval.

        Parameters
        ----------
        tmin : float | None
            Start time of selection in seconds.
        tmax : float | None
            End time of selection in seconds.
        %(include_tmax)s
        %(verbose)s

        Returns
        -------
        inst : instance of Raw, Epochs, Evoked, AverageTFR, or SourceEstimate
            The cropped time-series object, modified in-place.

        Notes
        -----
        %(notes_tmax_included_by_default)s
        """
        t_vars = dict(tmin=tmin, tmax=tmax)
        for name, t_var in t_vars.items():
            _validate_type(
                t_var,
                types=("numeric", None),
                item_name=name,
            )

        if tmin is None:
            tmin = self.tmin
        elif tmin < self.tmin:
            warn(f'tmin is not in time interval. tmin is set to '
                 f'{type(self)}.tmin ({self.tmin:g} sec)')
            tmin = self.tmin

        if tmax is None:
            tmax = self.tmax
        elif tmax > self.tmax:
            warn(f'tmax is not in time interval. tmax is set to '
                 f'{type(self)}.tmax ({self.tmax:g} sec)')
            tmax = self.tmax
            include_tmax = True

        mask = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'],
                          include_tmax=include_tmax)
        self._set_times(self.times[mask])
        self._raw_times = self._raw_times[mask]
        self._update_first_last()
        self._data = self._data[..., mask]

        return self

    @verbose
    def decimate(self, decim, offset=0, verbose=None):
        """Decimate the time-series data.

        Parameters
        ----------
        %(decim)s
        %(offset_decim)s
        %(verbose)s

        Returns
        -------
        inst : MNE-object
            The decimated object.

        See Also
        --------
        mne.Epochs.resample
        mne.io.Raw.resample

        Notes
        -----
        %(decim_notes)s

        If ``decim`` is 1, this method does not copy the underlying data.

        .. versionadded:: 0.10.0

        References
        ----------
        .. footbibliography::
        """
        # if epochs have frequencies, they are not in time (EpochsTFR)
        # and so do not need to be checked whether they have been
        # appropriately filtered to avoid aliasing
        decim, offset, new_sfreq = _check_decim(
            self.info, decim, offset, check_filter=not hasattr(self, 'freqs'))
        start_idx = int(round(-self._raw_times[0] * (self.info['sfreq'] *
                                                     self._decim)))
        self._decim *= decim
        i_start = start_idx % self._decim + offset
        decim_slice = slice(i_start, None, self._decim)
        with self.info._unlock():
            self.info['sfreq'] = new_sfreq

        if self.preload:
            if decim != 1:
                self._data = self._data[..., decim_slice].copy()
                self._raw_times = self._raw_times[decim_slice].copy()
            else:
                self._data = np.ascontiguousarray(self._data)
            self._decim_slice = slice(None)
            self._decim = 1
        else:
            self._decim_slice = decim_slice
        self._set_times(self._raw_times[self._decim_slice])
        self._update_first_last()
        return self

    def time_as_index(self, times, use_rounding=False):
        """Convert time to indices.

        Parameters
        ----------
        times : list-like | float | int
            List of numbers or a number representing points in time.
        use_rounding : bool
            If True, use rounding (instead of truncation) when converting
            times to indices. This can help avoid non-unique indices.

        Returns
        -------
        index : ndarray
            Indices corresponding to the times supplied.
        """
        from ..source_estimate import _BaseSourceEstimate
        if isinstance(self, _BaseSourceEstimate):
            sfreq = 1. / self.tstep
        else:
            sfreq = self.info['sfreq']
        index = (np.atleast_1d(times) - self.times[0]) * sfreq
        if use_rounding:
            index = np.round(index)
        return index.astype(int)

    def _handle_tmin_tmax(self, tmin, tmax):
        """Convert seconds to index into data.

        Parameters
        ----------
        tmin : int | float | None
            Start time of data to get in seconds.
        tmax : int | float | None
            End time of data to get in seconds.

        Returns
        -------
        start : int
            Integer index into data corresponding to tmin.
        stop : int
            Integer index into data corresponding to tmax.

        """
        _validate_type(tmin, types=('numeric', None), item_name='tmin',
                       type_name="int, float, None")
        _validate_type(tmax, types=('numeric', None), item_name='tmax',
                       type_name='int, float, None')

        # handle tmin/tmax as start and stop indices into data array
        n_times = self.times.size
        start = 0 if tmin is None else self.time_as_index(tmin)[0]
        stop = n_times if tmax is None else self.time_as_index(tmax)[0]

        # truncate start/stop to the open interval [0, n_times]
        start = min(max(0, start), n_times)
        stop = min(max(0, stop), n_times)

        return start, stop

    def shift_time(self, tshift, relative=True):
        """Shift time scale in epoched or evoked data.

        Parameters
        ----------
        tshift : float
            The (absolute or relative) time shift in seconds. If ``relative``
            is True, positive tshift increases the time value associated with
            each sample, while negative tshift decreases it.
        relative : bool
            If True, increase or decrease time values by ``tshift`` seconds.
            Otherwise, shift the time values such that the time of the first
            sample equals ``tshift``.

        Returns
        -------
        epochs : MNE-object
            The modified instance.

        Notes
        -----
        This method allows you to shift the *time* values associated with each
        data sample by an arbitrary amount. It does *not* resample the signal
        or change the *data* values in any way.
        """
        _check_preload(self, 'shift_time')
        start = tshift + (self.times[0] if relative else 0.)
        new_times = start + np.arange(len(self.times)) / self.info['sfreq']
        self._set_times(new_times)
        self._update_first_last()
        return self

    def _update_first_last(self):
        """Update self.first and self.last (sample indices)."""
        self.first = int(round(self.times[0] * self.info['sfreq']))
        self.last = len(self.times) + self.first - 1


def _prepare_write_metadata(metadata):
    """Convert metadata to JSON for saving."""
    if metadata is not None:
        if not isinstance(metadata, list):
            metadata = metadata.to_json(orient='records')
        else:  # Pandas DataFrame
            metadata = json.dumps(metadata)
        assert isinstance(metadata, str)
    return metadata


def _prepare_read_metadata(metadata):
    """Convert saved metadata back from JSON."""
    if metadata is not None:
        pd = _check_pandas_installed(strict=False)
        # use json.loads because this preserves ordering
        # (which is necessary for round-trip equivalence)
        metadata = json.loads(metadata, object_pairs_hook=OrderedDict)
        assert isinstance(metadata, list)
        if pd:
            metadata = pd.DataFrame.from_records(metadata)
            assert isinstance(metadata, pd.DataFrame)
    return metadata


class _FakeNoPandas(object):  # noqa: D101
    def __enter__(self):  # noqa: D105

        def _check(strict=True):
            if strict:
                raise RuntimeError('Pandas not installed')
            else:
                return False

        import mne
        self._old_check = _check_pandas_installed
        mne.epochs._check_pandas_installed = _check
        mne.utils.mixin._check_pandas_installed = _check

    def __exit__(self, *args):  # noqa: D105
        import mne
        mne.epochs._check_pandas_installed = self._old_check
        mne.utils.mixin._check_pandas_installed = self._old_check
