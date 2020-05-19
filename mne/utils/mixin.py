# -*- coding: utf-8 -*-
"""Some utility functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)


from copy import deepcopy
import logging
import json

import numpy as np

from .check import _check_pandas_installed, _check_preload, _validate_type
from ._logging import warn, verbose
from .numerics import object_size, object_hash


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

            1. ``epochs[idx]``: Return ``Epochs`` object with a subset of
               epochs (supports single index and python-style slicing).

            2. ``epochs['name']``: Return ``Epochs`` object with a copy of the
               subset of epochs corresponding to an experimental condition as
               specified by 'name'.

               If conditions are tagged by names separated by '/' (e.g.
               'audio/left', 'audio/right'), and 'name' is not in itself an
               event key, this selects every event whose condition contains
               the 'name' tag (e.g., 'left' matches 'audio/left' and
               'visual/left'; but not 'audio_left'). Note that tags selection
               is insensitive to order: tags like 'auditory/left' and
               'left/auditory' will be treated the same way when accessed.

            3. ``epochs[['name_1', 'name_2', ... ]]``: Return ``Epochs`` object
               with a copy of the subset of epochs corresponding to multiple
               experimental conditions as specified by
               ``'name_1', 'name_2', ...`` .

               If conditions are separated by '/', selects every item
               containing every list tag (e.g. ['audio', 'left'] selects
               'audio/left' and 'audio/center/left', but not 'audio/right').

            4. ``epochs['pandas query']``: Return ``Epochs`` object with a
               copy of the subset of epochs (and matching metadata) that match
               ``pandas query`` called with ``self.metadata.eval``, e.g.::

                   epochs["col_a > 2 and col_b == 'foo'"]

               This is only called if Pandas is installed and ``self.metadata``
               is a :class:`pandas.DataFrame`.

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
        del self._data
        inst = self.copy() if copy else self
        self._data = inst._data = data
        del self

        select = inst._item_to_select(item)
        has_selection = hasattr(inst, 'selection')
        if has_selection:
            key_selection = inst.selection[select]
            if reason is not None:
                for k in np.setdiff1d(inst.selection, key_selection):
                    inst.drop_log[k] = [reason]
            inst.selection = key_selection

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
        keys = keys if isinstance(keys, (list, tuple)) else [keys]
        try:
            # Assume it's a condition name
            return np.where(np.any(
                np.array([self.events[:, 2] == self.event_id[k]
                          for k in _hid_match(self.event_id, keys)]),
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
                    mask = self.metadata.eval(keys[0], engine='python').values
                except Exception as exp:
                    msg += (' The epochs.metadata Pandas query did not '
                            'yield any results: %s' % (exp.args[0],))
                else:
                    return np.where(mask)[0]
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
        if self.preload:
            if self._current >= len(self._data):
                raise StopIteration  # signal the end
            epoch = self._data[self._current]
            self._current += 1
        else:
            is_good = False
            while not is_good:
                if self._current >= len(self.events):
                    raise StopIteration  # signal the end properly
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
        metadata = json.loads(metadata, object_pairs_hook=dict)
        assert isinstance(metadata, list)
        if pd:
            metadata = pd.DataFrame.from_records(metadata)
            assert isinstance(metadata, pd.DataFrame)
    return metadata


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
    use_keys = []
    for key in keys:
        if not isinstance(key, str):
            raise KeyError('keys must be strings, got %s (%s)'
                           % (type(key), key))
        use_keys.extend(k for k in event_id.keys()
                        if set(key.split('/')).issubset(k.split('/')))
    if len(use_keys) == 0:
        raise KeyError('Event "{}" is not in Epochs. Event_ids must be one of '
                       '"{}"'.format(key, ', '.join(event_id.keys())))
    use_keys = list(set(use_keys))  # deduplicate if necessary
    return use_keys


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


class ShiftTimeMixin(object):
    """Class for shift_time method (Epochs, Evoked, and DipoleFixed)."""

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
        epochs : instance of Epochs
            The modified Epochs instance.

        Notes
        -----
        This method allows you to shift the *time* values associated with each
        data sample by an arbitrary amount. It does *not* resample the signal
        or change the *data* values in any way.
        """
        from ..epochs import BaseEpochs
        _check_preload(self, 'shift_time')
        start = tshift + (self.times[0] if relative else 0.)
        new_times = start + np.arange(len(self.times)) / self.info['sfreq']
        if isinstance(self, BaseEpochs):
            self._set_times(new_times)
        else:
            self.times = new_times
            self._update_first_last()
        return self

    def _update_first_last(self):
        """Update self.first and self.last (sample indices)."""
        self.first = int(round(self.times[0] * self.info['sfreq']))
        self.last = len(self.times) + self.first - 1
