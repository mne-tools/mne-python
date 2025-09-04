# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import re
import warnings
from collections import Counter, OrderedDict, UserDict, UserList
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from itertools import takewhile
from textwrap import shorten

import numpy as np
from scipy.io import loadmat

from ._fiff.constants import FIFF
from ._fiff.open import fiff_open
from ._fiff.tag import read_tag
from ._fiff.tree import dir_tree_find
from ._fiff.write import (
    _safe_name_list,
    end_block,
    start_and_end_file,
    start_block,
    write_double,
    write_float,
    write_name_list_sanitized,
    write_string,
)
from .utils import (
    _check_dict_keys,
    _check_dt,
    _check_fname,
    _check_option,
    _check_pandas_installed,
    _check_time_format,
    _convert_times,
    _DefaultEventParser,
    _dt_to_stamp,
    _is_numeric,
    _mask_to_onsets_offsets,
    _on_missing,
    _pl,
    _stamp_to_dt,
    _validate_type,
    check_fname,
    fill_doc,
    int_like,
    logger,
    verbose,
    warn,
)

# For testing windows_like_datetime, we monkeypatch "datetime" in this module.
# Keep the true datetime object around for _validate_type use.
_datetime = datetime


class _AnnotationsExtrasDict(UserDict):
    """A dictionary for storing extra fields of annotations.

    The keys of the dictionary are strings, and the values can be
    strings, integers, floats, or None.
    """

    def __setitem__(self, key: str, value: str | int | float | None) -> None:
        _validate_type(key, str, "key")
        if key in ("onset", "duration", "description", "ch_names"):
            raise ValueError(f"Key '{key}' is reserved and cannot be used in extras.")
        _validate_type(
            value,
            (str, int, float, None),
            "value",
        )
        super().__setitem__(key, value)


class _AnnotationsExtrasList(UserList):
    """A list of dictionaries for storing extra fields of annotations.

    Each dictionary in the list corresponds to an annotation and contains
    extra fields.
    The keys of the dictionaries are strings, and the values can be
    strings, integers, floats, or None.
    """

    def __repr__(self):
        return repr(self.data)

    @staticmethod
    def _validate_value(
        value: dict | _AnnotationsExtrasDict | None,
    ) -> _AnnotationsExtrasDict:
        _validate_type(
            value,
            (dict, _AnnotationsExtrasDict, None),
            "extras dict value",
            "dict or None",
        )
        return (
            value
            if isinstance(value, _AnnotationsExtrasDict)
            else _AnnotationsExtrasDict(value or {})
        )

    def __init__(self, initlist=None):
        if not (isinstance(initlist, _AnnotationsExtrasList) or initlist is None):
            initlist = [self._validate_value(v) for v in initlist]
        super().__init__(initlist)

    def __setitem__(  # type: ignore[override]
        self,
        key: int | slice,
        value,
    ) -> None:
        _validate_type(key, (int, slice), "key", "int or slice")
        if isinstance(key, int):
            iterable = False
            value = [value]
        else:
            _validate_type(value, Iterable, "value", "Iterable when key is a slice")
            iterable = True

        new_values = [self._validate_value(v) for v in value]
        if not iterable:
            new_values = new_values[0]
        super().__setitem__(key, new_values)

    def __iadd__(self, other):
        if not isinstance(other, _AnnotationsExtrasList):
            other = _AnnotationsExtrasList(other)
        super().__iadd__(other)

    def append(self, item):
        super().append(self._validate_value(item))

    def insert(self, i, item):
        super().insert(i, self._validate_value(item))

    def extend(self, other):
        if not isinstance(other, _AnnotationsExtrasList):
            other = _AnnotationsExtrasList(other)
        super().extend(other)


def _validate_extras(extras, length: int):
    _validate_type(extras, (None, list, _AnnotationsExtrasList), "extras")
    if extras is not None and len(extras) != length:
        raise ValueError(
            f"extras must be None or a list of length {length}, got {len(extras)}."
        )
    if isinstance(extras, _AnnotationsExtrasList):
        return extras
    return _AnnotationsExtrasList(extras or [None] * length)


def _check_o_d_s_c_e(onset, duration, description, ch_names, extras):
    onset = np.atleast_1d(np.array(onset, dtype=float))
    if onset.ndim != 1:
        raise ValueError(
            f"Onset must be a one dimensional array, got {onset.ndim} (shape "
            f"{onset.shape})."
        )
    duration = np.array(duration, dtype=float)
    if duration.ndim == 0 or duration.shape == (1,):
        duration = np.repeat(duration, len(onset))
    if duration.ndim != 1:
        raise ValueError(
            f"Duration must be a one dimensional array, got {duration.ndim}."
        )

    description = np.array(description, dtype=str)
    if description.ndim == 0 or description.shape == (1,):
        description = np.repeat(description, len(onset))
    if description.ndim != 1:
        raise ValueError(
            f"Description must be a one dimensional array, got {description.ndim}."
        )
    _safe_name_list(description, "write", "description")

    # ch_names: convert to ndarray of tuples
    _validate_type(ch_names, (None, tuple, list, np.ndarray), "ch_names")
    if ch_names is None:
        ch_names = [()] * len(onset)
    ch_names = list(ch_names)
    for ai, ch in enumerate(ch_names):
        _validate_type(ch, (list, tuple, np.ndarray), f"ch_names[{ai}]")
        ch_names[ai] = tuple(ch)
        for ci, name in enumerate(ch_names[ai]):
            _validate_type(name, str, f"ch_names[{ai}][{ci}]")
    ch_names = _ndarray_ch_names(ch_names)

    if not (len(onset) == len(duration) == len(description) == len(ch_names)):
        raise ValueError(
            "Onset, duration, description, and ch_names must be "
            f"equal in sizes, got {len(onset)}, {len(duration)}, "
            f"{len(description)}, and {len(ch_names)}."
        )

    extras = _validate_extras(extras, len(onset))
    return onset, duration, description, ch_names, extras


def _ndarray_ch_names(ch_names):
    # np.array(..., dtype=object) if all entries are empty will give
    # an empty array of shape (n_entries, 0) which is not helpful. So let's
    # force it to give us an array of shape (n_entries,) full of empty
    # tuples
    out = np.empty(len(ch_names), dtype=object)
    out[:] = ch_names
    return out


@fill_doc
class Annotations:
    """Annotation object for annotating segments of raw data.

    .. note::
       To convert events to `~mne.Annotations`, use
       `~mne.annotations_from_events`. To convert existing `~mne.Annotations`
       to events, use  `~mne.events_from_annotations`.

    Parameters
    ----------
    onset : array of float, shape (n_annotations,)
        The starting time of annotations in seconds after ``orig_time``.
    duration : array of float, shape (n_annotations,) | float
        Durations of the annotations in seconds. If a float, all the
        annotations are given the same duration.
    description : array of str, shape (n_annotations,) | str
        Array of strings containing description for each annotation. If a
        string, all the annotations are given the same description. To reject
        epochs, use description starting with keyword 'bad'. See example above.
    orig_time : float | str | datetime | tuple of int | None
        A POSIX Timestamp, datetime or a tuple containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisition is started at the
        same time. If it is a string, it should conform to the ISO8601 format.
        More precisely to this '%%Y-%%m-%%d %%H:%%M:%%S.%%f' particular case of
        the ISO8601 format where the delimiter between date and time is ' '.
    %(ch_names_annot)s

        .. versionadded:: 0.23
    extras : list[dict[str, int | float | str | None] | None] | None
        Optional list of dicts containing extra fields for each annotation.
        The number of items must match the number of annotations.

        .. versionadded:: 1.10

    See Also
    --------
    mne.annotations_from_events
    mne.events_from_annotations

    Notes
    -----
    Annotations are added to instance of :class:`mne.io.Raw` as the attribute
    :attr:`raw.annotations <mne.io.Raw.annotations>`.

    To reject bad epochs using annotations, use
    annotation description starting with 'bad' keyword. The epochs with
    overlapping bad segments are then rejected automatically by default.

    To remove epochs with blinks you can do:

    >>> eog_events = mne.preprocessing.find_eog_events(raw)  # doctest: +SKIP
    >>> n_blinks = len(eog_events)  # doctest: +SKIP
    >>> onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  # doctest: +SKIP
    >>> duration = np.repeat(0.5, n_blinks)  # doctest: +SKIP
    >>> description = ['bad blink'] * n_blinks  # doctest: +SKIP
    >>> annotations = mne.Annotations(onset, duration, description)  # doctest: +SKIP
    >>> raw.set_annotations(annotations)  # doctest: +SKIP
    >>> epochs = mne.Epochs(raw, events, event_id, tmin, tmax)  # doctest: +SKIP

    **ch_names**

    Specifying channel names allows the creation of channel-specific
    annotations. Once the annotations are assigned to a raw instance with
    :meth:`mne.io.Raw.set_annotations`, if channels are renamed by the raw
    instance, the annotation channels also get renamed. If channels are dropped
    from the raw instance, any channel-specific annotation that has no channels
    left in the raw instance will also be removed.

    **orig_time**

    If ``orig_time`` is None, the annotations are synced to the start of the
    data (0 seconds). Otherwise the annotations are synced to sample 0 and
    ``raw.first_samp`` is taken into account the same way as with events.

    When setting annotations, the following alignments
    between ``raw.info['meas_date']`` and ``annotation.orig_time`` take place:

    ::

        ----------- meas_date=XX, orig_time=YY -----------------------------

             |              +------------------+
             |______________|     RAW          |
             |              |                  |
             |              +------------------+
         meas_date      first_samp
             .
             .         |         +------+
             .         |_________| ANOT |
             .         |         |      |
             .         |         +------+
             .     orig_time   onset[0]
             .
             |                   +------+
             |___________________|      |
             |                   |      |
             |                   +------+
         orig_time            onset[0]'

        ----------- meas_date=XX, orig_time=None ---------------------------

             |              +------------------+
             |______________|     RAW          |
             |              |                  |
             |              +------------------+
             .              N         +------+
             .              o_________| ANOT |
             .              n         |      |
             .              e         +------+
             .
             |                        +------+
             |________________________|      |
             |                        |      |
             |                        +------+
         orig_time                 onset[0]'

        ----------- meas_date=None, orig_time=YY ---------------------------

             N              +------------------+
             o______________|     RAW          |
             n              |                  |
             e              +------------------+
                       |         +------+
                       |_________| ANOT |
                       |         |      |
                       |         +------+

                    [[[ CRASH ]]]

        ----------- meas_date=None, orig_time=None -------------------------

             N              +------------------+
             o______________|     RAW          |
             n              |                  |
             e              +------------------+
             .              N         +------+
             .              o_________| ANOT |
             .              n         |      |
             .              e         +------+
             .
             N                        +------+
             o________________________|      |
             n                        |      |
             e                        +------+
         orig_time                 onset[0]'

    .. warning::
       This means that when ``raw.info['meas_date'] is None``, doing
       ``raw.set_annotations(raw.annotations)`` will not alter ``raw`` if and
       only if ``raw.first_samp == 0``. When it's non-zero,
       ``raw.set_annotations`` will assume that the "new" annotations refer to
       the original data (with ``first_samp==0``), and will be re-referenced to
       the new time offset!

    **Specific annotation**

    ``BAD_ACQ_SKIP`` annotation leads to specific reading/writing file
    behaviours. See :meth:`mne.io.read_raw_fif` and
    :meth:`Raw.save() <mne.io.Raw.save>` notes for details.
    """  # noqa: E501

    def __init__(
        self,
        onset,
        duration,
        description,
        orig_time=None,
        ch_names=None,
        *,
        extras=None,
    ):
        self._orig_time = _handle_meas_date(orig_time)
        self.onset, self.duration, self.description, self.ch_names, self._extras = (
            _check_o_d_s_c_e(onset, duration, description, ch_names, extras)
        )
        self._sort()  # ensure we're sorted

    @property
    def orig_time(self):
        """The time base of the Annotations."""
        return self._orig_time

    @property
    def extras(self):
        """The extras of the Annotations.

        The ``extras`` attribute is a list of dictionaries.
        It can easily be converted to a pandas DataFrame using:
        ``pd.DataFrame(extras)``.
        """
        return self._extras

    @extras.setter
    def extras(self, extras):
        self._extras = _validate_extras(extras, len(self.onset))

    @property
    def _extras_columns(self) -> set[str]:
        """The set containing all the keys in all extras dicts."""
        return {k for d in self.extras for k in d}

    def __eq__(self, other):
        """Compare to another Annotations instance."""
        if not isinstance(other, Annotations):
            return False
        return (
            np.array_equal(self.onset, other.onset)
            and np.array_equal(self.duration, other.duration)
            and np.array_equal(self.description, other.description)
            and np.array_equal(self.ch_names, other.ch_names)
            and self.orig_time == other.orig_time
        )

    def __repr__(self):
        """Show the representation."""
        counter = Counter(self.description)
        kinds = ", ".join(["{} ({})".format(*k) for k in sorted(counter.items())])
        kinds = (": " if len(kinds) > 0 else "") + kinds
        ch_specific = ", channel-specific" if self._any_ch_names() else ""
        s = (
            f"Annotations | {len(self.onset)} segment"
            f"{_pl(len(self.onset))}{ch_specific}{kinds}"
        )
        return "<" + shorten(s, width=77, placeholder=" ...") + ">"

    def __len__(self):
        """Return the number of annotations.

        Returns
        -------
        n_annot : int
            The number of annotations.
        """
        return len(self.duration)

    def __add__(self, other):
        """Add (concatencate) two Annotation objects."""
        out = self.copy()
        out += other
        return out

    def __iadd__(self, other):
        """Add (concatencate) two Annotation objects in-place.

        Both annotations must have the same orig_time
        """
        if len(self) == 0:
            self._orig_time = other.orig_time
        if self.orig_time != other.orig_time:
            raise ValueError(
                "orig_time should be the same to add/concatenate 2 annotations (got "
                f"{self.orig_time} != {other.orig_time})"
            )
        return self.append(
            other.onset,
            other.duration,
            other.description,
            other.ch_names,
            extras=other.extras,
        )

    def __iter__(self):
        """Iterate over the annotations."""
        # Figure this out once ahead of time for consistency and speed (for
        # thousands of annotations)
        with_ch_names = self._any_ch_names()
        for idx in range(len(self.onset)):
            yield self.__getitem__(idx, with_ch_names=with_ch_names)

    def __getitem__(self, key, *, with_ch_names=None, with_extras=True):
        """Propagate indexing and slicing to the underlying numpy structure."""
        if isinstance(key, int_like):
            out_keys = ("onset", "duration", "description", "orig_time")
            out_vals = (
                self.onset[key],
                self.duration[key],
                self.description[key],
                self.orig_time,
            )
            if with_ch_names or (with_ch_names is None and self._any_ch_names()):
                out_keys += ("ch_names",)
                out_vals += (self.ch_names[key],)
            if with_extras:
                out_keys += ("extras",)
                out_vals += (self.extras[key],)
            return OrderedDict(zip(out_keys, out_vals))
        else:
            key = list(key) if isinstance(key, tuple) else key
            return Annotations(
                onset=self.onset[key],
                duration=self.duration[key],
                description=self.description[key],
                orig_time=self.orig_time,
                ch_names=self.ch_names[key],
                extras=[self.extras[i] for i in np.arange(len(self.extras))[key]],
            )

    @fill_doc
    def append(self, onset, duration, description, ch_names=None, *, extras=None):
        """Add an annotated segment. Operates inplace.

        Parameters
        ----------
        onset : float | array-like
            Annotation time onset from the beginning of the recording in
            seconds.
        duration : float | array-like
            Duration of the annotation in seconds.
        description : str | array-like
            Description for the annotation. To reject epochs, use description
            starting with keyword 'bad'.
        %(ch_names_annot)s

            .. versionadded:: 0.23
        extras : list[dict[str, int | float | str | None] | None] | None
            Optional list of dicts containing extras fields for each annotation.
            The number of items must match the number of annotations.

            .. versionadded:: 1.10

        Returns
        -------
        self : mne.Annotations
            The modified Annotations object.

        Notes
        -----
        The array-like support for arguments allows this to be used similarly
        to not only ``list.append``, but also
        `list.extend <https://docs.python.org/3/library/stdtypes.html#mutable-sequence-types>`__.
        """  # noqa: E501
        onset, duration, description, ch_names, extras = _check_o_d_s_c_e(
            onset, duration, description, ch_names, extras
        )
        self.onset = np.append(self.onset, onset)
        self.duration = np.append(self.duration, duration)
        self.description = np.append(self.description, description)
        self.ch_names = np.append(self.ch_names, ch_names)
        self.extras.extend(extras)
        self._sort()
        return self

    def copy(self):
        """Return a copy of the Annotations.

        Returns
        -------
        inst : instance of Annotations
            A copy of the object.
        """
        return deepcopy(self)

    def delete(self, idx):
        """Remove an annotation. Operates inplace.

        Parameters
        ----------
        idx : int | array-like of int
            Index of the annotation to remove. Can be array-like to
            remove multiple indices.
        """
        self.onset = np.delete(self.onset, idx)
        self.duration = np.delete(self.duration, idx)
        self.description = np.delete(self.description, idx)
        self.ch_names = np.delete(self.ch_names, idx)
        if isinstance(idx, int_like):
            del self.extras[idx]
        elif len(idx) > 0:
            # convert slice-like idx to ints, and delete list items in reverse order
            for i in np.sort(np.arange(len(self.extras))[idx])[::-1]:
                del self.extras[i]

    @fill_doc
    def to_data_frame(self, time_format="datetime"):
        """Export annotations in tabular structure as a pandas DataFrame.

        Parameters
        ----------
        %(time_format_df_raw)s
            Default is ``datetime``.

            .. versionadded:: 1.7

        Returns
        -------
        result : pandas.DataFrame
            Returns a pandas DataFrame with onset, duration, and
            description columns. A column named ch_names is added if any
            annotations are channel-specific.
        """
        pd = _check_pandas_installed(strict=True)
        valid_time_formats = ["ms", "timedelta", "datetime"]
        dt = _handle_meas_date(self.orig_time)
        if dt is None:
            dt = _handle_meas_date(0)
        time_format = _check_time_format(time_format, valid_time_formats, dt)
        dt = dt.replace(tzinfo=None)
        times = _convert_times(self.onset, time_format, dt)
        df = dict(onset=times, duration=self.duration, description=self.description)
        if self._any_ch_names():
            df.update(ch_names=self.ch_names)
        df = pd.DataFrame(df)
        extras_df = pd.DataFrame(self.extras)
        df = pd.concat([df, extras_df], axis=1)
        return df

    def count(self):
        """Count annotations.

        Returns
        -------
        counts : dict
            A dictionary containing unique annotation descriptions as keys with their
            counts as values.
        """
        return count_annotations(self)

    def _any_ch_names(self):
        return any(len(ch) for ch in self.ch_names)

    def _prune_ch_names(self, info, on_missing):
        # this prunes channel names and if a given channel-specific annotation
        # no longer has any channels left, it gets dropped
        keep = set(info["ch_names"])
        ch_names = self.ch_names
        warned = False
        drop_idx = list()
        for ci, ch in enumerate(ch_names):
            if len(ch):
                names = list()
                for name in ch:
                    if name not in keep:
                        if not warned:
                            _on_missing(
                                on_missing,
                                "At least one channel name in "
                                f"annotations missing from info: {name}",
                            )
                            warned = True
                    else:
                        names.append(name)
                ch_names[ci] = tuple(names)
                if not len(ch_names[ci]):
                    drop_idx.append(ci)
        if len(drop_idx):
            self.delete(drop_idx)
        return self

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save annotations to FIF, CSV or TXT.

        Typically annotations get saved in the FIF file for raw data
        (e.g., as ``raw.annotations``), but this offers the possibility
        to also save them to disk separately in different file formats
        which are easier to share between packages.

        Parameters
        ----------
        fname : path-like
            The filename to use.
        %(overwrite)s

            .. versionadded:: 0.23
        %(verbose)s

        Notes
        -----
        The format of the information stored in the saved annotation objects
        depends on the chosen file format. :file:`.csv` files store the onset
        as timestamps (e.g., ``2002-12-03 19:01:56.676071``),
        whereas :file:`.txt` files store onset as seconds since start of the
        recording (e.g., ``45.95597082905339``).
        """
        check_fname(
            fname,
            "annotations",
            (
                "-annot.fif",
                "-annot.fif.gz",
                "_annot.fif",
                "_annot.fif.gz",
                ".txt",
                ".csv",
            ),
        )
        fname = _check_fname(fname, overwrite=overwrite)
        if fname.suffix == ".txt":
            _write_annotations_txt(fname, self)
        elif fname.suffix == ".csv":
            _write_annotations_csv(fname, self)
        else:
            with start_and_end_file(fname) as fid:
                _write_annotations(fid, self)

    def _sort(self):
        """Sort in place."""
        # instead of argsort here we use sorted so that it gives us
        # the onset-then-duration hierarchy
        vals = sorted(zip(self.onset, self.duration, range(len(self))))
        order = list(list(zip(*vals))[-1]) if len(vals) else []
        self.onset = self.onset[order]
        self.duration = self.duration[order]
        self.description = self.description[order]
        self.ch_names = self.ch_names[order]
        self.extras = [self.extras[i] for i in order]

    @verbose
    def crop(
        self, tmin=None, tmax=None, emit_warning=False, use_orig_time=True, verbose=None
    ):
        """Remove all annotation that are outside of [tmin, tmax].

        The method operates inplace.

        Parameters
        ----------
        tmin : float | datetime | None
            Start time of selection in seconds.
        tmax : float | datetime | None
            End time of selection in seconds.
        emit_warning : bool
            Whether to emit warnings when limiting or omitting annotations.
            Defaults to False.
        use_orig_time : bool
            Whether to use orig_time as an offset.
            Defaults to True.
        %(verbose)s

        Returns
        -------
        self : instance of Annotations
            The cropped Annotations object.
        """
        if len(self) == 0:
            return self  # no annotations, nothing to do
        if not use_orig_time or self.orig_time is None:
            offset = _handle_meas_date(0)
        else:
            offset = self.orig_time
        if tmin is None:
            tmin = timedelta(seconds=self.onset.min()) + offset
        if tmax is None:
            tmax = timedelta(seconds=(self.onset + self.duration).max()) + offset
        for key, val in [("tmin", tmin), ("tmax", tmax)]:
            _validate_type(
                val, ("numeric", _datetime), key, "numeric, datetime, or None"
            )
        absolute_tmin = _handle_meas_date(tmin)
        absolute_tmax = _handle_meas_date(tmax)
        del tmin, tmax
        if absolute_tmin > absolute_tmax:
            raise ValueError(
                f"tmax should be greater than or equal to tmin ({absolute_tmin} < "
                f"{absolute_tmax})."
            )
        logger.debug(f"Cropping annotations {absolute_tmin} - {absolute_tmax}")

        onsets, durations, descriptions, ch_names, extras = [], [], [], [], []
        out_of_bounds, clip_left_elem, clip_right_elem = [], [], []
        for idx, (onset, duration, description, ch, extra) in enumerate(
            zip(self.onset, self.duration, self.description, self.ch_names, self.extras)
        ):
            # if duration is NaN behave like a zero
            if np.isnan(duration):
                duration = 0.0
            # convert to absolute times
            absolute_onset = timedelta(seconds=onset) + offset
            absolute_offset = absolute_onset + timedelta(seconds=duration)
            out_of_bounds.append(
                absolute_onset > absolute_tmax or absolute_offset < absolute_tmin
            )
            if out_of_bounds[-1]:
                clip_left_elem.append(False)
                clip_right_elem.append(False)
                logger.debug(
                    f"  [{idx}] Dropping "
                    f"({absolute_onset} - {absolute_offset}: {description})"
                )
            else:
                # clip the left side
                clip_left_elem.append(absolute_onset < absolute_tmin)
                if clip_left_elem[-1]:
                    absolute_onset = absolute_tmin
                clip_right_elem.append(absolute_offset > absolute_tmax)
                if clip_right_elem[-1]:
                    absolute_offset = absolute_tmax
                if clip_left_elem[-1] or clip_right_elem[-1]:
                    durations.append((absolute_offset - absolute_onset).total_seconds())
                else:
                    durations.append(duration)
                onsets.append((absolute_onset - offset).total_seconds())
                logger.debug(
                    f"  [{idx}] Keeping  "
                    f"({absolute_onset} - {absolute_offset} -> "
                    f"{onset} - {onset + duration})"
                )
                descriptions.append(description)
                ch_names.append(ch)
                extras.append(extra)
        logger.debug(f"Cropping complete (kept {len(onsets)})")
        self.onset = np.array(onsets, float)
        self.duration = np.array(durations, float)
        assert (self.duration >= 0).all()
        self.description = np.array(descriptions, dtype=str)
        self.ch_names = _ndarray_ch_names(ch_names)
        self.extras = extras

        if emit_warning:
            omitted = np.array(out_of_bounds).sum()
            if omitted > 0:
                warn(f"Omitted {omitted} annotation(s) that were outside data range.")
            limited = (np.array(clip_left_elem) | np.array(clip_right_elem)).sum()
            if limited > 0:
                warn(
                    f"Limited {limited} annotation(s) that were expanding outside the"
                    " data range."
                )

        return self

    @verbose
    def set_durations(self, mapping, verbose=None):
        """Set annotation duration(s). Operates inplace.

        Parameters
        ----------
        mapping : dict | float
            A dictionary mapping the annotation description to a duration in
            seconds e.g. ``{'ShortStimulus' : 3, 'LongStimulus' : 12}``.
            Alternatively, if a number is provided, then all annotations
            durations are set to the single provided value.
        %(verbose)s

        Returns
        -------
        self : mne.Annotations
            The modified Annotations object.

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        _validate_type(mapping, (int, float, dict))

        if isinstance(mapping, dict):
            _check_dict_keys(
                mapping,
                self.description,
                valid_key_source="data",
                key_description="Annotation description(s)",
            )
            for stim in mapping:
                map_idx = [desc == stim for desc in self.description]
                self.duration[map_idx] = mapping[stim]

        elif _is_numeric(mapping):
            self.duration = np.ones(self.description.shape) * mapping

        else:
            raise ValueError(
                "Setting durations requires the mapping of "
                "descriptions to times to be provided as a dict. "
                f"Instead {type(mapping)} was provided."
            )

        return self

    @verbose
    def rename(self, mapping, verbose=None):
        """Rename annotation description(s). Operates inplace.

        Parameters
        ----------
        mapping : dict
            A dictionary mapping the old description to a new description,
            e.g. {'1.0' : 'Control', '2.0' : 'Stimulus'}.
        %(verbose)s

        Returns
        -------
        self : mne.Annotations
            The modified Annotations object.

        Notes
        -----
        .. versionadded:: 0.24.0
        """
        _validate_type(mapping, dict)
        _check_dict_keys(
            mapping,
            self.description,
            valid_key_source="data",
            key_description="Annotation description(s)",
        )
        self.description = np.array([str(mapping.get(d, d)) for d in self.description])
        return self


class EpochAnnotationsMixin:
    """Mixin class for Annotations in Epochs."""

    @property
    def annotations(self):  # noqa: D102
        return self._annotations

    @verbose
    def set_annotations(self, annotations, on_missing="raise", *, verbose=None):
        """Setter for Epoch annotations from Raw.

        This method does not handle offsetting the times based
        on first_samp or measurement dates, since that is expected
        to occur in Raw.set_annotations().

        Parameters
        ----------
        annotations : instance of mne.Annotations | None
            Annotations to set.
        %(on_missing_ch_names)s
        %(verbose)s

        Returns
        -------
        self : instance of Epochs
            The epochs object with annotations.

        Notes
        -----
        Annotation onsets and offsets are stored as time in seconds (not as
        sample numbers).

        If you have an ``-epo.fif`` file saved to disk created before 1.0,
        annotations can be added correctly only if no decimation or
        resampling was performed. We thus suggest to regenerate your
        :class:`mne.Epochs` from raw and re-save to disk with 1.0+ if you
        want to safely work with :class:`~mne.Annotations` in epochs.

        Since this method does not handle offsetting the times based
        on first_samp or measurement dates, the recommended way to add
        Annotations is::

            raw.set_annotations(annotations)
            annotations = raw.annotations
            epochs.set_annotations(annotations)

        .. versionadded:: 1.0
        """
        _validate_type(annotations, (Annotations, None), "annotations")
        if annotations is None:
            self._annotations = None
        else:
            if getattr(self, "_unsafe_annot_add", False):
                warn(
                    "Adding annotations to Epochs created (and saved to disk) before "
                    "1.0 will yield incorrect results if decimation or resampling was "
                    "performed on the instance, we recommend regenerating the Epochs "
                    "and re-saving them to disk."
                )
            new_annotations = annotations.copy()
            new_annotations._prune_ch_names(self.info, on_missing)
            self._annotations = new_annotations
        return self

    def get_annotations_per_epoch(self, *, with_extras=False):
        """Get a list of annotations that occur during each epoch.

        Parameters
        ----------
        with_extras : bool
            Whether to include the annotations extra fields in the output,
            as an additional last element of the tuple. Default is False.

            .. versionadded:: 1.10

        Returns
        -------
        epoch_annots : list
            A list of lists (with length equal to number of epochs) where each
            inner list contains any annotations that overlap the corresponding
            epoch. Annotations are stored as a :class:`tuple` of onset,
            duration, description (not as a :class:`~mne.Annotations` object),
            where the onset is now relative to time=0 of the epoch, rather than
            time=0 of the original continuous (raw) data.
        """
        # create a list of annotations for each epoch
        epoch_annot_list = [[] for _ in range(len(self.events))]

        # check if annotations exist
        if self.annotations is None:
            return epoch_annot_list

        # when each epoch and annotation starts/stops
        # no need to account for first_samp here...
        epoch_tzeros = self.events[:, 0] / self._raw_sfreq
        epoch_starts, epoch_stops = (
            np.atleast_2d(epoch_tzeros) + np.atleast_2d(self.times[[0, -1]]).T
        )
        # ... because first_samp isn't accounted for here either
        annot_starts = self._annotations.onset
        annot_stops = annot_starts + self._annotations.duration

        # the first two cases (annot_straddles_epoch_{start|end}) will both
        # (redundantly) capture cases where an annotation fully encompasses
        # an epoch (e.g., annot from 1-4s, epoch from 2-3s). The redundancy
        # doesn't matter because results are summed and then cast to bool (all
        # we care about is presence/absence of overlap).
        annot_straddles_epoch_start = np.logical_and(
            np.atleast_2d(epoch_starts) >= np.atleast_2d(annot_starts).T,
            np.atleast_2d(epoch_starts) < np.atleast_2d(annot_stops).T,
        )

        annot_straddles_epoch_end = np.logical_and(
            np.atleast_2d(epoch_stops) > np.atleast_2d(annot_starts).T,
            np.atleast_2d(epoch_stops) <= np.atleast_2d(annot_stops).T,
        )

        # this captures the only remaining case we care about: annotations
        # fully contained within an epoch (or exactly coextensive with it).
        annot_fully_within_epoch = np.logical_and(
            np.atleast_2d(epoch_starts) <= np.atleast_2d(annot_starts).T,
            np.atleast_2d(epoch_stops) >= np.atleast_2d(annot_stops).T,
        )

        # combine all cases to get array of shape (n_annotations, n_epochs).
        # Nonzero entries indicate overlap between the corresponding
        # annotation (row index) and epoch (column index).
        all_cases = (
            annot_straddles_epoch_start
            + annot_straddles_epoch_end
            + annot_fully_within_epoch
        )

        # for each Epoch-Annotation overlap occurrence:
        for annot_ix, epo_ix in zip(*np.nonzero(all_cases)):
            this_annot = self._annotations[annot_ix]
            this_tzero = epoch_tzeros[epo_ix]
            # adjust annotation onset to be relative to epoch tzero...
            annot = (
                this_annot["onset"] - this_tzero,
                this_annot["duration"],
                this_annot["description"],
            )
            if with_extras:
                annot += (this_annot["extras"],)
            # ...then add it to the correct sublist of `epoch_annot_list`
            epoch_annot_list[epo_ix].append(annot)
        return epoch_annot_list

    def add_annotations_to_metadata(self, overwrite=False, *, with_extras=True):
        """Add raw annotations into the Epochs metadata data frame.

        Adds three columns to the ``metadata`` consisting of a list
        in each row:
        - ``annot_onset``: the onset of each Annotation within
        the Epoch relative to the start time of the Epoch (in seconds).
        - ``annot_duration``: the duration of each Annotation
        within the Epoch in seconds.
        - ``annot_description``: the free-form text description of each
        Annotation.

        Parameters
        ----------
        overwrite : bool
            Whether to overwrite existing columns in metadata or not.
            Default is False.
        with_extras : bool
            Whether to include the annotations extra fields in the output,
            as an additional last element of the tuple. Default is True.

            .. versionadded:: 1.10

        Returns
        -------
        self : instance of Epochs
            The modified instance (instance is also modified inplace).

        Notes
        -----
        .. versionadded:: 1.0
        """
        pd = _check_pandas_installed()

        # check if annotations exist
        if self.annotations is None:
            warn(
                f"There were no Annotations stored in {self}, so "
                "metadata was not modified."
            )
            return self

        # get existing metadata DataFrame or instantiate an empty one
        if self._metadata is not None:
            metadata = self._metadata
        else:
            data = np.empty((len(self.events), 0))
            metadata = pd.DataFrame(data=data)

        if (
            any(
                name in metadata.columns
                for name in ["annot_onset", "annot_duration", "annot_description"]
            )
            and not overwrite
        ):
            raise RuntimeError(
                "Metadata for Epochs already contains columns "
                '"annot_onset", "annot_duration", or "annot_description".'
            )

        # get the Epoch annotations, then convert to separate lists for
        # onsets, durations, and descriptions
        epoch_annot_list = self.get_annotations_per_epoch(with_extras=with_extras)
        onset, duration, description = [], [], []
        extras = {k: [] for k in self.annotations._extras_columns}
        for epoch_annot in epoch_annot_list:
            for ix, annot_prop in enumerate((onset, duration, description)):
                entry = [annot[ix] for annot in epoch_annot]

                # round onset and duration to avoid IO round trip mismatch
                if ix < 2:
                    entry = np.round(entry, decimals=12).tolist()

                annot_prop.append(entry)
            for k in extras.keys():
                entry = [annot[3].get(k, None) for annot in epoch_annot]
                extras[k].append(entry)

        # Create a new Annotations column that is instantiated as an empty
        # list per Epoch.
        metadata["annot_onset"] = pd.Series(onset)
        metadata["annot_duration"] = pd.Series(duration)
        metadata["annot_description"] = pd.Series(description)
        for k, v in extras.items():
            metadata[f"annot_{k}"] = pd.Series(v)

        # reset the metadata
        self.metadata = metadata
        return self


def _combine_annotations(
    one, two, one_n_samples, one_first_samp, two_first_samp, sfreq
):
    """Combine a tuple of annotations."""
    assert one is not None
    assert two is not None
    shift = one_n_samples / sfreq  # to the right by the number of samples
    shift += one_first_samp / sfreq  # to the right by the offset
    shift -= two_first_samp / sfreq  # undo its offset
    onset = np.concatenate([one.onset, two.onset + shift])
    duration = np.concatenate([one.duration, two.duration])
    description = np.concatenate([one.description, two.description])
    ch_names = np.concatenate([one.ch_names, two.ch_names])
    return Annotations(onset, duration, description, one.orig_time, ch_names)


def _handle_meas_date(meas_date):
    """Convert meas_date to datetime or None.

    If `meas_date` is a string, it should conform to the ISO8601 format.
    More precisely to this '%Y-%m-%d %H:%M:%S.%f' particular case of the
    ISO8601 format where the delimiter between date and time is ' '.
    Note that ISO8601 allows for ' ' or 'T' as delimiters between date and
    time.
    """
    if isinstance(meas_date, str):
        ACCEPTED_ISO8601 = "%Y-%m-%d %H:%M:%S.%f"
        try:
            meas_date = datetime.strptime(meas_date, ACCEPTED_ISO8601)
        except ValueError:
            meas_date = None
        else:
            meas_date = meas_date.replace(tzinfo=timezone.utc)
    elif isinstance(meas_date, tuple):
        # old way
        meas_date = _stamp_to_dt(meas_date)
    if meas_date is not None:
        if np.isscalar(meas_date):
            # It would be nice just to do:
            #
            #     meas_date = datetime.fromtimestamp(meas_date, timezone.utc)
            #
            # But Windows does not like timestamps < 0. So we'll use
            # our specialized wrapper instead:
            meas_date = np.array(np.modf(meas_date)[::-1])
            meas_date *= [1, 1e6]
            meas_date = _stamp_to_dt(np.round(meas_date))
        _check_dt(meas_date)  # run checks
    return meas_date


def _sync_onset(raw, onset, inverse=False):
    """Adjust onsets in relation to raw data."""
    offset = (-1 if inverse else 1) * raw._first_time
    assert raw.info["meas_date"] == raw.annotations.orig_time
    annot_start = onset - offset
    return annot_start


def _annotations_starts_stops(raw, kinds, name="skip_by_annotation", invert=False):
    """Get starts and stops from given kinds.

    onsets and ends are inclusive.
    """
    _validate_type(kinds, (str, list, tuple), name)
    if isinstance(kinds, str):
        kinds = [kinds]
    else:
        for kind in kinds:
            _validate_type(kind, "str", "All entries")

    if len(raw.annotations) == 0:
        onsets, ends = np.array([], int), np.array([], int)
    else:
        idxs = [
            idx
            for idx, desc in enumerate(raw.annotations.description)
            if any(desc.upper().startswith(kind.upper()) for kind in kinds)
        ]
        # onsets are already sorted
        onsets = raw.annotations.onset[idxs]
        onsets = _sync_onset(raw, onsets)
        ends = onsets + raw.annotations.duration[idxs]
        onsets = raw.time_as_index(onsets, use_rounding=True)
        ends = raw.time_as_index(ends, use_rounding=True)
    assert (onsets <= ends).all()  # all durations >= 0
    if invert:
        # We need to eliminate overlaps here, otherwise wacky things happen,
        # so we carefully invert the relationship
        mask = np.zeros(len(raw.times), bool)
        for onset, end in zip(onsets, ends):
            mask[onset:end] = True
        mask = ~mask
        extras = onsets == ends
        extra_onsets, extra_ends = onsets[extras], ends[extras]
        onsets, ends = _mask_to_onsets_offsets(mask)
        # Keep ones where things were exactly equal
        del extras
        # we could do this with a np.insert+np.searchsorted, but our
        # ordered-ness should get us it for free
        onsets = np.sort(np.concatenate([onsets, extra_onsets]))
        ends = np.sort(np.concatenate([ends, extra_ends]))
        assert (onsets <= ends).all()
    return onsets, ends


def _write_annotations(fid, annotations):
    """Write annotations."""
    start_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)
    write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, annotations.onset)
    write_float(
        fid, FIFF.FIFF_MNE_BASELINE_MAX, annotations.duration + annotations.onset
    )
    write_name_list_sanitized(
        fid, FIFF.FIFF_COMMENT, annotations.description, name="description"
    )
    if annotations.orig_time is not None:
        write_double(fid, FIFF.FIFF_MEAS_DATE, _dt_to_stamp(annotations.orig_time))
    if annotations._any_ch_names():
        write_string(
            fid, FIFF.FIFF_MNE_EPOCHS_DROP_LOG, json.dumps(tuple(annotations.ch_names))
        )
    if any(d is not None for d in annotations.extras):
        write_string(
            fid,
            FIFF.FIFF_FREE_LIST,
            json.dumps([extra.data for extra in annotations.extras]),
        )
    end_block(fid, FIFF.FIFFB_MNE_ANNOTATIONS)


def _write_annotations_csv(fname, annot):
    annot = annot.to_data_frame()
    if "ch_names" in annot:
        annot["ch_names"] = [
            _safe_name_list(ch, "write", name=f'annot["ch_names"][{ci}')
            for ci, ch in enumerate(annot["ch_names"])
        ]
    extras_columns = set(annot.columns) - {
        "onset",
        "duration",
        "description",
        "ch_names",
    }
    for col in extras_columns:
        if len(dtypes := annot[col].apply(type).unique()) > 1:
            warn(
                f"Extra field '{col}' contains heterogeneous dtypes ({dtypes}). "
                "Loading these CSV annotations may not return the original dtypes."
            )
    annot.to_csv(fname, index=False)


def _write_annotations_txt(fname, annot):
    content = "# MNE-Annotations\n"
    if annot.orig_time is not None:
        # for backward compat, we do not write tzinfo (assumed UTC)
        content += f"# orig_time : {annot.orig_time.replace(tzinfo=None)}\n"
    content += "# onset, duration, description"
    n_cols = 3
    data = [annot.onset, annot.duration, annot.description]
    if annot._any_ch_names():
        n_cols += 1
        content += ", ch_names"
        data.append(
            [
                _safe_name_list(ch, "write", f"annot.ch_names[{ci}]")
                for ci, ch in enumerate(annot.ch_names)
            ]
        )
    if len(extras_columns := annot._extras_columns) > 0:
        n_cols += len(extras_columns)
        for column in extras_columns:
            content += f", {column}"
            values = [extra.get(column, None) for extra in annot.extras]
            if len(dtypes := set(type(v) for v in values)) > 1:
                warn(
                    f"Extra field '{column}' contains heterogeneous dtypes ({dtypes}). "
                    "Loading these TXT annotations may not return the original dtypes."
                )
            data.append([val if val is not None else "" for val in values])
    content += "\n"
    data = np.array(data, dtype=str).T
    assert data.ndim == 2
    assert data.shape[0] == len(annot.onset)
    assert data.shape[1] == n_cols
    with open(fname, "wb") as fid:
        fid.write(content.encode())
        np.savetxt(fid, data, delimiter=",", fmt="%s")


@fill_doc
def read_annotations(
    fname, sfreq="auto", uint16_codec=None, encoding="utf8", ignore_marker_types=False
) -> Annotations:
    r"""Read annotations from a file.

    This function reads a ``.fif``, ``.fif.gz``, ``.vmrk``, ``.amrk``,
    ``.edf``, ``.bdf``, ``.gdf``, ``.txt``, ``.csv``, ``.cnt``, ``.cef``, or
    ``.set`` file and makes an :class:`mne.Annotations` object.

    Parameters
    ----------
    fname : path-like
        The filename.
    sfreq : float | ``'auto'``
        The sampling frequency in the file. This parameter is necessary for
        \*.vmrk, \*.amrk, and \*.cef files as Annotations are expressed in
        seconds and \*.vmrk/\*.amrk/\*.cef files are in samples. For any other
        file format, ``sfreq`` is omitted. If set to 'auto' then the ``sfreq``
        is taken from the respective info file of the same name with according
        file extension (\*.vhdr/\*.ahdr for brainvision; \*.dap for Curry 7;
        \*.cdt.dpa for Curry 8). So data.vmrk/amrk looks for sfreq in
        data.vhdr/ahdr, data.cef looks in data.dap and data.cdt.cef looks in
        data.cdt.dpa.
    uint16_codec : str | None
        This parameter is only used in EEGLAB (\*.set) and omitted otherwise.
        If your \*.set file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        ``'latin1'`` or ``'utf-8'``) should be used when reading character
        arrays and can therefore help you solve this problem.
    %(encoding_edf)s
        Only used when reading EDF annotations.
    ignore_marker_types : bool
        If ``True``, ignore marker types in BrainVision files (and only use their
        descriptions). Defaults to ``False``.

    Returns
    -------
    annot : instance of Annotations
        The annotations.

    Notes
    -----
    The annotations stored in a ``.csv`` require the onset columns to be
    timestamps. If you have onsets as floats (in seconds), you should use the
    ``.txt`` extension.
    """
    from .io.brainvision.brainvision import _read_annotations_brainvision
    from .io.cnt.cnt import _read_annotations_cnt
    from .io.ctf.markers import _read_annotations_ctf
    from .io.curry.curry import _read_annotations_curry
    from .io.edf.edf import _read_annotations_edf
    from .io.eeglab.eeglab import _read_annotations_eeglab

    fname = _check_fname(
        fname,
        overwrite="read",
        must_exist=True,
        need_dir=str(fname).endswith(".ds"),  # for CTF
        name="fname",
    )
    readers = {
        ".csv": _read_annotations_csv,
        ".cnt": _read_annotations_cnt,
        ".ds": _read_annotations_ctf,
        ".cef": _read_annotations_curry,
        ".set": _read_annotations_eeglab,
        ".edf": _read_annotations_edf,
        ".bdf": _read_annotations_edf,
        ".gdf": _read_annotations_edf,
        ".vmrk": _read_annotations_brainvision,
        ".amrk": _read_annotations_brainvision,
        ".txt": _read_annotations_txt,
    }
    kwargs = {
        ".vmrk": {"sfreq": sfreq, "ignore_marker_types": ignore_marker_types},
        ".amrk": {"sfreq": sfreq, "ignore_marker_types": ignore_marker_types},
        ".cef": {"sfreq": sfreq},
        ".set": {"uint16_codec": uint16_codec},
        ".edf": {"encoding": encoding},
        ".bdf": {"encoding": encoding},
        ".gdf": {"encoding": encoding},
    }
    if fname.suffix in readers:
        annotations = readers[fname.suffix](fname, **kwargs.get(fname.suffix, {}))
    elif fname.name.endswith(("fif", "fif.gz")):
        # Read FiF files
        ff, tree, _ = fiff_open(fname, preload=False)
        with ff as fid:
            annotations = _read_annotations_fif(fid, tree)
    elif fname.name.startswith("events_") and fname.suffix == ".mat":
        annotations = _read_brainstorm_annotations(fname)
    else:
        raise OSError(f'Unknown annotation file format "{fname}"')

    if annotations is None:
        raise OSError(f'No annotation data found in file "{fname}"')
    return annotations


def _read_annotations_csv(fname):
    """Read annotations from csv.

    The dtypes of the extra fields will automatically be inferred
    by pandas. If some fields have heterogeneous types on the
    different rows, this automatic inference may return unexpected
    types.
    If you need to save heterogeneous extra dtypes, we recommend
    saving to FIF.

    Parameters
    ----------
    fname : path-like
        The filename.

    Returns
    -------
    annot : instance of Annotations
        The annotations.
    """
    pd = _check_pandas_installed(strict=True)
    df = pd.read_csv(fname, keep_default_na=False)
    orig_time = df["onset"].values[0]
    try:
        float(orig_time)
        warn(
            "It looks like you have provided annotation onsets as floats. "
            "These will be interpreted as MILLISECONDS. If that is not what "
            "you want, save your CSV as a TXT file; the TXT reader accepts "
            "onsets in seconds."
        )
    except ValueError:
        pass
    onset_dt = pd.to_datetime(df["onset"])
    onset = (onset_dt - onset_dt[0]).dt.total_seconds()
    duration = df["duration"].values.astype(float)
    description = df["description"].values
    ch_names = None
    if "ch_names" in df.columns:
        ch_names = [
            _safe_name_list(val, "read", "annotation channel name")
            for val in df["ch_names"].values
        ]
    extra_columns = list(
        df.columns.difference(["onset", "duration", "description", "ch_names"])
    )
    extras = None
    if len(extra_columns) > 0:
        extras = df[extra_columns].to_dict(orient="records")
    return Annotations(onset, duration, description, orig_time, ch_names, extras=extras)


def _read_brainstorm_annotations(fname, orig_time=None):
    """Read annotations from a Brainstorm events_ file.

    Parameters
    ----------
    fname : path-like
        The filename
    orig_time : float | int | instance of datetime | array of int | None
        A POSIX Timestamp, datetime or an array containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisition is started at the
        same time.

    Returns
    -------
    annot : instance of Annotations | None
        The annotations.
    """

    def get_duration_from_times(t):
        return t[1] - t[0] if t.shape[0] == 2 else np.zeros(len(t[0]))

    annot_data = loadmat(fname)
    onsets, durations, descriptions = (list(), list(), list())
    for label, _, _, _, times, _, _ in annot_data["events"][0]:
        onsets.append(times[0])
        durations.append(get_duration_from_times(times))
        n_annot = len(times[0])
        descriptions += [str(label[0])] * n_annot

    return Annotations(
        onset=np.concatenate(onsets),
        duration=np.concatenate(durations),
        description=descriptions,
        orig_time=orig_time,
    )


def _is_iso8601(candidate_str):
    ISO8601 = r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\.\d{6}$"
    return re.compile(ISO8601).match(candidate_str) is not None


def _read_annotations_txt_parse_header(fname):
    def is_orig_time(x):
        return x.startswith("# orig_time :")

    def is_columns(x):
        return x.startswith("# onset, duration, description")

    with open(fname) as fid:
        header = list(takewhile(lambda x: x.startswith("#"), fid))

    orig_values = [h[13:].strip() for h in header if is_orig_time(h)]
    orig_values = [_handle_meas_date(orig) for orig in orig_values if _is_iso8601(orig)]

    columns = [[c.strip() for c in h[2:].split(",")] for h in header if is_columns(h)]

    return (
        None if not orig_values else orig_values[0],
        (None if not columns else columns[0]),
        len(header),
    )


def _read_annotations_txt(fname):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        out = np.loadtxt(fname, delimiter=",", dtype=np.bytes_, unpack=True)
    orig_time, columns, n_rows_header = _read_annotations_txt_parse_header(fname)
    ch_names = extras = None
    if len(out) == 0:
        onset, duration, desc = [], [], []
    else:
        if columns is None:
            # No column names were present in the header
            # We assume the first three columns are onset, duration, description
            # And eventually a fourth column with ch_names
            _check_option("text header", len(out), (3, 4))
            columns = ["onset", "duration", "description"] + (
                ["ch_names"] if len(out) == 4 else []
            )
        col_map = {col: i for i, col in enumerate(columns)}
        if len(col_map) != len(columns):
            raise ValueError(
                "Duplicate column names found in header. Please check the file format."
            )
        if missing := {"onset", "duration", "description"} - set(col_map.keys()):
            raise ValueError(
                f"Column(s) {missing} not found in header. "
                "Please check the file format."
            )
        _check_option("text header len", len(out), (len(columns),))
        onset = out[col_map["onset"]]
        duration = out[col_map["duration"]]
        desc = out[col_map["description"]]
        if "ch_names" in col_map:
            ch_names = out[col_map["ch_names"]]
        extra_columns = set(col_map.keys()) - {
            "onset",
            "duration",
            "description",
            "ch_names",
        }
        if extra_columns:
            pd = _check_pandas_installed(strict=False)
            if pd:
                df = pd.read_csv(
                    fname,
                    delimiter=",",
                    names=columns,
                    usecols=extra_columns,
                    skiprows=n_rows_header,
                    header=None,
                    keep_default_na=False,
                )
                extras = df.to_dict(orient="records")
            else:
                warn(
                    "Extra fields found in the header but pandas is not installed. "
                    "Therefore the dtypes of the extra fields can not automatically "
                    "be inferred so they will be loaded as strings."
                )
                extras = [
                    {
                        col_name: out[col_map[col_name]][i].decode("UTF-8")
                        for col_name in extra_columns
                    }
                    for i in range(len(onset))
                ]

    onset = [float(o.decode()) for o in np.atleast_1d(onset)]
    duration = [float(d.decode()) for d in np.atleast_1d(duration)]
    desc = [str(d.decode()).strip() for d in np.atleast_1d(desc)]
    if ch_names is not None:
        ch_names = [
            _safe_name_list(ch.decode().strip(), "read", f"ch_names[{ci}]")
            for ci, ch in enumerate(ch_names)
        ]

    annotations = Annotations(
        onset=onset,
        duration=duration,
        description=desc,
        orig_time=orig_time,
        ch_names=ch_names,
        extras=extras,
    )

    return annotations


def _read_annotations_fif(fid, tree):
    """Read annotations."""
    annot_data = dir_tree_find(tree, FIFF.FIFFB_MNE_ANNOTATIONS)
    if len(annot_data) == 0:
        annotations = None
    else:
        annot_data = annot_data[0]
        orig_time = ch_names = extras = None
        onset, duration, description = list(), list(), list()
        for ent in annot_data["directory"]:
            kind = ent.kind
            pos = ent.pos
            tag = read_tag(fid, pos)
            if kind == FIFF.FIFF_MNE_BASELINE_MIN:
                onset = tag.data
                onset = list() if onset is None else onset
            elif kind == FIFF.FIFF_MNE_BASELINE_MAX:
                duration = tag.data
                duration = list() if duration is None else duration - onset
            elif kind == FIFF.FIFF_COMMENT:
                description = _safe_name_list(tag.data, "read", "description")
            elif kind == FIFF.FIFF_MEAS_DATE:
                orig_time = tag.data
                try:
                    orig_time = float(orig_time)  # old way
                except TypeError:
                    orig_time = tuple(orig_time)  # new way
            elif kind == FIFF.FIFF_MNE_EPOCHS_DROP_LOG:
                ch_names = tuple(tuple(x) for x in json.loads(tag.data))
            elif kind == FIFF.FIFF_FREE_LIST:
                extras = json.loads(tag.data)
        assert len(onset) == len(duration) == len(description)
        if extras is not None:
            assert len(extras) == len(onset)
        annotations = Annotations(
            onset, duration, description, orig_time, ch_names, extras=extras
        )
    return annotations


def _select_annotations_based_on_description(descriptions, event_id, regexp):
    """Get a collection of descriptions and returns index of selected."""
    regexp_comp = re.compile(".*" if regexp is None else regexp)

    event_id_ = dict()
    dropped = []
    # Iterate over the sorted descriptions so that the Counter mapping
    # is slightly less arbitrary
    for desc in sorted(descriptions):
        if desc in event_id_:
            continue

        if regexp_comp.match(desc) is None:
            continue

        if isinstance(event_id, dict):
            if desc in event_id:
                event_id_[desc] = event_id[desc]
            else:
                continue
        else:
            trigger = event_id(desc)
            if trigger is not None:
                event_id_[desc] = trigger
            else:
                dropped.append(desc)

    event_sel = [ii for ii, kk in enumerate(descriptions) if kk in event_id_]

    if len(event_sel) == 0 and regexp is not None:
        raise ValueError("Could not find any of the events you specified.")

    return event_sel, event_id_


def _select_events_based_on_id(events, event_desc):
    """Get a collection of events and returns index of selected."""
    event_desc_ = dict()
    func = event_desc.get if isinstance(event_desc, dict) else event_desc
    event_ids = events[np.unique(events[:, 2], return_index=True)[1], 2]
    for e in event_ids:
        trigger = func(e)
        if trigger is not None:
            event_desc_[e] = trigger

    event_sel = [ii for ii, e in enumerate(events) if e[2] in event_desc_]

    if len(event_sel) == 0:
        raise ValueError("Could not find any of the events you specified.")

    return event_sel, event_desc_


def _check_event_id(event_id, raw):
    from .io import Raw, RawArray
    from .io.brainvision.brainvision import (
        RawBrainVision,
        _BVEventParser,
        _check_bv_annot,
    )

    if event_id is None:
        return _DefaultEventParser()
    elif event_id == "auto":
        if isinstance(raw, RawBrainVision):
            return _BVEventParser()
        elif isinstance(raw, Raw | RawArray) and _check_bv_annot(
            raw.annotations.description
        ):
            logger.info("Non-RawBrainVision raw using branvision markers")
            return _BVEventParser()
        else:
            return _DefaultEventParser()
    elif callable(event_id) or isinstance(event_id, dict):
        return event_id
    else:
        raise ValueError(
            "Invalid type for event_id (should be None, str, "
            f"dict or callable). Got {type(event_id)}."
        )


def _check_event_description(event_desc, events):
    """Check event_id and convert to default format."""
    if event_desc is None:  # convert to int to make typing-checks happy
        event_desc = list(np.unique(events[:, 2]))

    if isinstance(event_desc, dict):
        for val in event_desc.values():
            _validate_type(val, (str, None), "Event names")
    elif isinstance(event_desc, Iterable):
        event_desc = np.asarray(event_desc)
        if event_desc.ndim != 1:
            raise ValueError(f"event_desc must be 1D, got shape {event_desc.shape}")
        event_desc = dict(zip(event_desc, map(str, event_desc)))
    elif callable(event_desc):
        pass
    else:
        raise ValueError(
            "Invalid type for event_desc (should be None, list, "
            f"1darray, dict or callable). Got {type(event_desc)}."
        )

    return event_desc


@verbose
def events_from_annotations(
    raw,
    event_id="auto",
    regexp=r"^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$",
    use_rounding=True,
    chunk_duration=None,
    tol=1e-8,
    verbose=None,
):
    """Get :term:`events` and ``event_id`` from an Annotations object.

    Parameters
    ----------
    raw : instance of Raw
        The raw data for which Annotations are defined.
    event_id : dict | callable | None | ``'auto'``
        Can be:

        - **dict**: map descriptions (keys) to integer event codes (values).
          Only the descriptions present will be mapped, others will be ignored.
        - **callable**: must take a string input and return an integer event
          code, or return ``None`` to ignore the event.
        - **None**: Map descriptions to unique integer values based on their
          ``sorted`` order.
        - **'auto' (default)**: prefer a raw-format-specific parser:

          - Brainvision: map stimulus events to their integer part; response
            events to integer part + 1000; optic events to integer part + 2000;
            'SyncStatus/Sync On' to 99998; 'New Segment/' to 99999;
            all others like ``None`` with an offset of 10000.
          - Other raw formats: Behaves like None.

          .. versionadded:: 0.18
    regexp : str | None
        Regular expression used to filter the annotations whose
        descriptions is a match. The default ignores descriptions beginning
        ``'bad'`` or ``'edge'`` (case-insensitive).

        .. versionchanged:: 0.18
           Default ignores bad and edge descriptions.
    use_rounding : bool
        If True, use rounding (instead of truncation) when converting
        times to indices. This can help avoid non-unique indices.
    chunk_duration : float | None
        Chunk duration in seconds. If ``chunk_duration`` is set to None
        (default), generated events correspond to the annotation onsets.
        If not, :func:`mne.events_from_annotations` returns as many events as
        they fit within the annotation duration spaced according to
        ``chunk_duration``. As a consequence annotations with duration shorter
        than ``chunk_duration`` will not contribute events.
    tol : float
        The tolerance used to check if a chunk fits within an annotation when
        ``chunk_duration`` is not ``None``. If the duration from a computed
        chunk onset to the end of the annotation is smaller than
        ``chunk_duration`` minus ``tol``, the onset will be discarded.
    %(verbose)s

    Returns
    -------
    %(events)s
    event_id : dict
        The event_id variable that can be passed to :class:`~mne.Epochs`.

    See Also
    --------
    mne.annotations_from_events

    Notes
    -----
    For data formats that store integer events as strings (e.g., NeuroScan
    ``.cnt`` files), passing the Python built-in function :class:`int` as the
    ``event_id`` parameter will do what most users probably want in those
    circumstances: return an ``event_id`` dictionary that maps event ``'1'`` to
    integer event code ``1``, ``'2'`` to ``2``, etc.
    """
    if len(raw.annotations) == 0:
        event_id = dict() if not isinstance(event_id, dict) else event_id
        return np.empty((0, 3), dtype=int), event_id

    annotations = raw.annotations

    event_id = _check_event_id(event_id, raw)

    event_sel, event_id_ = _select_annotations_based_on_description(
        annotations.description, event_id=event_id, regexp=regexp
    )

    if chunk_duration is None:
        inds = raw.time_as_index(
            annotations.onset, use_rounding=use_rounding, origin=annotations.orig_time
        )
        if annotations.orig_time is not None:
            inds += raw.first_samp
        values = [event_id_[kk] for kk in annotations.description[event_sel]]
        inds = inds[event_sel]
    else:
        inds = values = np.array([]).astype(int)
        for annot in annotations[event_sel]:
            annot_offset = annot["onset"] + annot["duration"]
            _onsets = np.arange(annot["onset"], annot_offset, chunk_duration)
            good_events = annot_offset - _onsets >= chunk_duration - tol
            if good_events.any():
                _onsets = _onsets[good_events]
                _inds = raw.time_as_index(
                    _onsets, use_rounding=use_rounding, origin=annotations.orig_time
                )
                _inds += raw.first_samp
                inds = np.append(inds, _inds)
                _values = np.full(
                    shape=len(_inds),
                    fill_value=event_id_[annot["description"]],
                    dtype=int,
                )
                values = np.append(values, _values)

    events = np.c_[inds, np.zeros(len(inds)), values].astype(int)

    logger.info(f"Used Annotations descriptions: {list(event_id_.keys())}")

    return events, event_id_


@verbose
def annotations_from_events(
    events, sfreq, event_desc=None, first_samp=0, orig_time=None, verbose=None
):
    """Convert an event array to an Annotations object.

    Parameters
    ----------
    events : ndarray, shape (n_events, 3)
        The events.
    sfreq : float
        Sampling frequency.
    event_desc : dict | array-like | callable | None
        Events description. Can be:

        - **dict**: map integer event codes (keys) to descriptions (values).
          Only the descriptions present will be mapped, others will be ignored.
        - **array-like**: list, or 1d array of integers event codes to include.
          Only the event codes present will be mapped, others will be ignored.
          Event codes will be passed as string descriptions.
        - **callable**: must take a integer event code as input and return a
          string description or None to ignore it.
        - **None**: Use integer event codes as descriptions.
    first_samp : int
        The first data sample (default=0). See :attr:`mne.io.Raw.first_samp`
        docstring.
    orig_time : float | str | datetime | tuple of int | None
        Determines the starting time of annotation acquisition. If None
        (default), starting time is determined from beginning of raw data
        acquisition. For details, see :meth:`mne.Annotations` docstring.
    %(verbose)s

    Returns
    -------
    annot : instance of Annotations
        The annotations.

    See Also
    --------
    mne.events_from_annotations

    Notes
    -----
    Annotations returned by this function will all have zero (null) duration.

    Creating events from annotations via the function
    `mne.events_from_annotations` takes in event mappings with
    key→value pairs as description→ID, whereas `mne.annotations_from_events`
    takes in event mappings with key→value pairs as ID→description.
    If you need to use these together, you can invert the mapping by doing::

        event_desc = {v: k for k, v in event_id.items()}
    """
    event_desc = _check_event_description(event_desc, events)
    event_sel, event_desc_ = _select_events_based_on_id(events, event_desc)
    events_sel = events[event_sel]
    onsets = (events_sel[:, 0] - first_samp) / sfreq
    descriptions = [event_desc_[e[2]] for e in events_sel]
    durations = np.zeros(len(events_sel))  # dummy durations

    # Create annotations
    annots = Annotations(
        onset=onsets, duration=durations, description=descriptions, orig_time=orig_time
    )

    return annots


def _adjust_onset_meas_date(annot, raw):
    """Adjust the annotation onsets based on raw meas_date."""
    # If there is a non-None meas date, then the onset should take into
    # account the first_samp / first_time.
    if raw.info["meas_date"] is not None:
        annot.onset += raw.first_time


def count_annotations(annotations):
    """Count annotations.

    Parameters
    ----------
    annotations : mne.Annotations
        The annotations instance.

    Returns
    -------
    counts : dict
        A dictionary containing unique annotation descriptions as keys with their
        counts as values.

    Examples
    --------
        >>> annotations = mne.Annotations([0, 1, 2], [1, 2, 1], ["T0", "T1", "T0"])
        >>> count_annotations(annotations)
        {'T0': 2, 'T1': 1}
    """
    types, counts = np.unique(annotations.description, return_counts=True)
    return {str(t): int(count) for t, count in zip(types, counts)}
