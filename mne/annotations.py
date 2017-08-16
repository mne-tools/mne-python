# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

from datetime import datetime
import time

import numpy as np

from .utils import _pl
from .externals.six import string_types


class Annotations(object):
    """Annotation object for annotating segments of raw data.

    Annotations are added to instance of :class:`mne.io.Raw` as an attribute
    named ``annotations``. To reject bad epochs using annotations, use
    annotation description starting with 'bad' keyword. The epochs with
    overlapping bad segments are then rejected automatically by default.

    To remove epochs with blinks you can do::

        >>> eog_events = mne.preprocessing.find_eog_events(raw)  # doctest: +SKIP
        >>> n_blinks = len(eog_events)  # doctest: +SKIP
        >>> onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25  # doctest: +SKIP
        >>> duration = np.repeat(0.5, n_blinks)  # doctest: +SKIP
        >>> description = ['bad blink'] * n_blinks  # doctest: +SKIP
        >>> annotations = mne.Annotations(onset, duration, description)  # doctest: +SKIP
        >>> raw.annotations = annotations  # doctest: +SKIP
        >>> epochs = mne.Epochs(raw, events, event_id, tmin, tmax)  # doctest: +SKIP

    Parameters
    ----------
    onset : array of float, shape (n_annotations,)
        Annotation time onsets relative to the ``orig_time``, the starting time
        of annotation acquisition.
    duration : array of float, shape (n_annotations,)
        Durations of the annotations in seconds.
    description : array of str, shape (n_annotations,) | str
        Array of strings containing description for each annotation. If a
        string, all the annotations are given the same description. To reject
        epochs, use description starting with keyword 'bad'. See example above.
    orig_time : float | int | instance of datetime | array of int | None
        A POSIX Timestamp, datetime or an array containing the timestamp as the
        first element and microseconds as the second element. Determines the
        starting time of annotation acquisition. If None (default),
        starting time is determined from beginning of raw data acquisition.
        In general, ``raw.info['meas_date']`` (or None) can be used for syncing
        the annotations with raw data if their acquisiton is started at the
        same time.

    Notes
    -----
    If ``orig_time`` is None, the annotations are synced to the start of the
    data (0 seconds). Otherwise the annotations are synced to sample 0 and
    ``raw.first_samp`` is taken into account the same way as with events.
    """  # noqa: E501

    def __init__(self, onset, duration, description,
                 orig_time=None):  # noqa: D102
        if orig_time is not None:
            if isinstance(orig_time, datetime):
                orig_time = float(time.mktime(orig_time.timetuple()))
            elif not np.isscalar(orig_time):
                orig_time = orig_time[0] + orig_time[1] / 1000000.
            else:  # isscalar
                orig_time = float(orig_time)  # np.int not serializable
        self.orig_time = orig_time

        onset = np.array(onset, dtype=float)
        if onset.ndim != 1:
            raise ValueError('Onset must be a one dimensional array.')
        duration = np.array(duration, dtype=float)
        if isinstance(description, string_types):
            description = np.repeat(description, len(onset))
        if duration.ndim != 1:
            raise ValueError('Duration must be a one dimensional array.')
        if not (len(onset) == len(duration) == len(description)):
            raise ValueError('Onset, duration and description must be '
                             'equal in sizes.')
        if any([';' in desc for desc in description]):
            raise ValueError('Semicolons in descriptions not supported.')

        self.onset = onset
        self.duration = duration
        self.description = np.array(description, dtype=str)

    def __repr__(self):
        """Show the representation."""
        kinds = sorted(set('%s' % d.split(' ')[0].lower()
                           for d in self.description))
        kinds = ['%s (%s)' % (kind, sum(d.lower().startswith(kind)
                                        for d in self.description))
                 for kind in kinds]
        kinds = ', '.join(kinds[:3]) + ('' if len(kinds) <= 3 else '...')
        kinds = (': ' if len(kinds) > 0 else '') + kinds
        return ('<Annotations  |  %s segment%s %s >'
                % (len(self.onset), _pl(len(self.onset)), kinds))

    def __len__(self):
        """Return the number of annotations."""
        return len(self.duration)

    def append(self, onset, duration, description):
        """Add an annotated segment. Operates inplace.

        Parameters
        ----------
        onset : float
            Annotation time onset from the beginning of the recording in
            seconds.
        duration : float
            Duration of the annotation in seconds.
        description : str
            Description for the annotation. To reject epochs, use description
            starting with keyword 'bad'
        """
        self.onset = np.append(self.onset, onset)
        self.duration = np.append(self.duration, duration)
        self.description = np.append(self.description, description)

    def delete(self, idx):
        """Remove an annotation. Operates inplace.

        Parameters
        ----------
        idx : int | list of int
            Index of the annotation to remove.
        """
        self.onset = np.delete(self.onset, idx)
        self.duration = np.delete(self.duration, idx)
        self.description = np.delete(self.description, idx)


def _combine_annotations(annotations, last_samps, first_samps, sfreq,
                         meas_date):
    """Combine a tuple of annotations."""
    if not any(annotations):
        return None
    elif annotations[1] is None:
        return annotations[0]
    elif annotations[0] is None:
        old_onset = list()
        old_duration = list()
        old_description = list()
        old_orig_time = None
    else:
        old_onset = annotations[0].onset
        old_duration = annotations[0].duration
        old_description = annotations[0].description
        old_orig_time = annotations[0].orig_time

    extra_samps = len(first_samps)  # Account for sample 0
    if old_orig_time is not None and annotations[1].orig_time is None:
        meas_date = _handle_meas_date(meas_date)
        extra_samps += sfreq * (meas_date - old_orig_time) + first_samps[0]

    onset = annotations[1].onset + (np.sum(last_samps) + extra_samps -
                                    np.sum(first_samps)) / sfreq

    onset = np.concatenate([old_onset, onset])
    duration = np.concatenate([old_duration, annotations[1].duration])
    description = np.concatenate([old_description, annotations[1].description])
    return Annotations(onset, duration, description, old_orig_time)


def _handle_meas_date(meas_date):
    """Convert meas_date to seconds."""
    if meas_date is None:
        meas_date = 0
    elif not np.isscalar(meas_date):
        if len(meas_date) > 1:
            meas_date = meas_date[0] + meas_date[1] / 1000000.
        else:
            meas_date = meas_date[0]
    return meas_date


def _sync_onset(raw, onset, inverse=False):
    """Adjust onsets in relation to raw data."""
    meas_date = _handle_meas_date(raw.info['meas_date'])
    if raw.annotations.orig_time is None:
        orig_time = meas_date
    else:
        offset = -raw._first_time if inverse else raw._first_time
        orig_time = raw.annotations.orig_time - offset

    annot_start = orig_time - meas_date + onset
    return annot_start


def _annotations_starts_stops(raw, kinds, name='unknown'):
    """Get starts and stops from given kinds."""
    if not isinstance(kinds, (string_types, list, tuple)):
        raise TypeError('%s must be str, list, or tuple, got %s'
                        % (type(kinds), name))
    elif isinstance(kinds, string_types):
        kinds = [kinds]
    elif not all(isinstance(kind, string_types) for kind in kinds):
        raise TypeError('All entries in %s must be str' % (name,))
    if raw.annotations is None:
        return np.array([], int), np.array([], int)
    idxs = [idx for idx, desc in enumerate(raw.annotations.description)
            if any(desc.upper().startswith(kind.upper())
                   for kind in kinds)]
    onsets = raw.annotations.onset[idxs]
    onsets = _sync_onset(raw, onsets)
    ends = onsets + raw.annotations.duration[idxs]
    order = np.argsort(onsets)
    onsets = raw.time_as_index(onsets[order])
    ends = raw.time_as_index(ends[order])
    return onsets, ends
