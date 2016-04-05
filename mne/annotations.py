# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

from datetime import datetime
import time

import numpy as np

from .externals.six import string_types


class Annotations(object):
    """Annotation object for annotating segments of raw data.

    Parameters
    ----------
    onset : array of float, shape (n_annotations,)
        Annotation time onsets from the beginning of the recording.
    duration : array of float, shape (n_annotations,)
        Durations of the annotations.
    description : array of str, shape (n_annotations,) | str
        Array of strings containing description for each annotation. If a
        string, all the annotations are given the same description.
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
    Annotations are synced to sample 0. ``raw.first_samp`` is taken
    into account in the same way as with events.
    """

    def __init__(self, onset, duration, description, orig_time=None):

        if orig_time is not None:
            if isinstance(orig_time, datetime):
                orig_time = float(time.mktime(orig_time.timetuple()))
            elif not np.isscalar(orig_time):
                orig_time = orig_time[0] + orig_time[1] / 1000000.
            else:  # isscalar
                orig_time = float(orig_time)  # np.int not serializable
        self.orig_time = orig_time

        onset = np.array(onset)
        if onset.ndim != 1:
            raise ValueError('Onset must be a one dimensional array.')
        duration = np.array(duration)
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
        self.description = np.array(description)


def _combine_annotations(annotations, last_samps, first_samps, sfreq):
    """Helper for combining a tuple of annotations."""
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

    if annotations[1].orig_time is None:
        onset = (annotations[1].onset +
                 (sum(last_samps[:-1]) - sum(first_samps[:-1])) / sfreq)
    else:
        onset = annotations[1].onset
    onset = np.concatenate([old_onset, onset])
    duration = np.concatenate([old_duration, annotations[1].duration])
    description = np.concatenate([old_description, annotations[1].description])
    return Annotations(onset, duration, description, old_orig_time)


def _onset_to_seconds(raw, onset):
    """Helper function for adjusting annotation onsets in relation to raw data.
    """
    meas_date = raw.info['meas_date']
    if meas_date is None:
        meas_date = 0
    elif not np.isscalar(meas_date):
        meas_date = meas_date[0] + meas_date[1] / 1000000.
    if raw.annotations.orig_time is None:
        orig_time = meas_date
    else:
        orig_time = raw.annotations.orig_time

    annot_start = (orig_time - meas_date + onset -
                   raw.first_samp / raw.info['sfreq'])
    return annot_start
