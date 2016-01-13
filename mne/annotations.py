# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
from datetime import datetime
import time


class Annotations(object):

    def __init__(self, onsets, durations, descriptions, orig_time=None):
        """Annotation object for annotating segments of raw data.

        Parameters
        ----------
        onsets: array of float
            Annotation time onsets from the beginning of the recording.
        durations: array of float
            Durations of the annotations.
        descriptions: array of str
            Array of strings containing description for each annotation.
        orig_time: int | instance of datetime | None
            Timestamp or a datetime determining the starting time of annotation
            acquisition. If None (default), starting time is determined from
            beginning of raw data.
        """
        if orig_time is not None:
            if isinstance(orig_time, datetime):
                orig_time = time.mktime(orig_time.timetuple())
            elif not np.isscalar(orig_time):
                orig_time = orig_time[0]
        self.orig_time = orig_time

        onsets = np.array(onsets)
        if len(onsets.shape) != 1:
            raise RuntimeError('Onsets must be a one dimensional array.')
        durations = np.array(durations)
        if len(durations.shape) != 1:
            raise RuntimeError('Durations must be a one dimensional array.')
        if not (len(onsets) == len(durations) == len(descriptions)):
            raise RuntimeError('Onsets, durations and descriptions must be '
                               'equal in sizes.')
        # sort the segments by start time
        order = onsets.argsort(axis=0)
        self.onsets = onsets[order]
        self.durations = durations[order]
        self.descriptions = np.array(descriptions)[order]
