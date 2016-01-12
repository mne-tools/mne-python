# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
from datetime import datetime
import time


class Annotations():

    def __init__(self, orig_time, segments, descriptions):
        """Annotation object for annotating segments of raw data.

        Parameters
        ----------
        orig_time: int | instance of datetime
            Timestamp or a datetime determining the starting time of segment
            acquisition.
        segments: array
            Array that contains segments of data in shape (t_start, duration).
        descriptions: array
            Array of strings containing description for each segment.
        """
        if isinstance(orig_time, datetime):
            orig_time = time.mktime(orig_time.timetuple())
        self.orig_time = orig_time
        segments = np.array(segments)
        if len(segments.shape) != 2 or segments.shape[1] != 2:
            raise RuntimeError('Segments must be an array in shape '
                               '(n_segments, 2).')
        if len(segments) != len(descriptions):
            raise RuntimeError('Segments and descriptions are different in '
                               'size. Every segment must have a description.')
        # sort the segments by start time
        self.segments = segments[segments[:, 0].argsort(axis=0)]
        self.descriptions = descriptions
