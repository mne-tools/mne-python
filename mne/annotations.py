# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import numpy as np
from datetime import datetime
import time

class Annotations():

    def __init__(self, orig_time, segments, descriptions):
        """
        """
        if isinstance(orig_time, datetime):
            orig_time = time.mktime(orig_time.timetuple())
        self.orig_time = orig_time
        if len(segments) != len(descriptions):
            raise RuntimeError('Segments and descriptions are different in '
                               'size. Every segment must have a description.')
        segments = np.array(segments)
        # sort the segments by start time
        self.segments = segments[segments[:, 0].argsort(axis=0)]
        self.descriptions = descriptions
