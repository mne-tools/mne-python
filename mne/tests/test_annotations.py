# Authors: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD 3 clause

import numpy as np
from datetime import datetime
from nose.tools import assert_raises

from mne.annotations import Annotations


def test_annotations():
    """Test annotation class."""
    onset = list(range(10))
    duration = np.ones(10)
    description = np.repeat('test', 10)
    dt = datetime.utcnow()
    # Test time shifts.
    for orig_time in [None, dt, 1424254822, [1424254822, 532093]]:
        annot = Annotations(onset, duration, description, orig_time)
        annot._serialize()

    assert_raises(ValueError, Annotations, onset, duration, description[:9])
    assert_raises(ValueError, Annotations, [onset, 1], duration, description)
    assert_raises(ValueError, Annotations, onset, [duration, 1], description)
