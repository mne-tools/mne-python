import numpy as np

from ..source import circular_source_labels

def test_circular_source_labels():
    """ Test generation of circular source labels """

    seeds = [0, 50000]
    hemis = [0, 1]
    labels = circular_source_labels('sample', seeds, 3, hemis)

    for label, seed, hemi in zip(labels, seeds, hemis):
        assert(np.any(label['vertices'] == seed))
        if hemi == 0:
            assert(label['hemi'] == 'lh')
        else:
            assert(label['hemi'] == 'rh')

