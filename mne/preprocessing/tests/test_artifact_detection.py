# Author: Adonay Nunes <adonay.s.nunes@gmail.com>
#
# License: BSD (3-clause)


import os.path as op
import numpy as np
from numpy.testing import assert_allclose
from mne.chpi import read_head_pos
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import (annotate_movement, compute_average_dev_head_t)

data_path = testing.data_path(download=False)
sss_path = op.join(data_path, 'SSS')
pre = op.join(sss_path, 'test_move_anon_')
raw_fname = pre + 'raw.fif'
pos_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.pos')


@testing.requires_testing_data
def test_movement_annotation_head_correction():
    """Test correct detection movement artifact and dev_head_t."""
    raw = read_raw_fif(raw_fname, allow_maxshield='yes').load_data()
    pos = read_head_pos(pos_fname)

    # Check 5 rotation segments are detected
    annot_rot, [] = annotate_movement(raw, pos, rotation_velocity_limit=5)
    assert(annot_rot.duration.size == 5)

    # Check 2 translation vel. segments are detected
    annot_tra, [] = annotate_movement(raw, pos, translation_velocity_limit=.05)
    assert(annot_tra.duration.size == 2)

    # Check 1 movement distance segment is detected
    annot_dis, disp = annotate_movement(raw, pos, mean_distance_limit=.02)
    assert(annot_dis.duration.size == 1)

    # Check correct trans mat
    raw.set_annotations(annot_rot + annot_tra + annot_dis)
    dev_head_t = compute_average_dev_head_t(raw, pos)

    dev_head_t_ori = np.array([
                              [0.9957292, -0.08688804, 0.03120615, 0.00698271],
                              [0.09020767, 0.9875856, -0.12859731, -0.0159098],
                              [-0.01964518, 0.1308631, 0.99120578, 0.07258289],
                              [0., 0., 0., 1.]])

    assert_allclose(dev_head_t_ori, dev_head_t['trans'], rtol=1e-5, atol=0)
