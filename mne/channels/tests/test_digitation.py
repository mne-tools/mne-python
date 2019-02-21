# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import os.path as op

import pytest

from mne import __file__ as mne_init_path
from mne.channels.digitization import read_pos
from mne.datasets import sample

from mne.channels import Digitization
from mne.channels.montage import _set_montage, get_builtin_montages
from mne.viz import plot_alignment
from mayavi import mlab

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

POS_FNAME = op.join(op.dirname(mne_init_path), 'channels', 'data', 'test.pos')


def get_trans():
    trans = mne.Transform(fro='head', to='mri')
    trans['trans'] = np.array([[0.99981296, -0.00503971,  0.01867181,  0.00255929],
                               [0.00692004,  0.99475515, -0.10205064, -0.02091804],
                               [0.01805957,  0.10216076,  0.99460393, -0.04416016],
                               [0.,          0.,          0.,          1.        ]])
    return trans

def test_digitization():
    read_pos(POS_FNAME)
