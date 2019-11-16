import os
import os.path as op
import numpy as np

import mne
from mne.datasets import sample
from mne import read_evokeds

from mne.viz import plot_evoked_field
from mne.forward import make_field_map

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked data
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
surf_maps = make_field_map(evoked, subject='sample',
                           subjects_dir=op.join(data_path, 'subjects'),
                           trans=fname_trans)
plot_evoked_field(evoked, surf_maps)
