"""
======================
Plot M/EEG field lines
======================

In this example, M/EEG data are remapped onto the
MEG helmet (MEG) and subject's head surface (EEG).
This process can be computationally intensive.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

# License: BSD (3-clause)

from mne.datasets import sample
from mne import make_field_map, read_evokeds

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
# If trans_fname is set to None then only MEG estimates can be visualized

condition = 'Left Auditory'
evoked = read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))

# Compute the field maps to project MEG and EEG data to MEG helmet
# and scalp surface
maps = make_field_map(evoked, trans_fname, subject='sample',
                      subjects_dir=subjects_dir, n_jobs=1)

# explore several points in time
[evoked.plot_field(maps, time=time) for time in [0.09, .11]]
