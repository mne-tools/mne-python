"""
======================
Plot M/EEG field lines
======================

In this example, M/EEG data are remapped onto the
MEG helmet (MEG) and subject's head surface (EEG).
This process can be computationally intensive.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Denis A. Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

# License: BSD (3-clause)

print(__doc__)

import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
setno = 'Left Auditory'

evoked = mne.fiff.read_evoked(evoked_fname, setno=setno,
                              baseline=(-0.2, 0.0))


surf_maps = mne.make_surface_map(evoked, trans_fname=trans_fname,
                             subject='sample', subjects_dir=subjects_dir,
                             n_jobs=-1)

for time in [0.09, .11]:
    mne.viz.plot_evoked_field(evoked, surf_maps, time=time)
