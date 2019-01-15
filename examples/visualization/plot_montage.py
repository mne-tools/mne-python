"""
======================================
Plotting sensor layouts of EEG Systems
======================================

Show sensor layouts of different EEG systems.

XXX: things to refer properly:
:ref:`example_eeg_sensors_on_the_scalp`
:ref:`tut_erp`
"""
# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.viz import plot_alignment
from mayavi import mlab

print(__doc__)




data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'
trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
raw.pick_types(meg=False, eeg=True, eog=True)
print(raw.info['chs'][0]['loc'])

# Nothing
fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)
mlab.view(135, 80)

# With montage
montage = mne.channels.read_montage('standard_1020')
raw.set_montage(montage)
fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)
mlab.view(135, 80)

# with a name
raw.set_montage('mgh60')  # test loading with string argument
fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
                     eeg=['original', 'projected'], meg=[],
                     coord_frame='head', subjects_dir=subjects_dir)
mlab.view(135, 80)

