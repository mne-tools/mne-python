"""
====================================================
Read a forward operator and display sensitivity maps
====================================================
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne.datasets import sample
data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'

fwd = mne.read_forward_solution(fname, surf_ori=True)
leadfield = fwd['sol']['data']

print("Leadfield size : %d x %d" % leadfield.shape)

grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivity map

import matplotlib.pyplot as plt
plt.matshow(leadfield[:, :500])
plt.xlabel('sources')
plt.ylabel('sensors')
plt.title('Lead field matrix (500 dipoles only)')

plt.figure()
plt.hist([grad_map.data.ravel(), mag_map.data.ravel(), eeg_map.data.ravel()],
         bins=20, label=['Gradiometers', 'Magnetometers', 'EEG'])
plt.legend()
plt.title('Normal orientation sensitivity')
plt.show()

args = dict(fmin=0.1, fmid=0.5, fmax=0.9, smoothing_steps=7)
grad_map.plot(subject='sample', time_label='Gradiometer sensitivity',
              subjects_dir=subjects_dir, **args)
