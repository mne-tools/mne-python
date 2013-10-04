"""
======================================================
Create a forward operator and display sensitivity maps
======================================================
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample
data_path = sample.data_path()

bem = data_path + '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'
meas = data_path + '/MEG/sample/sample_audvis_raw.fif'
src = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
mri = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
subjects_dir = data_path + '/subjects'
fname = 'test-fwd.fif'

fwd = mne.do_forward_solution('sample', meas, fname, bem=bem, src=src, mri=mri,
                              subjects_dir=subjects_dir, meg=True, eeg=True,
                              mindist=5.0, n_jobs=2, overwrite=True)

# convert to surface orientation for better visualization
fwd = mne.convert_forward_solution(fwd, surf_ori=True)
leadfield = fwd['sol']['data']

print "Leadfield size : %d x %d" % leadfield.shape

grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivy map

import pylab as pl
pl.matshow(leadfield[:, :500])
pl.xlabel('sources')
pl.ylabel('sensors')
pl.title('Lead field matrix (500 dipoles only)')

pl.figure()
pl.hist([grad_map.data.ravel(), mag_map.data.ravel(), eeg_map.data.ravel()],
        bins=20, label=['Gradiometers', 'Magnetometers', 'EEG'])
pl.legend()
pl.title('Normal orientation sensitivity')
pl.show()

args = dict(fmin=0.1, fmid=0.5, fmax=0.9, smoothing_steps=7)
grad_map.plot(subject='sample', time_label='Gradiometers sensitivity',
              subjects_dir=subjects_dir, **args)
