"""
====================================================
Read a forward operator and display sensitivity maps
====================================================
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample
data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'

fwd = mne.read_forward_solution(fname, surf_ori=True)
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
grad_map.plot(subject='sample', time_label='Gradiometers sensitivity', **args)
