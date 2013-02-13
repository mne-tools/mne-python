"""
=======================================================
Reading a forward operator and display sensitivity maps
=======================================================
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

grad_map = mne.proj.sensitivity_map(fwd, ch_type='grad')
mag_map = mne.proj.sensitivity_map(fwd, ch_type='mag')
eeg_map = mne.proj.sensitivity_map(fwd, ch_type='eeg')

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivy map

import pylab as pl
pl.matshow(leadfield[:, :500])
pl.xlabel('sources')
pl.ylabel('sensors')
pl.title('Lead field matrix')
pl.show()

grad_map.plot(subject='sample', surface='white', fmin=0.2, fmid=0.6, fmax=1)
mag_map.plot(subject='sample', fmin=0.2, fmid=0.6, fmax=1)
eeg_map.plot(subject='sample', surface='white', fmin=0.2, fmid=0.6, fmax=1)
