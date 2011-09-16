"""
==========================================================
Morph source estimates from one subject to another subject
==========================================================

A source estimate from a given subject 'sample' is morphed
to the anatomy of another subject 'fsaverage'. The output
is a source estimate defined on the anatomy of 'fsaverage'

"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.datasets import sample

data_path = sample.data_path('..')

subject_from = 'sample'
subject_to = 'fsaverage'

fname = data_path + '/MEG/sample/sample_audvis-meg'
src_fname = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'

# Read input stc file
stc_from = mne.SourceEstimate(fname)
# Morph
stc_to = mne.morph_data(subject_from, subject_to, stc_from)

stc_to.save('%s_audvis-meg' % subject_to)

# View source activations
import pylab as pl
pl.plot(stc_from.times, stc_from.data.mean(axis=0), 'r', label='from')
pl.plot(stc_to.times, stc_to.data.mean(axis=0), 'b', label='to')
pl.xlabel('time (ms)')
pl.ylabel('Mean Source amplitude')
pl.legend()
pl.show()
