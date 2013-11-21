"""
==================================================
Extracting time course from source_estimate object
==================================================

Load a SourceEstimate object from stc files and
extract the time course of activation in
individual labels, as well as in a complex label
formed through merging two labels.

"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

print __doc__

import os

import mne
from mne.datasets import sample
import pylab as pl

data_path = sample.data_path()
os.environ['SUBJECTS_DIR'] = data_path + '/subjects'
meg_path = data_path + '/MEG/sample'

# load the stc
stc = mne.read_source_estimate(meg_path + '/sample_audvis-meg')

# load the labels
aud_lh = mne.read_label(meg_path + '/labels/Aud-lh.label')
aud_rh = mne.read_label(meg_path + '/labels/Aud-rh.label')

# extract the time course for different labels from the stc
stc_lh = stc.in_label(aud_lh)
stc_rh = stc.in_label(aud_rh)
stc_bh = stc.in_label(aud_lh + aud_rh)

# calculate center of mass and transform to mni coordinates
vtx, _, t_lh = stc_lh.center_of_mass('sample')
mni_lh = mne.vertex_to_mni(vtx, 0, 'sample')[0]
vtx, _, t_rh = stc_rh.center_of_mass('sample')
mni_rh = mne.vertex_to_mni(vtx, 1, 'sample')[0]

# plot the activation
pl.figure()
pl.axes([.1, .275, .85, .625])
hl = pl.plot(stc.times, stc_lh.data.mean(0), 'b')
hr = pl.plot(stc.times, stc_rh.data.mean(0), 'g')
hb = pl.plot(stc.times, stc_bh.data.mean(0), 'r')
pl.xlabel('Time (s)')
pl.ylabel('Source amplitude (dSPM)')
pl.xlim(stc.times[0], stc.times[-1])

# add a legend including center-of-mass mni coordinates to the plot
labels = ['LH: center of mass = %s' % mni_lh.round(2),
          'RH: center of mass = %s' % mni_rh.round(2),
          'Combined LH & RH']
pl.figlegend([hl, hr, hb], labels, 'lower center')
pl.suptitle('Average activation in auditory cortex labels', fontsize=20)
pl.show()
