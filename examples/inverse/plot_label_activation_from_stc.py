"""
==================================================
Extracting time course from source_estimate object
==================================================


"""
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

print __doc__

import os

import mne
from mne.datasets import sample
import matplotlib.pyplot as plt

data_path = sample.data_path('..')
os.environ['SUBJECTS_DIR'] = data_path + '/subjects'
meg_path = data_path + '/MEG/sample'

# load the stc
stc = mne.read_source_estimate(meg_path + '/sample_audvis-meg')

# load the labels
aud_lh = mne.read_label(meg_path + '/labels/Aud-lh.label')
aud_rh = mne.read_label(meg_path + '/labels/Aud-rh.label')

# extract the time course for different labels from the stc
stc_lh = stc.label_stc(aud_lh)
stc_rh = stc.label_stc(aud_rh)
stc_bh = stc.label_stc(aud_lh + aud_rh)

# calculate center of mass and transform to mni coordinates
vtx, _, t_lh = stc_lh.center_of_mass('sample')
mni_lh = mne.vertex_to_mni(vtx, 0, 'sample')[0]
vtx, _, t_rh = stc_rh.center_of_mass('sample')
mni_rh = mne.vertex_to_mni(vtx, 1, 'sample')[0]

# plot the activation
plt.figure()
plt.axes([.1, .3, .8, .6])
hl = plt.plot(stc.times, stc_lh.data.mean(0), 'b')
plt.axvline(t_lh, color='b')
hr = plt.plot(stc.times, stc_rh.data.mean(0), 'g')
plt.axvline(t_rh, color='g')
hb = plt.plot(stc.times, stc_bh.data.mean(0), 'r')
plt.xlabel('Time [s]')
plt.ylabel('source amplitude (dSPM)')

# add a legend including center-of-mass mni coordinates to the plot
labels = ['LH: center of mass = %s' % mni_lh.round(2), 
          'RH: center of mass = %s' % mni_rh.round(2), 
          'Combined LH & RH']
plt.figlegend([hl, hr, hb], labels, 'lower right')
plt.title('Average activation in auditory cortex labels')
plt.show()
