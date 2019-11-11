"""
===============================================
Analyze Data Using Current Source Density (CSD)
===============================================

This script shows an example of how to use CSD. CSD
takes the spatial Laplacian of the sensor signal
(derivative in both x and y). The spatial derivative
reduces point spread. CSD transformed data have a
sharper or more distinct topography, reducing the
negative impact of volume conduction.

"""
# Authors: Alex Rockhill <aprockhill206@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Load sample subject data
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif',
                          preload=True)
raw = raw.pick_types(meg=False, eeg=True, eog=True, ecg=True, stim=True,
                     exclude=raw.info['bads'])
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=.5,
                    preload=True)
evoked = epochs['auditory'].average()

###############################################################################
# Let's look at the topography of CSD compared to average
fig, axes = plt.subplots(1, 6)
times = np.array([-0.1, 0., 0.05, 0.1, 0.15])
evoked.plot_topomap(times=times, cmap='Spectral_r', axes=axes[:5],
                    outlines='skirt', contours=4, time_unit='s',
                    colorbar=True, show=False, title='Average Reference')
fig, axes = plt.subplots(1, 6)
csd_evoked = mne.preprocessing.compute_current_source_density(evoked,
                                                              copy=True)
csd_evoked.plot_topomap(times=times, axes=axes[:5], cmap='Spectral_r',
                        outlines='skirt', contours=4, time_unit='s',
                        colorbar=True, title='Current Source Density')

###############################################################################
# Plot evoked
evoked.plot_joint(title='Average Reference', times=times, show=False)
csd_evoked.plot_joint(title='Current Source Density', times=times)

###############################################################################
# Look at the effect of smoothing and spline flexibility
fig, ax = plt.subplots(4, 4)
fig.set_size_inches(10, 10)
fig.subplots_adjust(hspace=0.5)
for i, lambda2 in enumerate([0, 1e-7, 1e-5, 1e-3]):
    for j, m in enumerate([5, 4, 3, 2]):
        this_csd_evoked = \
            mne.preprocessing.compute_current_source_density(evoked,
                                                             stiffness=m,
                                                             lambda2=lambda2,
                                                             copy=True)
        this_csd_evoked.plot_topomap(0.1, axes=ax[i, j],
                                     outlines='skirt', contours=4,
                                     time_unit='s',
                                     colorbar=False, show=False)
        ax[i, j].set_title('stiffness=%i\nlambda=%s' % (m, lambda2))

plt.show()

# References
# ----------
#
# [1] Perrin F, Bertrand O, Pernier J. "Scalp current density mapping:
#     Value and estimation from potential data." IEEE Trans Biomed Eng.
#     1987;34(4):283–288.
#
# [2] Perrin F, Pernier J, Bertrand O, Echallier JF. "Spherical splines
#     for scalp potential and current density mapping."
#     [Corrigenda EEG 02274, EEG Clin. Neurophysiol., 1990, 76, 565]
#     Electroenceph Clin Neurophysiol. 1989;72(2):184–187.
#
# [3] Kayser J, Tenke CE. "On the benefits of using surface Laplacian
#     (Current Source Density) methodology in electrophysiology."
#     Int J Psychophysiol. 2015 Sep; 97(3): 171–173.
