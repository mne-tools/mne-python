# -*- coding: utf-8 -*-
"""
=======================================
Brainstorm CTF phantom tutorial dataset
=======================================

Here we compute the evoked from raw for the Brainstorm CTF phantom
tutorial dataset. For comparison, see [1]_ and:

    http://neuroimage.usc.edu/brainstorm/Tutorials/PhantomCtf

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import fit_dipole
from mne.datasets.brainstorm import bst_phantom_ctf
from mne.io import read_raw_ctf

print(__doc__)

###############################################################################
# The data were collected with a CTF system at 2400 Hz.
data_path = bst_phantom_ctf.data_path()

# Switch to these to use the higher-SNR data:
# raw_path = op.join(data_path, 'phantom_200uA_20150709_01.ds')
# dip_freq = 7.
raw_path = op.join(data_path, 'phantom_20uA_20150603_03.ds')
dip_freq = 23.
erm_path = op.join(data_path, 'emptyroom_20150709_01.ds')
raw = read_raw_ctf(raw_path, preload=True)

###############################################################################
# The sinusoidal signal is generated on channel HDAC006, so we can use
# that to obtain precise timing.

sinusoid, times = raw[raw.ch_names.index('HDAC006-4408')]
plt.figure()
plt.plot(times[times < 1.], sinusoid.T[times < 1.])

###############################################################################
# Let's create some events using this signal by thresholding the sinusoid.

events = np.where(np.diff(sinusoid > 0.5) > 0)[1] + raw.first_samp
events = np.vstack((events, np.zeros_like(events), np.ones_like(events))).T

###############################################################################
# The CTF software compensation works reasonably well:

raw.plot()

###############################################################################
# But here we can get slightly better noise suppression, lower localization
# bias, and a better dipole goodness of fit with spatio-temporal (tSSS)
# Maxwell filtering:

raw.apply_gradient_compensation(0)  # must un-do software compensation first
mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
raw = mne.preprocessing.maxwell_filter(raw, **mf_kwargs)
raw.plot()

###############################################################################
# Our choice of tmin and tmax should capture exactly one cycle, so
# we can make the unusual choice of baselining using the entire epoch
# when creating our evoked data. We also then crop to a single time point
# (@t=0) because this is a peak in our signal.

tmin = -0.5 / dip_freq
tmax = -tmin
epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax,
                    baseline=(None, None))
evoked = epochs.average()
evoked.plot()
evoked.crop(0., 0.)
del raw, epochs

###############################################################################
# To do a dipole fit, let's use the covariance provided by the empty room
# recording.

raw_erm = read_raw_ctf(erm_path).apply_gradient_compensation(0)
raw_erm = mne.preprocessing.maxwell_filter(raw_erm, coord_frame='meg',
                                           **mf_kwargs)
cov = mne.compute_raw_covariance(raw_erm)
del raw_erm
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)
dip = fit_dipole(evoked, cov, sphere)[0]

###############################################################################
# Compare the actual position with the estimated one.

expected_pos = np.array([18., 0., 49.])
diff = np.sqrt(np.sum((dip.pos[0] * 1000 - expected_pos) ** 2))
print('Actual pos:     %s mm' % np.array_str(expected_pos, precision=1))
print('Estimated pos:  %s mm' % np.array_str(dip.pos[0] * 1000, precision=1))
print('Difference:     %0.1f mm' % diff)
print('Amplitude:      %0.1f nAm' % (1e9 * dip.amplitude[0]))
print('GOF:            %0.1f %%' % dip.gof[0])
