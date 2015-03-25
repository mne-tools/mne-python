"""
=========================================
Compute Rap-Music on evoked data
=========================================

Compute a Recursively Applied and Projected MUltiple Signal Classification
(RAP-MUSIC) on evoked dataset.

The reference for Rap-Music is:
J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
applied and projected (RAP) MUSIC. Trans. Sig. Proc. 47, 2
(February 1999), 332-340.
DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118
"""

# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne

from mne.datasets import sample
from mne.beamformer import rap_music
from mne.viz import plot_dipoles

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Left Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked.crop(tmin=-50e-3, tmax=300e-3)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
# left_temporal_channels = mne.read_selection('Right-temporal')
# left_temporal_channels = mne.read_selection('Right-temporal')
# evoked = mne.pick_types_evoked(evoked, meg=False, eeg=False, include=left_temporal_channels)

evoked = mne.pick_types_evoked(evoked, meg=True, eeg=False)

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Read noise covariance matrix and regularize it
noise_cov = mne.read_cov(cov_fname)

dipoles, residual = rap_music(evoked, forward, noise_cov, n_dipoles=2,
                              return_residual=True, verbose=True,
                              signal_ndim=2)
coord_trans = forward['mri_head_t']['trans']
plot_dipoles(dipoles, coord_trans, subject='sample',
             subjects_dir=subjects_dir, colors=[(0., 0., 1.), (1., 0., 0.)])

# Plot the time-series
plt.figure()
plt.plot(dipoles[0].times, dipoles[0].amplitude, 'b', linewidth=1.5)
plt.plot(dipoles[1].times, dipoles[1].amplitude, 'r', linewidth=1.5)
plt.ylabel('amplitude (nAm)')
plt.xlabel('times (ms)')

# Plot the evoked data and the residual.
evoked.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800]))
residual.plot(ylim=dict(grad=[-300, 300], mag=[-800, 800]))




# ############### test ########
import numpy as np
# active_set = np.array([1640, 5579])
active_set = np.array([1595, 5624])

data = np.array([dipoles[0].amplitude, dipoles[1].amplitude])
tmin = evoked.times.min()
tstep = 1. / evoked.info['sfreq']
src = forward['src']

forward = mne.pick_types_forward(forward, meg=True, eeg=False,
                                 exclude=['MEG 2443'])
forward = mne.convert_forward_solution(forward, force_fixed=True)

forward['source_nn'][active_set[0]] = np.array([.5, 0., 0.])
forward['source_nn'][active_set[1]] = np.array([0., 0., .5])

n_lh_points = len(src[0]['vertno'])
lh_vertno = src[0]['vertno'][active_set[active_set < n_lh_points]]
rh_vertno = src[1]['vertno'][active_set[active_set >= n_lh_points] -
                             n_lh_points]

src[0]['nn'][lh_vertno] = np.array([.5, 0., 0.])
src[1]['nn'][rh_vertno] = np.array([0., 0., .5])

forward['src'] = src
vertices = [lh_vertno, rh_vertno]
stc = mne.SourceEstimate(data, vertices=vertices, tmin=tmin, tstep=tstep)
rng = np.random.RandomState(0)
ev_sim = mne.simulation.generate_evoked(forward, stc, evoked, noise_cov,
                                        snr=20, random_state=rng)

dipoles, residual = rap_music(ev_sim, forward, noise_cov, n_dipoles=2,
                              return_residual=True, verbose=True,
                              signal_ndim=2)

