"""
================
 XDAWN Denoising
================

Xdawn filters are fitted on half of the epochs. The second half is then
projected on to the source space and projected back on to the sensor space
using only the first two Xdawn components. The process is similar to an ICA,
but is supervised in order to maximize the signal to noise ratio of the
evoked response.

References
----------
[1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
algorithm to enhance evoked potentials: application to brain-computer
interface. Biomedical Engineering, IEEE Transactions on, 56(8), 2035-2043.

[2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J. (2011,
August). Theoretical analysis of xDAWN algorithm: application to an
efficient sensor selection in a P300 BCI. In Signal Processing Conference,
2011 19th European (pp. 1382-1386). IEEE.
"""

# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)


from mne import (io, compute_raw_data_covariance, read_events, pick_types,
                 Epochs)
from mne.datasets import sample
from mne.preprocessing import Xdawn
from mne.viz import plot_image_epochs

print(__doc__)

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.3
event_id = dict(vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True)
raw.filter(1, 20, method='iir')  # replace baselining with high-pass
events = read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                   exclude='bads')

# Epoching
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                picks=picks, baseline=None, preload=True,
                add_eeg_ref=False, verbose=False)

# Divide epochs into a train and a test set to avoid potential circular
# analyses (a.k.a. double dipping) in later steps.
train = range(len(epochs) / 2)
test = range(len(epochs) / 2, len(epochs))

# Estimates signal covariance on first half of the data
half_time = epochs[train].events[-1, 0] / raw.info['sfreq']
raw.crop(0, half_time)
signal_cov = compute_raw_data_covariance(raw, picks=picks)

# Plot image epoch before xdawn
plot_image_epochs(epochs['vis_r'][test], picks=[230], vmin=-500, vmax=500)

# Xdawn instance
xd = Xdawn(n_components=2, signal_cov=signal_cov)

# Fit xdawn
xd.fit(epochs[train])

# Denoise epochs
epochs_denoised = xd.apply(epochs[test])

# Plot image epoch after xdawn
plot_image_epochs(epochs_denoised['vis_r'], picks=[230], vmin=-500, vmax=500)
