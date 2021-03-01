"""
====================================
Continuous Target Decoding with SPoC
====================================

Source Power Comodulation (SPoC) :footcite:`DahneEtAl2014` allows to identify
the composition of
orthogonal spatial filters that maximally correlate with a continuous target.

SPoC can be seen as an extension of the CSP for continuous variables.

Here, SPoC is applied to decode the (continuous) fluctuation of an
electromyogram from MEG beta activity using data from
`Cortico-Muscular Coherence example of FieldTrip
<http://www.fieldtriptoolbox.org/tutorial/coherence>`_
"""

# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

# Define parameters
fname = data_path() + '/SubjectCMC.ds'
raw = mne.io.read_raw_ctf(fname)
raw.crop(50., 250.)  # crop for memory purposes

# Filter muscular activity to only keep high frequencies
emg = raw.copy().pick_channels(['EMGlft']).load_data()
emg.filter(20., None, fir_design='firwin')

# Filter MEG data to focus on beta band
raw.pick_types(meg=True, ref_meg=True, eeg=False, eog=False).load_data()
raw.filter(15., 30., fir_design='firwin')

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=.250)

# Epoch length is 1.5 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1.500, baseline=None,
                    detrend=1, decim=8)
emg_epochs = Epochs(emg, events, tmin=0., tmax=1.500, baseline=None)

# Prepare classification
X = meg_epochs.get_data()
y = emg_epochs.get_data().var(axis=2)[:, 0]  # target is EMG power

# Classification pipeline with SPoC spatial filtering and Ridge Regression
spoc = SPoC(n_components=2, log=True, reg='oas', rank='full')
clf = make_pipeline(spoc, Ridge())
# Define a two fold cross-validation
cv = KFold(n_splits=2, shuffle=False)

# Run cross validaton
y_preds = cross_val_predict(clf, X, y, cv=cv)

# Plot the True EMG power and the EMG power predicted from MEG data
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
ax.plot(times, y_preds, color='b', label='Predicted EMG')
ax.plot(times, y, color='r', label='True EMG')
ax.set_xlabel('Time (s)')
ax.set_ylabel('EMG Power')
ax.set_title('SPoC MEG Predictions')
plt.legend()
mne.viz.tight_layout()
plt.show()

##############################################################################
# Plot the contributions to the detected components (i.e., the forward model)

spoc.fit(X, y)
spoc.plot_patterns(meg_epochs.info)

##############################################################################
# References
# ----------
# .. footbibliography::
