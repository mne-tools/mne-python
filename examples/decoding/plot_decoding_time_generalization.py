"""
==========================================================
Decoding sensor space data with Generalization Across Time
==========================================================

This example runs the analysis computed in:

Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
unexpected sounds", PLOS ONE, 2013,
http://www.ncbi.nlm.nih.gov/pubmed/24475052

The idea is to learn at one time instant and assess if the decoder
can predict accurately over time.
"""
# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import spm_face
from mne.decoding import GeneralizationAcrossTime

print(__doc__)

# Preprocess data
data_path = spm_face.data_path()
# Load and filter data, set up epochs
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces%d_3D_raw.fif'

raw = mne.io.Raw(raw_fname % 1, preload=True)  # Take first run

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.filter(1, 45, method='iir')

events = mne.find_events(raw, stim_channel='UPPT001')
event_id = {"faces": 1, "scrambled": 2}
tmin, tmax = -0.1, 0.5

decim = 4  # decimate to make the example faster to run
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True,
                    reject=dict(mag=1.5e-12), decim=decim, verbose=False)

# Define decoder. The decision function is employed to use cross-validation
gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1)

# fit and score
gat.fit(epochs)
gat.score(epochs)
gat.plot(vmin=0.1, vmax=0.9,
         title="Generalization Across Time (faces vs. scrambled)")
gat.plot_diagonal()  # plot decoding across time (correspond to GAT diagonal)
