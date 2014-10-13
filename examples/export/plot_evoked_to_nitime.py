"""
============================
Export evoked data to Nitime
============================

"""
# Author: Denis Engemann <denis.engemann@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne.datasets import sample
from nitime.viz import plot_tseries
import matplotlib.pyplot as plt


data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
evoked = mne.read_evokeds(fname, condition=0, baseline=(None, 0), proj=True)

# Pick channels to view
picks = mne.pick_types(evoked.info, meg='grad', eeg=False, exclude='bads')

evoked_ts = evoked.to_nitime(picks=picks)

plot_tseries(evoked_ts)

plt.show()
