"""
=================================
Plot SSP projections topographies
=================================

This example shows how to display topographies of SSP projection vectors.
The projections used are the ones correcting for ECG artifacts.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#         Denis A. Engemann <denis.engemann@gmail.com>
#         Teon Brooks <teon.brooks@gmail.com>

# License: BSD (3-clause)

print(__doc__)

from mne import read_proj, find_layout, read_evokeds
from mne.datasets import sample
from mne import viz
data_path = sample.data_path()

ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg_proj.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

evoked = read_evokeds(ave_fname, condition='Left Auditory')
projs = read_proj(ecg_fname)
evoked.add_proj(projs)

evoked.plot_projs_topomap(ch_type=['meg', 'eeg'])
