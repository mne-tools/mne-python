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

from mne import read_proj, read_evokeds
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg-proj.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

evoked = read_evokeds(ave_fname, condition='Left Auditory',
                      baseline=(None, 0.))

projs = read_proj(ecg_fname)
evoked.add_proj(projs)

evoked.plot_projs_topomap()
