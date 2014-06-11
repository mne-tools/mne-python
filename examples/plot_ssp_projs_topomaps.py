"""
=================================
Plot SSP projections topographies
=================================

This example shows how to display topographies of SSP projection vectors.
The projections used are the ones correcting for ECG artifacts.
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#         Denis A. Engemann <d.engemann@fz-juuelich.de>

# License: BSD (3-clause)

print(__doc__)

import matplotlib.pyplot as plt
from mne import read_proj, find_layout, read_evokeds
from mne.datasets import sample
from mne import viz
data_path = sample.data_path()

ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg_proj.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

evoked = read_evokeds(ave_fname, condition='Left Auditory')
projs = read_proj(ecg_fname)

layouts = [find_layout(evoked.info, k) for k in 'meg', 'eeg']

plt.figure(figsize=(12, 6))
viz.plot_projs_topomap(projs, layout=layouts)
viz.tight_layout(w_pad=0.5)
