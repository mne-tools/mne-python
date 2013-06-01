"""
=================================
Plot SSP projections topographies
=================================

This example shows how to display topographies of SSP projection vectors.
The projections used are the ones correcting for ECG artifacts.
"""
# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl
import mne
from mne.datasets import sample
data_path = sample.data_path()

ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg_proj.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

evoked = mne.fiff.read_evoked(ave_fname, setno=1)
projs = mne.read_proj(ecg_fname)

layouts = [mne.layouts.read_layout('Vectorview-all'),
           mne.layouts.make_eeg_layout(evoked.info)]

pl.figure(figsize=(10, 6))
mne.viz.plot_projs_topomap(projs, layout=layouts)
mne.viz.tight_layout()
