"""
=====================================
Plot grouped triaxial OPM topomaps
=====================================

This example demonstrates grouped radial/tangential topomap rendering for
colocated triaxial OPM sensors using a small segment of the UCL OPM auditory
dataset. The grouped rendering places radial maps alongside tangential maps
so orientation information is explicit.

"""
# Authors: MNE contributors
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

import mne

# Load a small segment of the UCL OPM dataset
subject = "sub-002"
data_path = mne.datasets.ucl_opm_auditory.data_path()
opm_file = (
    data_path / subject / "ses-001" / "meg" / "sub-002_ses-001_task-aef_run-001_meg.bin"
)

# Read and crop for speed
raw = mne.io.read_raw_fil(opm_file, verbose="error")
raw.crop(120, 210).load_data()

# Create epochs and average to get evoked
events = mne.find_events(raw, min_duration=0.1)
epochs = mne.Epochs(
    raw, events, tmin=-0.1, tmax=0.4, baseline=(None, 0), verbose="error"
)
evoked = epochs.average()

# Find a peak time and plot grouped topomap
t_peak = evoked.times[np.argmax(np.std(evoked.copy().pick("meg").data, axis=0))]
fig = evoked.plot_topomap(times=[float(t_peak)], ch_type="mag")
plt.show()
