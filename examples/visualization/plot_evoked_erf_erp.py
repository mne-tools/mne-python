"""
=================================
Plotting ERF/ERP with evoked data
=================================

Load evoked data and plot.

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
from mne.datasets import sample
from mne import read_evokeds

print(__doc__)

path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# load evoked and subtract baseline
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

evoked.plot()

###############################################################################
# Or plot manually after extracting peak latency

evoked = evoked.pick_types(meg=False, eeg=True)
times = 1e3 * evoked.times  # time in miliseconds

ch_max_name, latency = evoked.get_peak(mode='neg')

plt.figure()
plt.plot(times, 1e6 * evoked.data.T, 'k-')
plt.xlim([times[0], times[-1]])
plt.xlabel('time (ms)')
plt.ylabel('Potential (uV)')
plt.title('EEG evoked potential')

plt.axvline(latency * 1e3, color='red',
            label=ch_max_name, linewidth=2,
            linestyle='--')
plt.legend(loc='best')

plt.show()
