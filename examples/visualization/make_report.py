"""
================================
Make an MNE-Report with a Slider
================================

In this example, MEG evoked data are plotted in an html slider.
"""

# Authors: Teon Brooks <teon.brooks@gmail.com
#
# License: BSD (3-clause)

from mne.report import Report
from mne.datasets import sample
from mne import read_evokeds
from matplotlib import pyplot as plt


report = Report()
path = sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'

# Load the evoked data
evoked = read_evokeds(fname, condition='Left Auditory',
                      baseline=(None, 0), verbose=False)
evoked.crop(0, .2)
times = evoked.times[::4]
# Create a list of figs for the slider
figs = list()
for time in times:
    figs.append(evoked.plot_topomap(time, vmin=-300, vmax=300,
                                    res=100, show=False))
    plt.close(figs[-1])
report.add_slider_to_section(figs, times, 'Evoked Response')

# # to save report
# report.save('foobar.html', True)
