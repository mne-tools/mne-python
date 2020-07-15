"""
.. _ex-report:

================================
Make an MNE-Report with a Slider
================================

In this example, MEG evoked data are plotted in an HTML slider.
"""

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from mne.report import Report
from mne.datasets import sample
from mne import read_evokeds
from matplotlib import pyplot as plt


data_path = sample.data_path()
meg_path = data_path + '/MEG/sample'
subjects_dir = data_path + '/subjects'
evoked_fname = meg_path + '/sample_audvis-ave.fif'

###############################################################################
# Do standard folder parsing (this can take a couple of minutes):

report = Report(image_format='png', subjects_dir=subjects_dir,
                info_fname=evoked_fname, subject='sample',
                raw_psd=False)  # use False for speed here
report.parse_folder(meg_path, on_error='ignore', mri_decim=10)

###############################################################################
# Add a custom section with an evoked slider:

# Load the evoked data
evoked = read_evokeds(evoked_fname, condition='Left Auditory',
                      baseline=(None, 0), verbose=False)
evoked.crop(0, .2)
times = evoked.times[::4]
# Create a list of figs for the slider
figs = list()
for t in times:
    figs.append(evoked.plot_topomap(t, vmin=-300, vmax=300, res=100,
                                    show=False))
    plt.close(figs[-1])
report.add_slider_to_section(figs, times, 'Evoked Response',
                             image_format='png')  # can also use 'svg'

# Save the report
report.save('my_report.html', overwrite=True)
