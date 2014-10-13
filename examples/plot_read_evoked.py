"""
==================================
Reading and writing an evoked file
==================================

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

print(__doc__)

from mne import read_evokeds
from mne.datasets import sample

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

# Reading
condition = 'Left Auditory'
evoked = read_evokeds(fname, condition=condition, baseline=(None, 0),
                      proj=True)

###############################################################################
# Show result as a butteryfly plot:
# By using exclude=[] bad channels are not excluded and are shown in red
evoked.plot(exclude=[])

# Show result as a 2D image (x: time, y: channels, color: amplitude)
evoked.plot_image(exclude=[])
