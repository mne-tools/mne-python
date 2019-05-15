"""
========================================================
Import single subject from LIMO data set into MNE-Python
========================================================

Here we a define function to extract the eeg signal data
from the LIMO structures in the LIMO dataset, see [1]_ and:

    https://datashare.is.ed.ac.uk/handle/10283/2189?show=full

    https://github.com/LIMO-EEG-Toolbox

The code allows to:

Fetch single subjects epochs data for the LIMO data set.
Epochs information (i.e., sampling rate, number of epochs per condition,
number and name of EEG channels per subject, etc.) is extracted from
the LIMO .mat files stored on disk.
If files are not found, the function mne.datasets.limo.load_data() will
automatically download the data from a remote repository.

mne.datasets.limo.load_data() creates a custom info and
epochs structure in MNE-Python.
Missing channels can be interpolated if desired.

.. note:: Downloading the LIMO dataset for the first time can take some time (8 GB).


References
----------
.. [1] Guillaume, Rousselet. (2016). LIMO EEG Dataset, [dataset].
       University of Edinburgh, Centre for Clinical Brain Sciences.
       https://doi.org/10.7488/ds/1556.
"""

# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import limo
from mne.preprocessing import ICA
from mne.stats import linear_regression

print(__doc__)

# fetch data from subject 2
limo_epochs = load_data(subject=2, interpolate=True)

# check distribution of events (should be ordered)
mne.viz.plot_events(limo_epochs.events)

# drop EXG-channels (i.e. eog) as data has already been cleaned
limo_epochs.drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])

# plot evoked response for conditions A & B
limo_epochs['Face/A'].average().plot_joint(times=[.09, .15])
limo_epochs['Face/B'].average().plot_joint(times=[.09, .15])

# create design matrix for linear regression
design = limo_epochs.metadata
design['Face_Effect'] = np.where(design['Face'] == 'A', 1, 0)
design = design.assign(Intercept=1)
names = ['Intercept', 'Face_Effect', 'Noise']
# check design matrix
design[names].head()

# fit linear model
reg = linear_regression(limo_epochs, design[names], names=names)

reg['Face_Effect'].beta.plot_joint(title='Face_Effect',
                                   ts_args=dict(time_unit='s'),
                                   topomap_args=dict(time_unit='s'),
                                   times=[.16])

reg['Noise'].beta.plot_joint(title='Effect of Noise',
                             ts_args=dict(time_unit='s'),
                             topomap_args=dict(time_unit='s'),
                             times=[.125, .225])
