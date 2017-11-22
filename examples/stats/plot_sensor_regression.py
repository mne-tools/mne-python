"""
=====================================
Sensor space least squares regression
=====================================

Predict single trial activity from a continuous variable.
A single-trial regression is performed in each sensor and timepoint
individually, resulting in an :class:`mne.Evoked` object which contains the
regression coefficient (beta value) for each combination of sensor and
timepoint. Example shows the regression coefficient; the t and p values are
also calculated automatically.

Here, we repeat a few of the analyses from [1]_ by accessing the metadata
object, which contains word-level information about various
psycholinguistically relevant features of the words for which we have EEG
activity.

For the general methodology, see e.g. [2]_


References
----------
.. [1]  Dufau, S., Grainger, J., Midgley, KJ., Holcomb, PJ. A thousand
   words are worth a picture: Snapshots of printed-word processing in an
   event-related potential megastudy. Psychological Science, 2015
.. [2]  Hauk et al. The time course of visual word recognition as revealed by
   linear regression analysis of ERP data. Neuroimage, 2006
"""
# Authors: Tal Linzen <linzen@nyu.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#
# License: BSD (3-clause)

import pandas as pd
import mne
from mne.stats import linear_regression
from mne.datasets import kiloword

# Load the data
path = kiloword.data_path() + '/kword_metadata-epo.fif'
epochs = mne.read_epochs(path)
print(epochs.metadata.head())

# Add intercept column
df = pd.DataFrame(epochs.metadata)
epochs.metadata = df.assign(Intercept=[1 for _ in epochs.events])

# Run and visualize the regression
names = ["Intercept", "Concreteness", "BigramFrequency"]
res = linear_regression(epochs, epochs.metadata[names], names=names)

for cond in names:
    res[cond].beta.plot_joint(title=cond)
