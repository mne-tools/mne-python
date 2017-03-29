"""
=========================================
Continuous Target Decoding with SPoC
=========================================

This example reproduces figures from Lalor et al's mTRF toolbox in
matlab [1]_. We will show how the :class:`mne.decoding.ReceptiveField` class
can perform a similar function along with :mod:`sklearn`. We will fit a
linear encoding model using the continuously-varying speech envelope to
predict activity of a 128 channel EEG system.

References
----------
.. [1] Crosse, M. J., Di Liberto, G. M., Bednar, A. & Lalor, E. C. (2016).
       The Multivariate Temporal Response Function (mTRF) Toolbox:
       A MATLAB Toolbox for Relating Neural Signals to Continuous Stimuli.
       Frontiers in Human Neuroscience 10, 604. doi:10.3389/fnhum.2016.00604

.. _figure 1: http://journal.frontiersin.org/article/10.3389/fnhum.2016.00604/full#F1
.. _figure 2: http://journal.frontiersin.org/article/10.3389/fnhum.2016.00604/full#F2
"""  # noqa: E501

# Authors: Chris Holdgraf <choldgraf@gmail.com>
#
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 3

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import SPoC
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
###############################################################################
# Simulations
# ----------------------------------
#
#

# define parameters

n_channels = 64
sfreq = 128.
mod_freq = 0.3
carrier_freq = 6
noise_level = 10

rs = np.random.RandomState(666)

times = np.arange(0, 120, 1 / sfreq)
target = np.sin(2 * np.pi * mod_freq * times) + 2

theta = np.sin(2 * np.pi * carrier_freq * times) * target

A = rs.randn(n_channels, 1)
noise = noise_level * rs.randn(n_channels, len(times))
eeg = np.dot(A, np.atleast_2d(theta)) + noise

window = 0.5  # time window in second
overlap = 0.9  # percent of overlap
n_sample_epochs = int(sfreq * window)
n_sample_overlap = int(n_sample_epochs * (1 - overlap))

indices_start = np.arange(0, len(times) - n_sample_epochs, n_sample_overlap)

X = np.zeros((len(indices_start), n_channels, n_sample_epochs))
y = np.zeros(len(indices_start))
for ii, start in enumerate(indices_start):
    sl = slice(start, start + n_sample_epochs)
    X[ii] = eeg[:, sl]
    y[ii] = target[sl].mean()


clf = make_pipeline(SPoC(n_components=1), Ridge())

cv = KFold(5)
preds = cross_val_predict(clf, X, y, cv=cv)

plt.plot(preds)
plt.plot(y)
plt.legend(['Spoc EEG power', 'Target envelop'])
plt.show()
plt.figure()
