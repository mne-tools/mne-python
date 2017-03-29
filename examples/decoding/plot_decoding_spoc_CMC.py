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
# ftp://ftp.fieldtriptoolbox.org/pub/fieldtrip/tutorial/SubjectCMC.zip
raw = mne.io.read_raw_ctf('/home/kirsh/Documents/Data/fieldtrip/CMC/SubjectCMC.ds', preload=True)
raw.crop(50, 350)
emg = raw.copy().pick_channels(['EMGlft'])
emg.filter(20, 100)

raw.pick_types(meg=True)
raw.filter(15, 30, method='iir')

sfreq = raw.info['sfreq']
n_channels = len(raw.ch_names)


window = 0.5  # time window in second
overlap = 0.2  # percent of overlap
n_sample_epochs = int(sfreq * window)
n_sample_overlap = int(n_sample_epochs * (1 - overlap))

indices_start = np.arange(0, len(raw) - n_sample_epochs, n_sample_overlap)

X = np.zeros((len(indices_start), n_channels, n_sample_epochs))
y = np.zeros(len(indices_start))
for ii, start in enumerate(indices_start):
    sl = slice(start, start + n_sample_epochs)
    X[ii] = raw._data[:, sl]
    y[ii] = np.log(emg._data[0, sl].var())


spoc = SPoC(n_components=4)
spoc.fit(X, y)
spoc.plot_patterns(raw.info, components=[0, 1, 181, 182])


clf = make_pipeline(SPoC(5, log=True), Ridge())

cv = KFold(5)
preds = cross_val_predict(clf, X, y, cv=cv)

plt.figure()
plt.plot(preds)
plt.plot(y)
plt.legend(['Spoc EEG power', 'Target envelop'])
plt.show()


from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import SPoC

clf = make_pipeline(Covariances(), SPoC(5, log=True), Ridge())

cv = KFold(5)
preds = cross_val_predict(clf, X, y, cv=cv)

plt.figure()
plt.plot(preds)
plt.plot(y)
plt.legend(['Spoc EEG power', 'Target envelop'])
plt.show()
