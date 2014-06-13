"""
===========================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
===========================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.

See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]

[1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Romain Trachel <romain.trachel@inria.fr>
#          Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

print(__doc__)
import numpy as np

try:
    # Python 3
    from urllib.request import urlretrieve as urlretrieve
except ImportError:
    # Python 2.7
    from urllib import urlretrieve as urlretrieve
import os

import mne
from mne import fiff
from mne.io.edf import read_raw_edf
from mne.datasets import sample
from mne.event import find_events
from mne.decoding import CSP
from mne.layouts import read_layout


data_path = sample.data_path()


def get_data(subject, runs=[6, 10, 14]):
    """Return full path to data file and download if necessary."""
    datapath = ['S{s:03d}R{r:02d}.edf'.format(s=subject, r=r) for r in runs]
    for p, r in zip(datapath, runs):
        if not os.path.exists(p):
            url = ('http://www.physionet.org/physiobank/database/eegmmidb/'
                   'S{s:03d}/S{s:03d}R{r:02d}.edf').format(s=subject, r=r)
            print('downloading {f} from {u}...'.format(f=p, u=url))
            urlretrieve(url, p)
    return datapath


class LogCSP(CSP):
    """This class replaces the CSP's normalized bandpower features with
    log-bandpower features.

    Averaged band power is chi-square distributed. Taking the logarithm brings
    it closer to the normal distribution, which improves LDA classification.
    """
    def __init__(self, *args, **kwargs):
        self = CSP.__init__(self, *args, **kwargs)

    def transform(self, epochs_data, y=None):
        X = CSP.transform(self, epochs_data, y)
        # undo normalization
        X *= self.std_
        X += self.mean_
        # calculate log-bandpower features
        X = np.log(X)
        return X


###############################################################################
## Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1, 4
event_id = dict(hands=2, feet=3)
subject = 1
#runs = [3, 7, 11]  # motor execution: left hand vs right hand
#runs = [4, 8, 12]  # motor imagery: left hand vs right hand
#runs = [5, 9, 13]  # motor execution: hands vs feet
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = get_data(subject, runs)

# Read raw data from EDF files
raw = None
for fname in raw_fnames:
    raw0 = read_raw_edf(fname, tal_channel=-1, preload=True)
    try:
        raw.append(raw0)
    except AttributeError:
        raw = raw0

# Apply band-pass filter
# Hack: fake filter information so raw.filter won't throw an exception at us.
raw.info['lowpass'] = raw.info['sfreq'] * 0.5
raw.info['highpass'] = 0
raw.filter(7, 30, method='iir')

print(raw.info['ch_names'])
raw.info['ch_names'] = [chn.strip('.').upper() for chn in raw.info['ch_names']]

layout = read_layout('../../examples/decoding/EEG1005.lay')
layout.names = [chn.upper() for chn in layout.names]

events = find_events(raw, output='onset', shortest_event=0,
                     stim_channel='STI 014')

picks = fiff.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True)

labels = epochs.events[:, -1] - 2
evoked = epochs.average()



###############################################################################
# Classification with linear discrimant analysis

from sklearn.lda import LDA
from sklearn.cross_validation import LeaveOneOut, ShuffleSplit

n_components = 4  # pick some components
svc = LDA()
csp = LogCSP(n_components=n_components, reg='lws')

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = epochs.get_data()

for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx], y_train)
    X_test = csp.transform(epochs_data[test_idx])

    # fit classifier
    svc.fit(X_train, y_train)

    scores.append(svc.score(X_test, y_test))

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# # Or use much more convenient scikit-learn cross_val_score function using
# # a Pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.cross_validation import cross_val_score
# cv = LeaveOneOut(len(labels))
# clf = Pipeline([('CSP', csp), ('SVC', svc)])
# scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
# print(scores.mean())  # should match results above
#
# # And using regularized csp with Ledoit-Wolf estimator
# csp = CSP(n_components=n_components, reg='lws')
# clf = Pipeline([('CSP', csp), ('SVC', svc)])
# scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
# print(scores.mean())  # should get better results than above

# can't use MNE's topomap (yet?) due to missing electrode positions.
from eegtopo.topoplot import Topoplot
from eegtopo.eegpos3d import positions
import matplotlib.pyplot as plt

scotlabels = {k.lower(): k for k in positions.keys()}
ch_locs = []
for ch_name in raw.info['ch_names'][:-1]:
    ch = ch_name.strip('.')
    ch = scotlabels[ch.lower()]
    ch_locs.append(list(positions[ch].vector))
ch_locs = np.array(ch_locs)

csp.fit(epochs_data, labels)
mpx = np.max(np.abs(csp.patterns_))

topo = Topoplot()
topo.set_locations(ch_locs)
for i, j in enumerate([0, 1, 2, 3, 60, 61, 62, 63]):
    plt.subplot(2, 4, i + 1)

    pattern = csp.patterns_[j, :]

    topo.set_values(pattern)
    topo.create_map()

    topo.plot_map(crange=[-mpx, mpx])
    topo.plot_locations()
    topo.plot_head()

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)
evoked.data = csp.patterns_.T
evoked.times = np.arange(evoked.data.shape[0])
evoked.plot_topomap(times=[0, 1, 2, 3, 60, 61, 62, 63], ch_type='eeg', layout=layout,
                    colorbar=False, size=1.5)

plt.show()