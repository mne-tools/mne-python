"""
==============================================================
Show all-to-all connectivity in sensor space in circular graph
==============================================================

XXX

"""

# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#         Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu
#         Nicolas P. Rougier (Code borrowed from its matplotlib gallery)
#
# License: BSD (3-clause)

print __doc__

import numpy as np

import mne
from mne import fiff
from mne.connectivity import spectral_connectivity
from mne.datasets import sample

###############################################################################
# Set parameters
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

# Set up pick list
exclude = raw.info['bads'] + ['MEG 2443']  # bads + 1 more

# Pick MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                        exclude=exclude)

# Create epochs for the visual condition
event_id, tmin, tmax = 3, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))

# Compute connectivity for band containing the evoked response.
# We exclude the baseline period
fmin, fmax = 3., 9.
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs,
    method='pli', mode='multitaper', sfreq=sfreq,
    fmin=fmin, fmax=fmax, faverage=True, tmin=tmin,
    mt_adaptive=False, n_jobs=2)

# the epochs contain an EOG channel, which we remove now
ch_names = epochs.ch_names
idx = [ch_names.index(name) for name in ch_names if name.startswith('MEG')]
con = con[idx][:, idx]

# con is a 3D array where the last dimension is size one since we averaged
# over frequencies in a single band. Here we make it 2D
con = con[:, :, 0]

import pylab as pl
import matplotlib.path as m_path
import matplotlib.patches as m_patches

# Data to be represented
# ----------
n = 50
con = con[:n][:, :n]
ch_names = ch_names[:n]
n = con.shape[0]
labels = ch_names
links = con
# ----------

# Make figure background the same colors as axes
fig = pl.figure(figsize=(8, 8), facecolor='white')

# Use a polar axes
axes = pl.subplot(111, polar=True)

# No ticks, we'll put our own
pl.xticks([])
pl.yticks([])

# Set y axes limit
pl.ylim(0, 10)

# Ring color from y=9 to y=10
theta = np.arange(np.pi / n, 2 * np.pi, 2 * np.pi / n)
radii = np.ones(n) * 10
width = 2 * np.pi / n
bars = axes.bar(theta, radii, width=width, bottom=9, edgecolor='w',
                 lw=2, facecolor='.9')
for i, bar in enumerate(bars):
    bar.set_facecolor(pl.cm.jet(i / float(n)))

linewidth = 1.
alpha = .5

# Draw lines between connected nodes
for i in range(n):
    for j in range(n):
        if links[i, j] == 0.:
            continue

        # Start point
        t0, r0 = i / float(n) * 2 * np.pi, 9

        # End point
        t1, r1 = j / float(n) * 2 * np.pi, 9

        # Some noise in start and end point
        t0 += .5 * np.random.uniform(-np.pi / n, +np.pi / n)
        t1 += .5 * np.random.uniform(-np.pi / n, +np.pi / n)

        verts = [(t0, r0), (t1, 5), (t1, r1)]
        codes = [m_path.Path.MOVETO, m_path.Path.CURVE3, m_path.Path.LINETO]
        path = m_path.Path(verts, codes)

        color = pl.cm.Reds(links[i, j] / np.max(links))

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth, alpha=alpha)
        axes.add_patch(patch)


# First, we measure the unit in screen coordinates
x0, y0 = axes.transData.transform_point((0.0, 0.0))
x1, y1 = axes.transData.transform_point((0.0, 1.0))
unit = float(x1 - x0)


pl.ion()
for i in range(len(labels)):
    angle_rad = i / float(len(labels)) * 2 * np.pi
    angle_deg = i / float(len(labels)) * 360
    label = pl.text(angle_rad, 10.5, labels[i], size=10, rotation=0,
                    horizontalalignment='center', verticalalignment="center")

    # To get text measure, we have to draw it first
    pl.draw()

    # Compute the text extent in data coordinate
    w = label.get_window_extent().width / unit

    # Adjust anchor point and angle
    label.set_y(10.5 + w / 2.0)
    label.set_rotation(angle_deg)
pl.ioff()

pl.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)

# Done
pl.savefig('circle.png', facecolor='white')
pl.show()
