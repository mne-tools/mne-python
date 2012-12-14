"""
=========================================================================
Compute source space connectivity and visualize it using a circular graph
=========================================================================

This example computes the all-to-all connectivity between 68 regions in
source space based on dSPM inverse solutions and a FreeSurfer cortical
parcellation. The connectivity is visualized using a circular graph which
is ordered based on the locations of the regions.
"""

# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD (3-clause)

print __doc__

import numpy as np
import mne
from mne.datasets import sample
from mne.fiff import Raw, pick_types
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity

data_path = sample.data_path('..')
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Load data
inverse_operator = read_inverse_operator(fname_inv)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# Set up pick list
include = []
exclude = raw.info['bads'] + ['EEG 053']  # bads + 1 more

# Pick MEG channels
picks = pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                   include=include, exclude=exclude)

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_normal=True, return_generator=True)

# Get lables for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.labels_from_parc('sample', parc='aparc',
                              subjects_dir=subjects_dir)

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator['src']
label_ts = mne.extract_label_ts_stcs(stcs, labels, src, mode='mean_flip',
                                     return_generator=True)

# Now we are ready to compute the connectivity in the alpha band. Notice
# from the status messages, how mne-python: 1) reads an epoch from the raw
# file, 2) applies SSP and baseline correction, 3) computes the inverse to
# obtain a source estimate, 4) averages the source estimate to obtain a
# time series for each label, 5) includes the label time series in the
# connectivity computation, and then moves to the next epoch. This
# behaviour is because we are using generators and allows us to
# compute connectivity in computationally efficient manner where the amount
# of memory (RAM) needed is independent from the number of epochs.
fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq']  # the sampling frequency

con, freqs, times, n_epochs, n_tapers = spectral_connectivity(label_ts,
        method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=2)

# Visualize the connectivity as a circular graph
import pylab as pl
import matplotlib.path as m_path
import matplotlib.patches as m_patches

# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]

lh_labels = [name for name in label_names if name.endswith('lh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (ypos, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# Save the plot order
label_plot_order = list()
label_plot_order.extend(lh_labels[::-1])  # reverse the order
label_plot_order.extend(rh_labels)

# Make figure background the same colors as axes
fig = pl.figure(figsize=(8, 8), facecolor='black')

# Use a polar axes
axes = pl.subplot(111, polar=True, axisbg='black')

# No ticks, we'll put our own
pl.xticks([])
pl.yticks([])

# Set y axes limit
pl.ylim(0, 10)

# Ring color from y=9 to y=10
n = len(labels)
theta = np.arange(np.pi / n, 2 * np.pi, 2 * np.pi / n)
radii = np.ones(n) * 10
width = 2 * np.pi / n
bars = axes.bar(theta, radii, width=width, bottom=9, edgecolor='w',
                lw=2, facecolor='.9')
for i, bar in enumerate(bars):
    bar.set_facecolor(pl.cm.jet(i / float(n)))

# Draw lines between connected nodes, only draw the 200 strongest connections
con_thresh = np.sort(con.ravel())[-200]

con_links = list()
for i in range(n):
    con_i = label_names.index(label_plot_order[i])
    for j in range(i):
        if i == j:
            continue

        con_j = label_names.index(label_plot_order[j])

        # Get the connection strength from lower-triangular part of the matrix
        con_val = con[max(con_i, con_j), min(con_i, con_j), 0]
        if con_val < con_thresh:
            continue  # connection is too weak, exclude it

        # Start point
        t0, r0 = np.pi / 2 + i / float(n) * 2 * np.pi, 8.8

        # End point
        t1, r1 = np.pi / 2 + j / float(n) * 2 * np.pi, 8.8

        # Some noise in start and end point
        t0 += .5 * np.random.uniform(-np.pi / n, +np.pi / n)
        t1 += .5 * np.random.uniform(-np.pi / n, +np.pi / n)

        # save the link for plotting later
        con_links.append((con_val, t0, r0, t1, r1))

# Sort the links in ascending order of connection strength, such that the
# strongest connections are drawn last
con_links.sort()

# Draw them
linewidth = 1.5
alpha = 1.
for link in con_links:
    con_val, t0, r0, t1, r1 = link
    verts = [(t0, r0), (t1, 5), (t1, r1)]
    codes = [m_path.Path.MOVETO, m_path.Path.CURVE3, m_path.Path.LINETO]
    path = m_path.Path(verts, codes)

    color = pl.cm.gist_heat((con_val - con_thresh) / (np.max(con) - con_thresh))

    # Actual line
    patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                linewidth=linewidth, alpha=alpha)
    axes.add_patch(patch)

# Put text labels. First, we measure the unit in screen coordinates
x0, y0 = axes.transData.transform_point((0.0, 0.0))
x1, y1 = axes.transData.transform_point((0.0, 1.0))
unit = float(x1 - x0)

pl.ion()
for i in range(len(labels)):
    angle_rad = np.pi / 2 + i / float(len(labels)) * 2 * np.pi
    angle_deg = 90 + i / float(len(labels)) * 360
    label = pl.text(angle_rad, 11., label_plot_order[i], size=10, rotation=0,
                    horizontalalignment='center', verticalalignment='center',
                    color='white')

    # To get text measure, we have to draw it first
    pl.draw()

    # Compute the text extent in data coordinate
    w = label.get_window_extent().width / unit

    # Adjust anchor point and angle
    label.set_y(11. + w / 2.0)

    if angle_deg < 270:
        # Flip the label, so text is always upright
        angle_deg += 180

    label.set_rotation(angle_deg)

pl.ioff()

pl.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)

# Done
pl.savefig('circle.png', facecolor='black')
pl.show()
