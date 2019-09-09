"""
.. _ex-publication-figure:

===================================
Make figures more publication ready
===================================

In this example, we take some MNE plots and make some changes to make
a figure closer to publication ready.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_stc = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-eeg-lh.stc')
fname_evoked = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

evoked = mne.read_evokeds(fname_evoked, 'Left Auditory')
evoked.pick_types(meg='grad').apply_baseline((None, 0.))
max_t = evoked.get_peak()[1]

stc = mne.read_source_estimate(fname_stc)

# Plot the STC, get the brain image, crop it
colormap = 'viridis'
clim = dict(kind='value', lims=[3, 6, 9])
brain = stc.plot(views='lat', hemi='split', background='w', size=(800, 400),
                 subject='sample', subjects_dir=subjects_dir,
                 colorbar=False, initial_time=max_t, clim=clim,
                 colormap=colormap)
brain_image = brain.screenshot()
# Crop to rectangle that has useful informamtion
brain_image = brain_image[(brain_image != 255).any(-1).any(1)]
brain_image = brain_image[:, (brain_image != 255).any(-1).any(0)]
brain.close()

# Tweak the style
plt.rcParams.update({
    'ytick.labelsize': 'small',
    'xtick.labelsize': 'small',
    'axes.labelsize': 'small',
    'axes.titlesize': 'medium',
    'grid.color': '0.75',
    'grid.linestyle': ':',
})

# Create a figure of the desired size
figsize = (4.5, 3.0)
fig = plt.figure(figsize=figsize)
axes = [
    plt.subplot2grid((2, 1), (0, 0)),
    plt.subplot2grid((2, 1), (1, 0)),
]
ev_idx = 0
br_idx = 1
evoked.plot(axes=axes[ev_idx:ev_idx + 1])
peak_line = axes[ev_idx].axvline(max_t, color=(0., 1., 0.), ls='--')
axes[ev_idx].legend(
    [axes[ev_idx].lines[0], peak_line], ['MEG data', 'Peak time'],
    frameon=True, columnspacing=0.1, labelspacing=0.1,
    fontsize=8, fancybox=True, handlelength=2.0)
axes[ev_idx].texts = []  # remove Nave
# Remove spines and add grid
axes[ev_idx].grid(True, zorder=2)
for key in ('top', 'right'):
    axes[ev_idx].spines[key].set(visible=False)

# Brain
axes[br_idx].imshow(brain_image)
axes[br_idx].axis('off')

# Add colorbar (should eventually be a function probably)
divider = make_axes_locatable(axes[br_idx])
cax = divider.append_axes('right', size='5%', pad=0.2)
cmap, scale_pts, diverging, _ = mne.viz._3d._limits_to_control_points(
    clim, 0, colormap, transparent=True, linearize=True)
vmin, vmax = scale_pts[0], scale_pts[-1]
ticks = clim['lims']
norm = Normalize(vmin=vmin, vmax=vmax)
cbar = ColorbarBase(cax, cmap, norm=norm, ticks=ticks,
                    label='Activation (F)', orientation='vertical')
for key in ('left', 'top', 'bottom'):
    cax.spines[key].set_visible(False)
cbar.patch.set(facecolor='0.5', edgecolor='0.5')
cbar.outline.set_visible(False)

# Add subplot labels
for ax, label in zip(axes, 'AB'):
    # fudge factor due to colorbar shrinking transAxes
    factor = 1.125 if ax is axes[br_idx] else 1.
    ax.text(-0.175 * factor, 1.1, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

fig.subplots_adjust(
    left=0.15, right=0.9, bottom=0.01, top=0.9, wspace=0.1, hspace=0.5)
