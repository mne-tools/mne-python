# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'mne logo'
===============================================================================

This script makes the logo for MNE.
"""
# @author: drmccloy
# Created on Mon Jul 20 11:28:16 2015
# License: BSD (3-clause)

import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.mlab import bivariate_normal
from matplotlib.path import Path
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox

# manually set values
dpi = 72.
center_fudge = np.array([2, 0])  # compensate for font bounding box padding
tagline_scale_fudge = 0.98  # to get justification right
tagline_offset_fudge = np.array([0.4, 0])

static_dir = op.join('..', 'doc', '_static')

# font, etc
rcp = {'font.sans-serif': ['Primetime'], 'font.style': 'normal',
       'font.weight': 'black', 'font.variant': 'normal', 'figure.dpi': dpi,
       'savefig.dpi': dpi, 'contour.negative_linestyle': 'solid'}
plt.rcdefaults()
rcParams.update(rcp)

# initialize figure (no axes, margins, etc)
fig = plt.figure(1, figsize=(5, 3), frameon=False, dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# fake field data
delta = 0.1
x = np.arange(-8.0, 8.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 8.0, 7.0, -5.0, 0.9, 1.0)
Z2 = bivariate_normal(X, Y, 15.0, 2.5, 2.6, -2.5, 2.5)
Z = Z2 - 0.7 * Z1

# color map: field gradient (yellow-red-gray-blue-cyan)
# yrtbc = {
#     'red': ((0, 1, 1), (0.4, 1, 1), (0.5, 0.5, 0.5), (0.6, 0, 0), (1, 0, 0)),
#     'blue': ((0, 0, 0), (0.4, 0, 0), (0.5, 0.5, 0.5), (0.6, 1, 1), (1, 1, 1)),  # noqa
#     'green': ((0, 1, 1), (0.4, 0, 0), (0.5, 0.5, 0.5), (0.6, 0, 0), (1, 1, 1)),  # noqa
# }
yrtbc = {'red': ((0.0, 1.0, 1.0), (0.5, 1.0, 0.0), (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0), (0.5, 0.0, 1.0), (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
         'alpha': ((0.0, 1.0, 1.0), (0.4, 0.8, 0.8), (0.5, 0.2, 0.2),
                   (0.6, 0.8, 0.8), (1.0, 1.0, 1.0))}
# color map: field lines (red | blue)
redbl = {'red': ((0., 1., 1.), (0.5, 1., 0.), (1., 0., 0.)),
         'blue': ((0., 0., 0.), (0.5, 0., 1.), (1., 1., 1.)),
         'green': ((0., 0., 0.), (1., 0., 0.)),
         'alpha': ((0., 0.4, 0.4), (1., 0.4, 0.4))}
mne_field_grad_cols = LinearSegmentedColormap('mne_grad', yrtbc)
mne_field_line_cols = LinearSegmentedColormap('mne_line', redbl)

# plot gradient and contour lines
im = plt.imshow(Z, cmap=mne_field_grad_cols, aspect='equal')
cs = plt.contour(Z, 9, cmap=mne_field_line_cols, linewidths=1)
plot_dims = np.r_[np.diff(ax.get_xbound()), np.diff(ax.get_ybound())]

# create MNE clipping mask
mne_path = TextPath((0, 0), 'MNE')
dims = mne_path.vertices.max(0) - mne_path.vertices.min(0)
vert = mne_path.vertices - dims / 2.
mult = (plot_dims / dims).min()
mult = [mult, -mult]  # y axis is inverted (origin at top left)
offset = plot_dims / 2. - center_fudge
mne_clip = Path(offset + vert * mult, mne_path.codes)
ax.add_patch(PathPatch(mne_clip, color='w', zorder=0, linewidth=0))
# apply clipping mask to field gradient and lines
im.set_clip_path(mne_clip, transform=im.get_transform())
for coll in cs.collections:
    coll.set_clip_path(mne_clip, transform=im.get_transform())
# get final position of clipping mask
mne_corners = mne_clip.get_extents().corners()

# add tagline
rcParams.update({'font.sans-serif': ['Cooper Hewitt'], 'font.weight': 'light'})
tag_path = TextPath((0, 0), 'MEG + EEG  ANALYSIS & VISUALIZATION')
dims = tag_path.vertices.max(0) - tag_path.vertices.min(0)
vert = tag_path.vertices - dims / 2.
mult = tagline_scale_fudge * (plot_dims / dims).min()
mult = [mult, -mult]  # y axis is inverted
offset = mne_corners[-1] - np.array([mne_clip.get_extents().size[0] / 2.,
                                     -dims[1]]) - tagline_offset_fudge
tag_clip = Path(offset + vert * mult, tag_path.codes)
tag_patch = PathPatch(tag_clip, facecolor='k', edgecolor='none', zorder=10)
ax.add_patch(tag_patch)
yl = ax.get_ylim()
yy = np.max([tag_clip.vertices.max(0)[-1],
             tag_clip.vertices.min(0)[-1]])
ax.set_ylim(np.ceil(yy), yl[-1])

# only save actual image extent plus a bit of padding
plt.draw()
plt.savefig(op.join(static_dir, 'mne_logo.png'), transparent=True)
plt.close()

# 92x22 image
w_px = 92
h_px = 22
center_fudge = np.array([12, 0.5])
scale_fudge = 2.1
rcParams.update({'font.sans-serif': ['Primetime'], 'font.weight': 'black'})
x = np.linspace(-8., 8., w_px / 2.)
y = np.linspace(-3., 3., h_px / 2.)
X, Y = np.meshgrid(x, y)
# initialize figure (no axes, margins, etc)
fig = plt.figure(1, figsize=(w_px / dpi, h_px / dpi), frameon=False, dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
# plot rainbow
im = ax.imshow(X, cmap=mne_field_grad_cols, aspect='equal', zorder=1)
im = ax.imshow(np.ones_like(X) * 0.5, cmap='Greys', aspect='equal', zorder=0,
               clim=[0, 1])
plot_dims = np.r_[np.diff(ax.get_xbound()), np.diff(ax.get_ybound())]
# MNE text in white
mne_path = TextPath((0, 0), 'MNE')
dims = mne_path.vertices.max(0) - mne_path.vertices.min(0)
vert = mne_path.vertices - dims / 2.
mult = scale_fudge * (plot_dims / dims).min()
mult = [mult, -mult]  # y axis is inverted (origin at top left)
offset = np.array([scale_fudge, 1.]) * \
    np.array([-dims[0], plot_dims[-1]]) / 2. - center_fudge
mne_clip = Path(offset + vert * mult, mne_path.codes)
mne_patch = PathPatch(mne_clip, facecolor='w', edgecolor='none', zorder=10)
ax.add_patch(mne_patch)
# adjust xlim and ylim
mne_corners = mne_clip.get_extents().corners()
xmin, ymin = np.min(mne_corners, axis=0)
xmax, ymax = np.max(mne_corners, axis=0)
xl = ax.get_xlim()
yl = ax.get_ylim()
xpad = np.abs(np.diff([xmin, xl[1]])) / 20.
ypad = np.abs(np.diff([ymax, ymin])) / 20.
ax.set_xlim(xmin - xpad, xl[1] + xpad)
ax.set_ylim(ymax + ypad, ymin - ypad)
plt.draw()
plt.savefig(op.join(static_dir, 'mne_logo_small.png'), transparent=True)
plt.close()
