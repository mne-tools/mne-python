"""
================================================================
Demonstration of how to use ClickableImage / generate_2d_layout.
================================================================

In this example, we open an image file, then use ClickableImage to
return 2D locations of mouse clicks (or load a file already created).
Then, we use generate_2d_layout to turn those xy positions into a layout
for use with plotting topo maps. In this way, you can take arbitrary xy
positions and turn them into a plottable layout.
"""
# Authors: Christopher Holdgraf <choldgraf@berkeley.edu>
#
# License: BSD (3-clause)
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
from os import path as op
import mne
from mne.viz import ClickableImage, add_background_image  # noqa
from mne.channels import generate_2d_layout  # noqa

print(__doc__)

# Set parameters and paths
plt.rcParams['image.cmap'] = 'gray'

im_path = op.join(op.dirname(mne.__file__), 'data', 'image', 'mni_brain.gif')
# We've already clicked and exported
layout_path = op.join(op.dirname(mne.__file__), 'data', 'image')
layout_name = 'custom_layout.lout'

###############################################################################
# Load data and click
im = imread(im_path)
plt.imshow(im)
"""
This code opens the image so you can click on it. Commented out
because we've stored the clicks as a layout file already.

# The click coordinates are stored as a list of tuples
click = ClickableImage(im)
click.plot_clicks()
coords = click.coords

# Generate a layout from our clicks and normalize by the image
lt = generate_2d_layout(np.vstack(coords), bg_image=im)
lt.save(layout_path + layout_name)  # To save if we want
"""
# We've already got the layout, load it
lt = mne.channels.read_layout(layout_name, path=layout_path, scale=False)

# Create some fake data
nchans = len(lt.pos)
nepochs = 50
sr = 1000
nsec = 5
events = np.arange(nepochs).reshape([-1, 1])
events = np.hstack([events, np.zeros([nepochs, 2], dtype=int)])
data = np.random.randn(nepochs, nchans, sr * nsec)
info = mne.create_info(nchans, sr, ch_types='eeg')
epochs = mne.EpochsArray(data, info, events)
evoked = epochs.average()

# Using the native plot_topo function with the image plotted in the background
f = evoked.plot_topo(layout=lt, fig_background=im)
