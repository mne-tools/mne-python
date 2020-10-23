"""
.. _tut_working_with_seeg:

======================
Working with sEEG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
stereoelectroencephalography (sEEG) data.

This example shows how to use:

- sEEG data
- channel locations in MNI space
- projection into a volume

For an example that involves ECoG data, channel locations in a
subject-specific MRI, or projection into a surface, see
:ref:`tut-working-with-ecog`. In the ECoG example, we show
how to visualize surface grid channels on the brain.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
import pandas as pd

import mne
from mne.datasets import fetch_fsaverage

print(__doc__)

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subject = 'fsaverage'
subjects_dir = sample_path + '/subjects'

###############################################################################
# Let's load some sEEG electrode locations and names, and turn them into
# a :class:`mne.channels.DigMontage` class. First, use pandas to read in the
# ``.tsv`` file.

# In mne-python, the electrode coordinates are required to be in meters
elec_df = pd.read_csv(misc_path + '/seeg/sample_seeg_electrodes.tsv',
                      sep='\t', header=0, index_col=None)
ch_names = elec_df['name'].tolist()

# the test channel coordinates were in mm, so we conver them to meters
ch_coords = elec_df[['x', 'y', 'z']].to_numpy(dtype=float) / 1000.
ch_pos = dict(zip(ch_names, ch_coords))
# Ideally the nasion/LPA/RPA will also be present from the digitization, here
# we use fiducials estimated from the subject's FreeSurfer MNI transformation:
lpa, nasion, rpa = mne.coreg.get_mni_fiducials(
    subject, subjects_dir=subjects_dir)
lpa, nasion, rpa = lpa['r'], nasion['r'], rpa['r']

###############################################################################
# Now we make a :class:`mne.channels.DigMontage` stating that the sEEG
# contacts are in the FreeSurfer surface RAS (i.e., MRI) coordinate system
# for the given subject. Keep in mind that ``fsaverage`` is special in that
# it is already in MNI space.

coord_frame = 'mri'
montage = mne.channels.make_dig_montage(
    ch_pos, coord_frame=coord_frame, nasion=nasion, lpa=lpa, rpa=rpa)
print('Created %s channel positions' % len(ch_names))

###############################################################################
# Now we get the :term:`trans` that transforms from our MRI coordinate system
# to the head coordinate frame. This transform will be applied to the
# data when applying the montage so that standard plotting functions like
# :func:`mne.viz.plot_evoked_topomap` will be aligned properly.

trans = mne.channels.compute_native_head_t(montage)
print(trans)

###############################################################################
# Now that we have our montage, we can load in our corresponding
# time-series data and set the montage to the raw data.

# first we'll load in the sample dataset
raw = mne.io.read_raw_edf(misc_path + '/seeg/sample_seeg.edf')

# drop bad channels
raw.info['bads'].extend([ch for ch in raw.ch_names if ch not in ch_names])
raw.load_data()
raw.drop_channels(raw.info['bads'])
raw.crop(0, 2)  # just process 2 sec of data for speed

# attach montage
raw.set_montage(montage)

# set channel types to sEEG (instead of EEG)
raw.set_channel_types({ch_name: 'seeg' for ch_name in raw.ch_names})

###############################################################################
# Next, we'll get the raw data and plot it's amplitude over time.

raw_lfp = raw.get_data()
raw.plot()

###############################################################################
# We can visualize this raw data on the brain as a heatmap.
# We will use the ``fsaverage`` volume.

# sphinx_gallery_thumbnail_number = 4

# setup a volume-based source space here
# get standard fsaverage volume source space
fetch_fsaverage(subjects_dir=subjects_dir)  # downloads it if necessary
fname_src = op.join(subjects_dir, 'fsaverage', 'bem',
                    'fsaverage-vol-5-src.fif')
vol_src = mne.read_source_spaces(fname_src)

evoked = mne.EvokedArray(raw_lfp, raw.info)
print(evoked.data.shape)
stc = mne.stc_near_sensors(evoked, trans, subject, subjects_dir=subjects_dir,
                           src=vol_src, mode='nearest')
print(vol_src)
print(evoked)
print(stc)
vmin, vmax = np.percentile(raw_lfp.flatten(), [10, 90])
clim = dict(kind='value', lims=[vmin * 0.9, vmin, vmax])
brain = stc.plot(src=vol_src, mode='stat_map', initial_time=0.68,
                 colormap='viridis', clim=clim,
                 subjects_dir=subjects_dir)
# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=20, tmin=0.62, tmax=0.72,
#                  interpolation='linear', framerate=5,
#                  time_viewer=True)
