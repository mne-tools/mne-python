"""
.. _tut-fnirs-processing:

Preprocessing functional near-infrared spectroscopy (fNIRS) data
================================================================

This tutorial covers how to convert functional near-infrared spectroscopy
(fNIRS) data from raw measurements to relative oxyhaemoglobin (HbO) and
deoxyhaemoglobin (HbR) concentration.

.. contents:: Page contents
   :local:
   :depth: 2

Here we will work with the :ref:`fNIRS motor data <fnirs-motor-dataset>`.
"""
# sphinx_gallery_thumbnail_number = 3

import os
import numpy as np
import matplotlib.pyplot as plt

import mne


fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()


###############################################################################
# Selecting channels appropriate for detecting neural responses
# -------------------------------------------------------------
#
# First we remove channels that are too close together to detect a neural
# response. To achieve this we pick all the channels that are not considered
# to be short (less than 1 cm distance between optodes).

picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks)
raw_intensity.pick(picks[dists > 0.01])
raw_intensity.plot(n_channels=len(raw_intensity.ch_names), duration=500)


###############################################################################
# Converting from raw intensity to optical density
# ------------------------------------------------
#
# The raw intensity values are then converted to optical density.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od.plot(n_channels=len(raw_od.ch_names), duration=500)


###############################################################################
# Converting from optical density to haemoglobin
# ----------------------------------------------
#
# Next we convert the optical density data to haemoglobin concentration using
# the modified Beer-Lambert law.

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=500)


###############################################################################
# Removing heart rate from signal
# -------------------------------
#
# The haemodynamic response has frequency content predominantly below 0.5 Hz.
# An increase in activity around 1 Hz can be seen that is due to the heart beat
# and is unwanted. So we use a low pass filter to remove this.
# A high pass filter is included to remove slow drifts in the data.

fig = raw_haemo.plot_psd(average=True)
fig.suptitle('Before filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)
raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)
fig = raw_haemo.plot_psd(average=True)
fig.suptitle('After filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)

###############################################################################
# Extract epochs
# --------------
#
# Now that the signal has been converted to relative haemoglobin concentration,
# and the unwanted heart rate component has been removed, we can extract epochs
# related to each of the experimental conditions.
#
# First we extract the events of interest and visualise them to ensure they are
# correct.

events, _ = mne.events_from_annotations(raw_haemo, event_id={'1.0': 1,
                                                             '2.0': 2,
                                                             '3.0': 3})
event_dict = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}
fig = mne.viz.plot_events(events, event_id=event_dict,
                          sfreq=raw_haemo.info['sfreq'])
fig.subplots_adjust(right=0.7)  # make room for the legend


###############################################################################
# Next we define the range of our epochs, the rejection criteria,
# baseline correction, and extract the epochs. We visualise the log of which
# epochs were dropped.

reject_criteria = dict(hbo=80e-6)
tmin, tmax = -5, 15

epochs = mne.Epochs(raw_haemo, events, event_id=event_dict,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria, reject_by_annotation=True,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True)
epochs.plot_drop_log()


###############################################################################
# View consistency of responses across trials
# -------------------------------------------
#
# Now we can view the haemodynamic response for our tapping condition.
# We visualise the response for both the oxy- and deoxyhaemoglobin, and
# observe the expected peak in HbO at around 6 seconds consistently across
# trials, and the consistent dip in HbR that is slightly delayed relative to
# the HbO peak.

epochs['Tapping'].plot_image(combine='mean', vmin=-30, vmax=30,
                             ts_args=dict(ylim=dict(hbo=[-15, 15],
                                                    hbr=[-15, 15])))


###############################################################################
# We can also view the epoched data for the control condition and observe
# that it does not show the expected morphology.

epochs['Control'].plot_image(combine='mean', vmin=-30, vmax=30,
                             ts_args=dict(ylim=dict(hbo=[-15, 15],
                                                    hbr=[-15, 15])))


###############################################################################
# View consistency of responses across channels
# ---------------------------------------------
#
# Similarly we can view how consistent the response was across the optode
# pairs that we selected. All the channels in this data were located over the
# motor cortex, and all channels show a similar pattern in the data.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
clims = dict(hbo=[-20, 20], hbr=[-20, 20])
epochs['Control'].average().plot_image(axes=axes[:, 0], clim=clims)
epochs['Tapping'].average().plot_image(axes=axes[:, 1], clim=clims)
for column, condition in enumerate(['Control', 'Tapping']):
    for ax in axes[:, column]:
        ax.set_title('{}: {}'.format(condition, ax.get_title()))


###############################################################################
# Plot standard fNIRS response image
# ----------------------------------
#
# Next we generate the most common visualisation of fNIRS data, plotting
# both the HbO and HbR on the same figure to illustrate the relation between
# the two signals.

evoked_dict = {'Tapping/HbO': epochs['Tapping'].average(picks='hbo'),
               'Tapping/HbR': epochs['Tapping'].average(picks='hbr'),
               'Control/HbO': epochs['Control'].average(picks='hbo'),
               'Control/HbR': epochs['Control'].average(picks='hbr')}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO='#AA3377', HbR='b')
styles_dict = dict(Control=dict(linestyle='dashed'))

mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.95,
                             colors=color_dict, styles=styles_dict)


###############################################################################
# View topographic representation of activity
# -------------------------------------------
#
# Next we view how the topographic activity changes throughout the response.

times = np.arange(-3.5, 13.2, 3.0)
topomap_args = dict(extrapolate='local')
epochs['Tapping'].average(picks='hbo').plot_joint(
    times=times, topomap_args=topomap_args)


###############################################################################
# Compare tapping of left and right hands
# ---------------------------------------
#
# Finally we generate topo maps for the left and right conditions to view
# the location of activity. First we visualise the HbO activity.

times = np.arange(4.0, 11.0, 1.0)
epochs['Tapping/Left'].average(picks='hbo').plot_topomap(
    times=times, **topomap_args)
epochs['Tapping/Right'].average(picks='hbo').plot_topomap(
    times=times, **topomap_args)

###############################################################################
# And we also view the HbR activity for the two conditions.

epochs['Tapping/Left'].average(picks='hbr').plot_topomap(
    times=times, **topomap_args)
epochs['Tapping/Right'].average(picks='hbr').plot_topomap(
    times=times, **topomap_args)

###############################################################################
# And we can plot the comparison at a single time point for two conditions.

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 5))
vmin, vmax, ts = -8, 8, 9.0

evoked_left = epochs['Tapping/Left'].average()
evoked_right = epochs['Tapping/Right'].average()

evoked_left.plot_topomap(ch_type='hbo', times=ts, axes=axes[0, 0],
                         vmin=vmin, vmax=vmax, colorbar=False,
                         **topomap_args)
evoked_left.plot_topomap(ch_type='hbr', times=ts, axes=axes[1, 0],
                         vmin=vmin, vmax=vmax, colorbar=False,
                         **topomap_args)
evoked_right.plot_topomap(ch_type='hbo', times=ts, axes=axes[0, 1],
                          vmin=vmin, vmax=vmax, colorbar=False,
                          **topomap_args)
evoked_right.plot_topomap(ch_type='hbr', times=ts, axes=axes[1, 1],
                          vmin=vmin, vmax=vmax, colorbar=False,
                          **topomap_args)

evoked_diff = mne.combine_evoked([evoked_left, -evoked_right], weights='equal')

evoked_diff.plot_topomap(ch_type='hbo', times=ts, axes=axes[0, 2],
                         vmin=vmin, vmax=vmax,
                         **topomap_args)
evoked_diff.plot_topomap(ch_type='hbr', times=ts, axes=axes[1, 2],
                         vmin=vmin, vmax=vmax, colorbar=True,
                         **topomap_args)

for column, condition in enumerate(
        ['Tapping Left', 'Tapping Right', 'Left-Right']):
    for row, chroma in enumerate(['HbO', 'HbR']):
        axes[row, column].set_title('{}: {}'.format(chroma, condition))
