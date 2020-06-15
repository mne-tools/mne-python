"""
.. _tut-fnirs-processing:

Preprocessing optical imaging data from the Imagent hardware/boxy software
================================================================

This tutorial covers how to convert optical imaging data from raw measurements
to relative oxyhaemoglobin (HbO) and deoxyhaemoglobin (HbR) concentration.
Phase data from the recording is also processed and plotted in several ways.

 .. contents:: Page contents
    :local:
    :depth: 2

 Here we will work with the :ref:`fNIRS motor data <fnirs-motor-dataset>`.
"""
# sphinx_gallery_thumbnail_number = 1

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
import re as re

import mne

# get our data
boxy_data_folder = mne.datasets.boxy_example.data_path()
boxy_raw_dir = os.path.join(boxy_data_folder, 'Participant-1')

# load AC and Phase data
raw_intensity_ac = mne.io.read_raw_boxy(boxy_raw_dir, 'AC',
                                        verbose=True).load_data()

raw_intensity_ph = mne.io.read_raw_boxy(boxy_raw_dir, 'Ph',
                                        verbose=True).load_data()

# get channel indices for our two montages
mtg_a = [raw_intensity_ac.ch_names[i_index] for i_index, i_label
         in enumerate(raw_intensity_ac.info['ch_names'])
         if re.search(r'S[1-5]_', i_label)]

mtg_b = [raw_intensity_ac.ch_names[i_index] for i_index, i_label
         in enumerate(raw_intensity_ac.info['ch_names'])
         if re.search(r'S([6-9]|10)_', i_label)]

# plot the raw data for each data type
# AC
scalings = dict(fnirs_raw=1e2)
raw_intensity_ac.plot(n_channels=5, duration=20, scalings=scalings,
                      show_scrollbars=True)

# Phase
scalings = dict(fnirs_ph=1e4)
raw_intensity_ph.plot(n_channels=5, duration=20, scalings=scalings,
                      show_scrollbars=True)

# ###############################################################################
# # View location of sensors over brain surface
# # -------------------------------------------
# #
# # Here we validate that the location of sources-detector pairs and channels
# # are in the expected locations. Sources are bright red dots, detectors are
# # dark red dots, with source-detector pairs connected by white lines.

subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage())

# plot both montages together
fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(raw_intensity_ac.info,
                             show_axes=True,
                             subject='fsaverage',
                             trans='fsaverage',
                             surfaces=['head-dense', 'brain'],
                             fnirs=['sources', 'detectors', 'pairs'],
                             mri_fiducials=True,
                             dig=True,
                             subjects_dir=subjects_dir,
                             fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=20, elevation=55, distance=0.6)

# plot montage A only
fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(raw_intensity_ac.copy().pick_channels(mtg_a).info,
                             show_axes=True,
                             subject='fsaverage',
                             trans='fsaverage',
                             surfaces=['head-dense', 'brain'],
                             fnirs=['sources', 'detectors', 'pairs'],
                             mri_fiducials=True,
                             dig=True,
                             subjects_dir=subjects_dir,
                             fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=20, elevation=55, distance=0.6)

# plot montage B only
fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(raw_intensity_ac.copy().pick_channels(mtg_b).info,
                             show_axes=True,
                             subject='fsaverage',
                             trans='fsaverage',
                             surfaces=['head-dense', 'brain'],
                             fnirs=['sources', 'detectors', 'pairs'],
                             mri_fiducials=True,
                             dig=True,
                             subjects_dir=subjects_dir,
                             fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=20, elevation=55, distance=0.6)

# ###############################################################################
# # Selecting channels appropriate for detecting neural responses
# # -------------------------------------------------------------
# #
# # First we remove channels that are too close together (short channels) to
# # detect a neural response (less than 3 cm distance between optodes).
# # These short channels can be seen in the figure above.
# # To achieve this we pick all the channels that are not considered to be short.

picks = mne.pick_types(raw_intensity_ac.info, meg=False, fnirs=True, stim=True)

dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity_ac.info, picks=picks)

raw_intensity_ac.pick(picks[dists < 0.03])


# ###############################################################################
# # Converting from raw intensity to optical density
# # ------------------------------------------------
# #
# # The raw intensity values are then converted to optical density.
# # We will only do this for either DC or AC data since they are measures of
# # light intensity.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity_ac)

raw_od.plot(n_channels=len(raw_od.ch_names),
            duration=500, show_scrollbars=False)

# ###############################################################################
# # Evaluating the quality of the data
# # ----------------------------------
# #
# # At this stage we can quantify the quality of the coupling
# # between the scalp and the optodes using the scalp coupling index. This
# # method looks for the presence of a prominent synchronous signal in the
# # frequency range of cardiac signals across both photodetected signals.
# #
# # In this example the data is clean and the coupling is good for all
# # channels, so we will not mark any channels as bad based on the scalp
# # coupling index.

sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

fig, ax = plt.subplots()
ax.hist(sci)
ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])

# ###############################################################################
# # In this example we will mark all channels with a SCI less than 0.5 as bad
# # (this dataset is quite clean, so no channels are marked as bad).

raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))

# ###############################################################################
# # At this stage it is appropriate to inspect your data
# # (for instructions on how to use the interactive data visualisation tool
# # see :ref:`tut-visualize-raw`)
# # to ensure that channels with poor scalp coupling have been removed.
# # If your data contains lots of artifacts you may decide to apply
# # artifact reduction techniques as described in :ref:`ex-fnirs-artifacts`.


# ###############################################################################
# # Converting from optical density to haemoglobin
# # ----------------------------------------------
# #
# # Next we convert the optical density data to haemoglobin concentration using
# # the modified Beer-Lambert law.

raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)

raw_haemo.plot(n_channels=len(raw_haemo.ch_names),
               duration=500, show_scrollbars=False)

# ###############################################################################
# # Removing heart rate from signal
# # -------------------------------
# #
# # The haemodynamic response has frequency content predominantly below 0.5 Hz.
# # An increase in activity around 1 Hz can be seen in the data that is due to
# # the person's heart beat and is unwanted. So we use a low pass filter to
# # remove this. A high pass filter is also included to remove slow drifts
# # in the data.

fig = raw_haemo.plot_psd(average=True)
fig.suptitle('Before filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)

raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)

fig = raw_haemo.plot_psd(average=True)
fig.suptitle('After filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)

# ###############################################################################
# # Extract epochs
# # --------------
# #
# # Now that the signal has been converted to relative haemoglobin concentration,
# # and the unwanted heart rate component has been removed, we can extract epochs
# # related to each of the experimental conditions.
# #
# # First we extract the events of interest and visualise them to ensure they are
# # correct.

# Since our events and timings for this data set are the same across montages,
# we are going to find events for each montage separately and combine them later

# Montage A Events
mtg_a_events = mne.find_events(raw_intensity_ac, stim_channel=['Markers a'])

mtg_a_event_dict = {'Montage_A/Event_1': 1,
                    'Montage_A/Event_2': 2,
                    'Montage A/Block 1 End': 1000,
                    'Montage A/Block 2 End': 2000}

fig = mne.viz.plot_events(mtg_a_events)
fig.subplots_adjust(right=0.7)  # make room for the legend

raw_intensity_ac.copy().pick_channels(mtg_a).plot(events=mtg_a_events, start=0,
                                                  duration=10, color='gray',
                                                  event_color={1: 'r',
                                                               2: 'b',
                                                               1000: 'k',
                                                               2000: 'k'})

# Montage B Events
mtg_b_events = mne.find_events(raw_intensity_ac, stim_channel=['Markers b'])

mtg_b_event_dict = {'Montage_B/Event_1': 1,
                    'Montage_B/Event_2': 2,
                    'Montage B/Block 1 End': 1000,
                    'Montage B/Block 2 End': 2000}

fig = mne.viz.plot_events(mtg_b_events)
fig.subplots_adjust(right=0.7)  # make room for the legend

raw_intensity_ac.copy().pick_channels(mtg_b).plot(events=mtg_b_events,
                                                  start=0, duration=10,
                                                  color='gray',
                                                  event_color={1: 'r',
                                                               2: 'b',
                                                               1000: 'k',
                                                               2000: 'k'})

# ###############################################################################
# # Next we define the range of our epochs, the rejection criteria,
# # baseline correction, and extract the epochs. We visualise the log of which
# # epochs were dropped.

# # We will make epochs from the ac-derived heamo data and the phase data
# # separately.

# reject_criteria = dict(hbo=80e-6)
reject_criteria = None
tmin, tmax = -0.2, 2
tmin_AC, tmax_AC = -2, 10

# Montage A
mtg_a = [i_index for i_index, i_label
         in enumerate(raw_haemo.info['ch_names'])
         if re.search(r'S[1-5]_', i_label)]

# haemo epochs
mtg_a_haemo_epochs = mne.Epochs(raw_haemo, mtg_a_events,
                                event_id=mtg_a_event_dict, tmin=tmin_AC,
                                tmax=tmax_AC, reject=reject_criteria,
                                reject_by_annotation=False, proj=True,
                                baseline=(None, 0), preload=True, detrend=None,
                                verbose=True, event_repeated='drop')
mtg_a_haemo_epochs.plot_drop_log()

mtg_a_epochs_ph = mne.Epochs(raw_intensity_ph, mtg_a_events,
                             event_id=mtg_a_event_dict, tmin=tmin, tmax=tmax,
                             reject=None, reject_by_annotation=False,
                             proj=False, baseline=(-0.2, 0), preload=True,
                             detrend=None, verbose=True)

# two ways to plot epochs, should be the same

# haemo epochs
fig = mne.viz.plot_epochs(mtg_a_haemo_epochs, n_epochs=5, n_channels=5,
                          scalings='auto', picks=mtg_a)
fig = mtg_a_haemo_epochs.plot(n_epochs=5, n_channels=5, scalings='auto',
                              picks=mtg_a)

# ph epochs
fig = mne.viz.plot_epochs(mtg_a_epochs_ph, n_epochs=5, n_channels=5,
                          scalings='auto', picks=mtg_a)
fig = mtg_a_epochs_ph.plot(n_epochs=5, n_channels=5, scalings='auto',
                           picks=mtg_a)


# Montage B
mtg_b = [i_index for i_index, i_label
         in enumerate(raw_haemo.info['ch_names'])
         if re.search(r'S([6-9]|10)_', i_label)]

# haemo epochs
mtg_b_haemo_epochs = mne.Epochs(raw_haemo, mtg_b_events,
                                event_id=mtg_b_event_dict, tmin=tmin_AC,
                                tmax=tmax_AC, reject=reject_criteria,
                                reject_by_annotation=False, proj=True,
                                baseline=(None, 0), preload=True, detrend=None,
                                verbose=True, event_repeated='drop')
mtg_b_haemo_epochs.plot_drop_log()

mtg_b_epochs_ph = mne.Epochs(raw_intensity_ph, mtg_b_events,
                             event_id=mtg_b_event_dict, tmin=tmin, tmax=tmax,
                             reject=None, reject_by_annotation=False,
                             proj=False, baseline=(-0.2, 0), preload=True,
                             detrend=None, verbose=True)

# two ways to plot epochs, should be the same
# haemo epochs
fig = mne.viz.plot_epochs(mtg_b_haemo_epochs, n_epochs=5, n_channels=5,
                          scalings='auto', picks=mtg_b)
fig = mtg_b_haemo_epochs.plot(n_epochs=5, n_channels=5, scalings='auto',
                              picks=mtg_b)

# ph epochs
fig = mne.viz.plot_epochs(mtg_b_epochs_ph, n_epochs=5, n_channels=5,
                          scalings='auto', picks=mtg_b)
fig = mtg_b_epochs_ph.plot(n_epochs=5, n_channels=5, scalings='auto',
                           picks=mtg_b)

# ###############################################################################
# # View consistency of responses across trials
# # -------------------------------------------
# #
# # Now we can view the haemodynamic response for our different events.

# haemo plots
# Montage A
hbo_a = [i_index for i_index, i_label
         in enumerate(mtg_a_haemo_epochs.info['ch_names'])
         if re.search(r'S[1-5]_D[0-9] hbo', i_label)]

hbr_a = [i_index for i_index, i_label
         in enumerate(mtg_a_haemo_epochs.info['ch_names'])
         if re.search(r'S[1-5]_D[0-9] hbr', i_label)]

mtg_a_haemo_epochs['Montage_A/Event_1'].plot_image(
    combine='mean', vmin=-30, vmax=30,
    group_by={'Mtg A, Event 1, Oxy': hbo_a, 'Mtg A, Event 1, De-Oxy': hbr_a},
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])))

mtg_a_haemo_epochs['Montage_A/Event_2'].plot_image(
    combine='mean', vmin=-30, vmax=30,
    group_by={'Mtg A, Event 2, Oxy': hbo_a, 'Mtg A, Event 2, De-Oxy': hbr_a},
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])))

# ph epochs
fig = mtg_a_epochs_ph['Montage_A/Event_1'].plot_image(
    combine='mean', vmin=-180, vmax=180, picks=mtg_a, colorbar=True,
    title='Montage A Event 1')

fig = mtg_a_epochs_ph['Montage_A/Event_2'].plot_image(
    combine='mean', vmin=-180, vmax=180, picks=mtg_a, colorbar=True,
    title='Montage A Event 2')


# Montage B
hbo_b = [i_index for i_index, i_label
         in enumerate(mtg_a_haemo_epochs.info['ch_names'])
         if re.search(r'S([6-9]|10)_D([0-9]|1[0-6]) hbo', i_label)]

hbr_b = [i_index for i_index, i_label
         in enumerate(mtg_a_haemo_epochs.info['ch_names'])
         if re.search(r'S([6-9]|10)_D([0-9]|1[0-6]) hbr', i_label)]

mtg_b_haemo_epochs['Montage_B/Event_1'].plot_image(
    combine='mean', vmin=-30, vmax=30,
    group_by={'Mtg B, Event 1, Oxy': hbo_b, 'Mtg B, Event 1, De-Oxy': hbr_b},
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])))

mtg_b_haemo_epochs['Montage_B/Event_2'].plot_image(
    combine='mean', vmin=-30, vmax=30,
    group_by={'Mtg B, Event 2, Oxy': hbo_b, 'Mtg B, Event 2, De-Oxy': hbr_b},
    ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])))

# ph epochs
fig = mtg_b_epochs_ph['Montage_B/Event_1'].plot_image(
    combine='mean', vmin=-180, vmax=180, picks=mtg_b, colorbar=True,
    title='Montage B Event 1')

fig = mtg_b_epochs_ph['Montage_B/Event_2'].plot_image(
    combine='mean', vmin=-180, vmax=180, picks=mtg_b, colorbar=True,
    title='Montage B Event 2')

# ###############################################################################
# # View consistency of responses across channels
# # ---------------------------------------------
# #
# # Similarly we can view how consistent the response is across the optode
# # pairs that we selected. All the channels in this data are located over the
# # motor cortex, and all channels show a similar pattern in the data.

# haemo evoked
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 6))
clim = dict(hbo=[-10, 10], hbr=[-10, 10])

mtg_a_1_evoked_ac = mtg_a_haemo_epochs['Montage_A/Event_1'].average()
mtg_a_2_evoked_ac = mtg_a_haemo_epochs['Montage_A/Event_2'].average()
mtg_b_1_evoked_ac = mtg_b_haemo_epochs['Montage_B/Event_1'].average()
mtg_b_2_evoked_ac = mtg_b_haemo_epochs['Montage_B/Event_2'].average()

mtg_a_1_evoked_ac.plot_image(axes=axes[0, 0], picks=hbo_a,
                             titles='HBO Montage A Event 1', clim=clim)
mtg_a_1_evoked_ac.plot_image(axes=axes[0, 1], picks=hbr_a,
                             titles='HBR Montage A Event 1', clim=clim)
mtg_a_2_evoked_ac.plot_image(axes=axes[1, 0], picks=hbo_a,
                             titles='HBO Montage A Event 2', clim=clim)
mtg_a_2_evoked_ac.plot_image(axes=axes[1, 1], picks=hbr_a,
                             titles='HBR Montage A Event 2', clim=clim)
mtg_b_1_evoked_ac.plot_image(axes=axes[2, 0], picks=hbo_b,
                             titles='HBO Montage B Event 1', clim=clim)
mtg_b_1_evoked_ac.plot_image(axes=axes[2, 1], picks=hbr_b,
                             titles='HBR Montage B Event 1', clim=clim)
mtg_b_2_evoked_ac.plot_image(axes=axes[3, 0], picks=hbo_b,
                             titles='HBO Montage B Event 2', clim=clim)
mtg_b_2_evoked_ac.plot_image(axes=axes[3, 1], picks=hbr_b,
                             titles='HBR Montage B Event 2', clim=clim)

# Combine Montages
mtg_a_channels_ac = [i_index for i_index, i_label
                     in enumerate(mtg_a_1_evoked_ac.info['ch_names'])
                     if re.search(r'S[1-5]_', i_label)]

mtg_b_channels_ac = [i_index for i_index, i_label
                     in enumerate(mtg_b_1_evoked_ac.info['ch_names'])
                     if re.search(r'S([6-9]|10)_', i_label)]

# zero channels that don't correspond to montage A/B
mtg_a_1_evoked_ac._data[mtg_b_channels_ac, :] = 0
mtg_a_2_evoked_ac._data[mtg_b_channels_ac, :] = 0
mtg_b_1_evoked_ac._data[mtg_a_channels_ac, :] = 0
mtg_b_2_evoked_ac._data[mtg_a_channels_ac, :] = 0

evoked_event_1_ac = mne.combine_evoked([mtg_a_1_evoked_ac, mtg_b_1_evoked_ac],
                                       'equal')
evoked_event_2_ac = mne.combine_evoked([mtg_a_2_evoked_ac, mtg_b_2_evoked_ac],
                                       'equal')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
clim = dict(fnirs_raw=[-20, 20])

evoked_event_1_ac.plot_image(axes=axes[:, 0],
                             titles=dict(hbo='HBO_Event_1', hbr='HBR_Event_1'),
                             clim=clim)
evoked_event_2_ac.plot_image(axes=axes[:, 1],
                             titles=dict(hbo='HBO_Event_2', hbr='HBR_Event_2'),
                             clim=clim)

# ph evoked
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
clim = dict(fnirs_ph=[-180, 180])

mtg_a_1_evoked_ph = mtg_a_epochs_ph['Montage_A/Event_1'].average()
mtg_a_2_evoked_ph = mtg_a_epochs_ph['Montage_A/Event_2'].average()
mtg_b_1_evoked_ph = mtg_b_epochs_ph['Montage_B/Event_1'].average()
mtg_b_2_evoked_ph = mtg_b_epochs_ph['Montage_B/Event_2'].average()

mtg_a_1_evoked_ph.plot_image(axes=axes[0, 0], picks=mtg_a,
                             titles='Montage A Event 1', clim=clim)
mtg_a_2_evoked_ph.plot_image(axes=axes[1, 0], picks=mtg_a,
                             titles='Montage A Event 2', clim=clim)
mtg_b_1_evoked_ph.plot_image(axes=axes[0, 1], picks=mtg_b,
                             titles='Montage B Event 1', clim=clim)
mtg_b_2_evoked_ph.plot_image(axes=axes[1, 1], picks=mtg_b,
                             titles='Montage B Event 2', clim=clim)

# Combine Montages
mtg_a_channels_ph = [i_index for i_index, i_label
                     in enumerate(mtg_a_1_evoked_ph.info['ch_names'])
                     if re.search(r'S[1-5]_', i_label)]

mtg_b_channels_ph = [i_index for i_index, i_label
                     in enumerate(mtg_b_1_evoked_ph.info['ch_names'])
                     if re.search(r'S([6-9]|10)_', i_label)]

# zero channels that don't correspond to montage A/B
mtg_a_1_evoked_ph._data[mtg_b_channels_ph, :] = 0
mtg_a_2_evoked_ph._data[mtg_b_channels_ph, :] = 0
mtg_b_1_evoked_ph._data[mtg_a_channels_ph, :] = 0
mtg_b_2_evoked_ph._data[mtg_a_channels_ph, :] = 0

evoked_event_1_ph = mne.combine_evoked([mtg_a_1_evoked_ph, mtg_b_1_evoked_ph],
                                       'equal')
evoked_event_2_ph = mne.combine_evoked([mtg_a_2_evoked_ph, mtg_b_2_evoked_ph],
                                       'equal')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
clim = dict(fnirs_ph=[-180, 180])

evoked_event_1_ph.plot_image(axes=axes[0], titles='Event_1', clim=clim)
evoked_event_2_ph.plot_image(axes=axes[1], titles='Event_2', clim=clim)

# ###############################################################################
# # Plot standard haemodynamic response image
# # ----------------------------------
# #
# # Plot both the HbO and HbR on the same figure to illustrate the relation
# # between the two signals.

# # We can also plot a similat figure for phase data.

# haemo
evoked_dict_ac = {'Event_1': evoked_event_1_ac, 'Event_2': evoked_event_2_ac}

color_dict = {'Event_1': 'r', 'Event_2': 'b'}

mne.viz.plot_compare_evokeds(evoked_dict_ac, combine="mean", ci=0.95,
                             colors=color_dict)

# ph
evoked_dict_ph = {'Event_1': evoked_event_1_ph, 'Event_2': evoked_event_2_ph}

color_dict = {'Event_1': 'r', 'Event_2': 'b'}

mne.viz.plot_compare_evokeds(evoked_dict_ph, combine="mean", ci=0.95,
                             colors=color_dict)

# ###############################################################################
# # View topographic representation of activity
# # -------------------------------------------
# #
# # Next we view how the topographic activity changes throughout the
# # haemodynamic and phase response.

# ac
times = np.arange(0.0, 10.0, 2.0)
topomap_args = dict(extrapolate='local')

fig = evoked_event_1_ac.plot_joint(times=times, topomap_args=topomap_args)
fig = evoked_event_2_ac.plot_joint(times=times, topomap_args=topomap_args)

# ph
times = np.arange(0.0, 2.0, 0.5)
topomap_args = dict(extrapolate='local')

fig = evoked_event_1_ph.plot_joint(times=times, topomap_args=topomap_args,
                                   title='Event 1 Phase')
fig = evoked_event_2_ph.plot_joint(times=times, topomap_args=topomap_args,
                                   title='Event 2 Phase')

# ###############################################################################
# # Compare Events 1 and 2
# # ---------------------------------------
# #
# # We generate topo maps for events 1 and 2 to view the location of activity.
# # First we visualise the HbO activity.

# ac HBO
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))

topomap_args = dict(extrapolate='local', size=3, res=256, sensors='k.')
times = 1.0

hbo_a = [i_index for i_index, i_label
         in enumerate(mtg_a_1_evoked_ac.info['ch_names'])
         if re.search(r'S[1-5]_D[0-9] hbo', i_label)]

hbo_b = [i_index for i_index, i_label
         in enumerate(mtg_b_1_evoked_ac.info['ch_names'])
         if re.search(r'S([6-9]|10)_D([0-9]|1[0-6]) hbo', i_label)]

evoked_event_1_ac.copy().pick(hbo_a).plot_topomap(times=times,
                                                  axes=axes[0, 0],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_2_ac.copy().pick(hbo_a).plot_topomap(times=times,
                                                  axes=axes[1, 0],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_1_ac.copy().pick(hbo_b).plot_topomap(times=times,
                                                  axes=axes[0, 1],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_2_ac.copy().pick(hbo_b).plot_topomap(times=times,
                                                  axes=axes[1, 1],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_1_ac.copy().pick(hbo_a+hbo_b).plot_topomap(times=times,
                                                        axes=axes[0, 2:],
                                                        colorbar=True,
                                                        **topomap_args)

evoked_event_2_ac.copy().pick(hbo_a+hbo_b).plot_topomap(times=times,
                                                        axes=axes[1, 2:],
                                                        colorbar=True,
                                                        **topomap_args)

for column, condition in enumerate(['Montage A', 'Montage B', 'Combined']):
    for row, chroma in enumerate(['HBO Event 1', 'HBO Event 2']):
        axes[row, column].set_title('{}: {}'.format(chroma, condition))
fig.tight_layout()


# ac HBR
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))

topomap_args = dict(extrapolate='local', size=3, res=256, sensors='k.')
times = 1.0

hbr_a = [i_index for i_index, i_label
         in enumerate(mtg_a_1_evoked_ac.info['ch_names'])
         if re.search(r'S[1-5]_D[0-9] hbr', i_label)]

hbr_b = [i_index for i_index, i_label
         in enumerate(mtg_b_1_evoked_ac.info['ch_names'])
         if re.search(r'S([6-9]|10)_D([0-9]|1[0-6]) hbr', i_label)]


evoked_event_1_ac.copy().pick(hbr_a).plot_topomap(times=times,
                                                  axes=axes[0, 0],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_2_ac.copy().pick(hbr_a).plot_topomap(times=times,
                                                  axes=axes[1, 0],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_1_ac.copy().pick(hbr_b).plot_topomap(times=times,
                                                  axes=axes[0, 1],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_2_ac.copy().pick(hbr_b).plot_topomap(times=times,
                                                  axes=axes[1, 1],
                                                  colorbar=False,
                                                  **topomap_args)

evoked_event_1_ac.copy().pick(hbr_a+hbr_b).plot_topomap(times=times,
                                                        axes=axes[0, 2:],
                                                        colorbar=True,
                                                        **topomap_args)
evoked_event_2_ac.copy().pick(hbr_a+hbr_b).plot_topomap(times=times,
                                                        axes=axes[1, 2:],
                                                        colorbar=True,
                                                        **topomap_args)

for column, condition in enumerate(['Montage A', 'Montage B', 'Combined']):
    for row, chroma in enumerate(['HBR Event 1', 'HBR Event 2']):
        axes[row, column].set_title('{}: {}'.format(chroma, condition))
fig.tight_layout()


# ph
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))

topomap_args = dict(extrapolate='local', size=3, res=256, sensors='k.')
times = 1.0

evoked_event_1_ph.copy().pick(mtg_a_channels_ph).plot_topomap(times=times,
                                                              axes=axes[0, 0],
                                                              colorbar=False,
                                                              **topomap_args)

evoked_event_2_ph.copy().pick(mtg_a_channels_ph).plot_topomap(times=times,
                                                              axes=axes[1, 0],
                                                              colorbar=False,
                                                              **topomap_args)

evoked_event_1_ph.copy().pick(mtg_b_channels_ph).plot_topomap(times=times,
                                                              axes=axes[0, 1],
                                                              colorbar=False,
                                                              **topomap_args)

evoked_event_2_ph.copy().pick(mtg_b_channels_ph).plot_topomap(times=times,
                                                              axes=axes[1, 1],
                                                              colorbar=False,
                                                              **topomap_args)

evoked_event_1_ph.plot_topomap(times=times, axes=axes[0, 2:], colorbar=True,
                               **topomap_args)
evoked_event_2_ph.plot_topomap(times=times, axes=axes[1, 2:], colorbar=True,
                               **topomap_args)

for column, condition in enumerate(['Montage A', 'Montage B', 'Combined']):
    for row, chroma in enumerate(['Event 1', 'Event 2']):
        axes[row, column].set_title('{}: {}'.format(chroma, condition))
fig.tight_layout()

# ###############################################################################
# # And we can plot the comparison at a single time point for two conditions.

# ac HBO
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))
vmin, vmax, ts = -0.192, 0.992, 0.1
vmin = -5
vmax = 5

evoked_event_1_ac.plot_topomap(ch_type='hbo', times=ts, axes=axes[0],
                               vmin=vmin, vmax=vmax,
                               colorbar=False, **topomap_args)

evoked_event_2_ac.plot_topomap(ch_type='hbo', times=ts, axes=axes[1],
                               vmin=vmin, vmax=vmax,
                               colorbar=False, **topomap_args)

evoked_diff_ac = mne.combine_evoked([evoked_event_1_ac, -evoked_event_2_ac],
                                    weights='equal')

evoked_diff_ac.plot_topomap(ch_type='hbo', times=ts, axes=axes[2:],
                            vmin=vmin, vmax=vmax,
                            colorbar=True, **topomap_args)

for column, condition in enumerate(
        ['HBO Event 1', 'HBO Event 2', 'HBO Difference']):
    axes[column].set_title('{}'.format(condition))
fig.tight_layout()


# ac HBR
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))
vmin, vmax, ts = -0.192, 0.992, 0.1
vmin = -5
vmax = 5

evoked_event_1_ac.plot_topomap(ch_type='hbr', times=ts, axes=axes[0],
                               vmin=vmin, vmax=vmax,
                               colorbar=False, **topomap_args)

evoked_event_2_ac.plot_topomap(ch_type='hbr', times=ts, axes=axes[1],
                               vmin=vmin, vmax=vmax,
                               colorbar=False, **topomap_args)

evoked_diff_ac = mne.combine_evoked([evoked_event_1_ac, -evoked_event_2_ac],
                                    weights='equal')

evoked_diff_ac.plot_topomap(ch_type='hbr', times=ts, axes=axes[2:],
                            vmin=vmin, vmax=vmax,
                            colorbar=True, **topomap_args)

for column, condition in enumerate(
        ['HBR Event 1', 'HBR Event 2', 'HBR Difference']):
    axes[column].set_title('{}'.format(condition))
fig.tight_layout()


# ph
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))
vmin, vmax, ts = -0.192, 0.992, 0.1
vmin = -180
vmax = 180

evoked_event_1_ph.plot_topomap(times=ts, axes=axes[0], vmin=vmin, vmax=vmax,
                               colorbar=False, **topomap_args)

evoked_event_2_ph.plot_topomap(times=ts, axes=axes[1], vmin=vmin, vmax=vmax,
                               colorbar=False, **topomap_args)

evoked_diff_ph = mne.combine_evoked([evoked_event_1_ph, -evoked_event_2_ph],
                                    weights='equal')

evoked_diff_ph.plot_topomap(times=ts, axes=axes[2:], vmin=vmin, vmax=vmax,
                            colorbar=True, **topomap_args)

for column, condition in enumerate(['Event 1', 'Event 2', 'Difference']):
    axes[column].set_title('{}'.format(condition))
fig.tight_layout()

# #############################################################################
# # Lastly, we can also look at the individual waveforms to see what is
# # driving the topographic plot above.

# ac HBO
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
mne.viz.plot_evoked_topo(evoked_event_1_ac.copy().pick('hbo'),
                         color='b', axes=axes, legend=False)
mne.viz.plot_evoked_topo(evoked_event_2_ac.copy().pick('hbo'),
                         color='r', axes=axes, legend=False)

# Tidy the legend
leg_lines = [line for line in axes.lines if line.get_c() == 'b'][:1]
leg_lines.append([line for line in axes.lines if line.get_c() == 'r'][0])
fig.legend(leg_lines, ['HBO Event 1', 'HBO Event 2'], loc='lower right')


# ac HBR
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
mne.viz.plot_evoked_topo(evoked_event_1_ac.copy().pick('hbr'),
                         color='b', axes=axes, legend=False)
mne.viz.plot_evoked_topo(evoked_event_2_ac.copy().pick('hbr'),
                         color='r', axes=axes, legend=False)

# Tidy the legend
leg_lines = [line for line in axes.lines if line.get_c() == 'b'][:1]
leg_lines.append([line for line in axes.lines if line.get_c() == 'r'][0])
fig.legend(leg_lines, ['HBR Event 1', 'HBR Event 2'], loc='lower right')


# ph
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
mne.viz.plot_evoked_topo(evoked_event_1_ph, color='b', axes=axes, legend=False)
mne.viz.plot_evoked_topo(evoked_event_2_ph, color='r', axes=axes, legend=False)

# Tidy the legend
leg_lines = [line for line in axes.lines if line.get_c() == 'b'][:1]
leg_lines.append([line for line in axes.lines if line.get_c() == 'r'][0])
fig.legend(leg_lines, ['Phase Event 1', 'Phase Event 2'], loc='lower right')
