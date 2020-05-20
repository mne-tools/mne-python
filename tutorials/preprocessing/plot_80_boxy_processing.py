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
# sphinx_gallery_thumbnail_number = 1

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
import re as re

import mne


boxy_data_folder = mne.datasets.boxy_example.data_path()
boxy_raw_dir = os.path.join(boxy_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_boxy(boxy_raw_dir, 'AC', verbose=True).load_data()

###separate data based on montages###
mtg_a_indices = [i_index for i_index,i_label in enumerate(raw_intensity.info['ch_names']) 
                 if re.search(r'(S[1-5]_|\bMarkers a\b)', i_label)]
mtg_b_indices = [i_index for i_index,i_label in enumerate(raw_intensity.info['ch_names']) 
                 if re.search(r'(S([6-9]|10)_|\bMarkers b\b)', i_label)]

mtg_a_intensity = raw_intensity.copy()
mtg_b_intensity = raw_intensity.copy()

mtg_a_intensity.pick(mtg_a_indices)
mtg_b_intensity.pick(mtg_b_indices)

# ###############################################################################
# # View location of sensors over brain surface
# # -------------------------------------------
# #
# # Here we validate that the location of sources-detector pairs and channels
# # are in the expected locations. Source-detector pairs are shown as lines
# # between the optodes, channels (the mid point of source-detector pairs) are
# # shown as dots.

subjects_dir = os.path.dirname(mne.datasets.fetch_fsaverage())

fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(raw_intensity.info, 
							 show_axes=True,
                             subject='fsaverage',
                             trans='fsaverage', 
                             surfaces=['head-dense', 'brain'],
                             fnirs=['sources','detectors', 'pairs'],
                             mri_fiducials=True,
                             dig=True,
                             subjects_dir=subjects_dir, 
                             fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=20, elevation=55, distance=0.6)

fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(mtg_a_intensity.info, 
							 show_axes=True,
                             subject='fsaverage',
                             trans='fsaverage', 
                             surfaces=['head-dense', 'brain'],
                             fnirs=['sources','detectors', 'pairs'],
                             mri_fiducials=True,
                             dig=True,
                             subjects_dir=subjects_dir, 
                             fig=fig)
mne.viz.set_3d_view(figure=fig, azimuth=20, elevation=55, distance=0.6)

fig = mne.viz.create_3d_figure(size=(800, 600), bgcolor='white')
fig = mne.viz.plot_alignment(mtg_b_intensity.info, 
							 show_axes=True,
                             subject='fsaverage',
                             trans='fsaverage', 
                             surfaces=['head-dense', 'brain'],
                             fnirs=['sources','detectors', 'pairs'],
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
# # detect a neural response (less than 1 cm distance between optodes).
# # These short channels can be seen in the figure above.
# # To achieve this we pick all the channels that are not considered to be short.

picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True, stim=True)
picks_a = mne.pick_types(mtg_a_intensity.info, meg=False, fnirs=True, stim=True)
picks_b = mne.pick_types(mtg_b_intensity.info, meg=False, fnirs=True, stim=True)

dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks)
dists_a = mne.preprocessing.nirs.source_detector_distances(
    mtg_a_intensity.info, picks=picks_a)
dists_b = mne.preprocessing.nirs.source_detector_distances(
    mtg_b_intensity.info, picks=picks_b)

raw_intensity.pick(picks[dists < 0.08])
mtg_a_intensity.pick(picks_a[dists_a < 0.08])
mtg_b_intensity.pick(picks_b[dists_b < 0.08])

scalings = dict(fnirs_raw=1e2)
raw_intensity.plot(n_channels=5,
                   duration=20, scalings=100, show_scrollbars=True)
mtg_a_intensity.plot(n_channels=5,
                   duration=20, scalings=100, show_scrollbars=True)
mtg_b_intensity.plot(n_channels=5,
                   duration=20, scalings=100, show_scrollbars=True)


# ###############################################################################
# # Converting from raw intensity to optical density
# # ------------------------------------------------
# #
# # The raw intensity values are then converted to optical density.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od_a = mne.preprocessing.nirs.optical_density(mtg_a_intensity)
raw_od_b = mne.preprocessing.nirs.optical_density(mtg_b_intensity)

raw_od.plot(n_channels=len(raw_od.ch_names),
            duration=500, show_scrollbars=False)
raw_od_a.plot(n_channels=len(raw_od_a.ch_names),
            duration=500, show_scrollbars=False)
raw_od_b.plot(n_channels=len(raw_od_b.ch_names),
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
sci_a = mne.preprocessing.nirs.scalp_coupling_index(raw_od_a)
sci_b = mne.preprocessing.nirs.scalp_coupling_index(raw_od_b)

fig, ax = plt.subplots()
ax.hist(sci)
ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])

fig, ax = plt.subplots()
ax.hist(sci_a)
ax.set(xlabel='Scalp Coupling Index-A', ylabel='Count', xlim=[0, 1])

fig, ax = plt.subplots()
ax.hist(sci_b)
ax.set(xlabel='Scalp Coupling Index-B', ylabel='Count', xlim=[0, 1])


# ###############################################################################
# # In this example we will mark all channels with a SCI less than 0.5 as bad
# # (this dataset is quite clean, so no channels are marked as bad).

raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))
raw_od_a.info['bads'] = list(compress(raw_od_a.ch_names, sci_a < 0.5))
raw_od_b.info['bads'] = list(compress(raw_od_b.ch_names, sci_b < 0.5))

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
raw_haemo_a = mne.preprocessing.nirs.beer_lambert_law(raw_od_a)
raw_haemo_b = mne.preprocessing.nirs.beer_lambert_law(raw_od_b)

raw_haemo.plot(n_channels=len(raw_haemo.ch_names),
                duration=500, show_scrollbars=False)

raw_haemo_a.plot(n_channels=len(raw_haemo_a.ch_names),
                duration=500, show_scrollbars=False)

raw_haemo_b.plot(n_channels=len(raw_haemo_b.ch_names),
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


fig = raw_haemo_a.plot_psd(average=True)
fig.suptitle('Before filtering Montage A', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)
raw_haemo_a = raw_haemo_a.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                              l_trans_bandwidth=0.02)
fig = raw_haemo_a.plot_psd(average=True)
fig.suptitle('After filtering Montage A', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)


fig = raw_haemo_b.plot_psd(average=True)
fig.suptitle('Before filtering Montage B', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)
raw_haemo_b = raw_haemo_b.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                              l_trans_bandwidth=0.02)
fig = raw_haemo_b.plot_psd(average=True)
fig.suptitle('After filtering Montage B', weight='bold', size='x-large')
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

mtg_a_events = mne.find_events(mtg_a_intensity, stim_channel='Markers a')

fig = mne.viz.plot_events(mtg_a_events)
fig.subplots_adjust(right=0.7)  # make room for the legend

mtg_b_events = mne.find_events(mtg_b_intensity, stim_channel='Markers b')

fig = mne.viz.plot_events(mtg_b_events)
fig.subplots_adjust(right=0.7)  # make room for the legend

# ###############################################################################
# # Next we define the range of our epochs, the rejection criteria,
# # baseline correction, and extract the epochs. We visualise the log of which
# # epochs were dropped.

# reject_criteria = dict(hbo=80e-6)
reject_criteria = None
tmin, tmax = -0.2, 1

mtg_a_haemo_epochs = mne.Epochs(raw_haemo_a, mtg_a_events,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria, reject_by_annotation=False,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True)
mtg_a_haemo_epochs.plot_drop_log()


mtg_b_haemo_epochs = mne.Epochs(raw_haemo_b, mtg_b_events,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria, reject_by_annotation=False,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True)
mtg_b_haemo_epochs.plot_drop_log()


#get epochs from the raw intensities
mtg_a_epochs = mne.Epochs(mtg_a_intensity, mtg_a_events, 
                    event_id=dict(event_1=1,event_2=2),
                    tmin=tmin, tmax=tmax,
                    reject=None, reject_by_annotation=False,
                    proj=False, baseline=(-0.2, 0), preload=True,
                    detrend=None, verbose=True)

mtg_b_epochs = mne.Epochs(mtg_b_intensity, mtg_b_events, 
                    event_id=dict(event_1=1,event_2=2),
                    tmin=tmin, tmax=tmax,
                    reject=None, reject_by_annotation=False,
                    proj=False, baseline=(-0.2, 0), preload=True,
                    detrend=None, verbose=True)

#two ways to plot epochs, should be the same
fig = mne.viz.plot_epochs(mtg_a_epochs,n_epochs=5,n_channels=5, scalings='auto')
fig = mtg_a_epochs.plot(n_epochs=5,n_channels=5, scalings='auto')

fig = mne.viz.plot_epochs(mtg_b_epochs,n_epochs=5,n_channels=5, scalings='auto')
fig = mtg_b_epochs.plot(n_epochs=5,n_channels=5, scalings='auto')


# ###############################################################################
# # View consistency of responses across trials
# # -------------------------------------------
# #
# # Now we can view the haemodynamic response for our tapping condition.
# # We visualise the response for both the oxy- and deoxyhaemoglobin, and
# # observe the expected peak in HbO at around 6 seconds consistently across
# # trials, and the consistent dip in HbR that is slightly delayed relative to
# # the HbO peak.

#haemo plots
mtg_a_haemo_epochs['1'].plot_image(combine='mean', vmin=-30, vmax=30,
                              ts_args=dict(ylim=dict(hbo=[-15, 15],
                                                    hbr=[-15, 15])))

mtg_a_haemo_epochs['2'].plot_image(combine='mean', vmin=-30, vmax=30,
                              ts_args=dict(ylim=dict(hbo=[-15, 15],
                                                    hbr=[-15, 15])))

mtg_b_haemo_epochs['1'].plot_image(combine='mean', vmin=-30, vmax=30,
                              ts_args=dict(ylim=dict(hbo=[-15, 15],
                                                    hbr=[-15, 15])))

mtg_b_haemo_epochs['2'].plot_image(combine='mean', vmin=-30, vmax=30,
                              ts_args=dict(ylim=dict(hbo=[-15, 15],
                                                    hbr=[-15, 15])))

#raw epochs
#separate first and last detectors
mtg_a_first_det = ([i_index for i_index,i_label in
                   enumerate(mtg_a_epochs.info['ch_names']) if
                   re.search(r'_D[1-4]', i_label)])

mtg_a_last_det = ([i_index for i_index,i_label in
                   enumerate(mtg_a_epochs.info['ch_names']) if
                   re.search(r'_D[5-8]', i_label)])

mtg_b_first_det = ([i_index for i_index,i_label in
                   enumerate(mtg_b_epochs.info['ch_names']) if
                   re.search(r'_D(9|1[0-2])', i_label)])

mtg_b_last_det = ([i_index for i_index,i_label in
                   enumerate(mtg_b_epochs.info['ch_names']) if
                   re.search(r'_D1[3-6]', i_label)])

#plot our two events for both montages
fig = mtg_a_epochs['event_1'].plot_image(combine='mean', vmin=-20, vmax=20, 
                                   colorbar=True, title='Montage A Event 1',
                                   group_by=dict(FIRST_DET=mtg_a_first_det, 
                                                 LAST_DET=mtg_a_last_det))

fig = mtg_a_epochs['event_2'].plot_image(combine='mean', vmin=-20, vmax=20, 
                                   colorbar=True, title='Montage A Event 2',
                                   group_by=dict(FIRST_DET=mtg_a_first_det, 
                                                 LAST_DET=mtg_a_last_det))

fig = mtg_b_epochs['event_1'].plot_image(combine='mean', vmin=-20, vmax=20, 
                                   colorbar=True, title='Montage B Event 1',
                                   group_by=dict(FIRST_DET=mtg_b_first_det, 
                                                 LAST_DET=mtg_b_last_det))

fig = mtg_b_epochs['event_2'].plot_image(combine='mean', vmin=-20, vmax=20, 
                                   colorbar=True, title='Montage B Event 2',
                                   group_by=dict(FIRST_DET=mtg_b_first_det, 
                                                 LAST_DET=mtg_b_last_det))

# ###############################################################################
# # View consistency of responses across channels
# # ---------------------------------------------
# #
# # Similarly we can view how consistent the response is across the optode
# # pairs that we selected. All the channels in this data are located over the
# # motor cortex, and all channels show a similar pattern in the data.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
clim=dict(fnirs_raw=[-20,20])
mtg_a_epochs['event_1'].average().plot_image(axes=axes[0, 0],titles='Montage A Event 1', clim=clim)
mtg_a_epochs['event_2'].average().plot_image(axes=axes[1, 0],titles='Montage A Event 2', clim=clim)
mtg_b_epochs['event_1'].average().plot_image(axes=axes[0, 1],titles='Montage B Event 1', clim=clim)
mtg_b_epochs['event_2'].average().plot_image(axes=axes[1, 1],titles='Montage B Event 2', clim=clim)

# ###############################################################################
# # Plot standard fNIRS response image
# # ----------------------------------
# #
# # Next we generate the most common visualisation of fNIRS data: plotting
# # both the HbO and HbR on the same figure to illustrate the relation between
# # the two signals.

mtg_a_evoked_dict = {'Montage_A_Event_1': mtg_a_epochs['event_1'].average(),
                'Montage_A_Event_2': mtg_a_epochs['event_2'].average()}

mtg_b_evoked_dict = {'Montage_B_Event_1': mtg_b_epochs['event_1'].average(),
                'Montage_B_Event_2': mtg_b_epochs['event_2'].average()}

###this seems to what our conditions/events to have the same number of channels,
###and the same channel names. Maybe we can't use this to compare montages??
###Gives an error if I try to compare both montages and events
color_dict = dict(Montage_A_Event_1='r', Montage_A_Event_2='b')
mne.viz.plot_compare_evokeds(mtg_a_evoked_dict, combine="mean", ci=0.95,
                              colors=color_dict)

color_dict = dict(Montage_B_Event_1='r', Montage_B_Event_2='b')
mne.viz.plot_compare_evokeds(mtg_b_evoked_dict, combine="mean", ci=0.95,
                              colors=color_dict)

# ###############################################################################
# # View topographic representation of activity
# # -------------------------------------------
# #
# # Next we view how the topographic activity changes throughout the response.

times = np.arange(-0.2, 1.0, 0.2)
topomap_args = dict(extrapolate='local')

fig = mtg_a_epochs['event_1'].average().plot_joint(times=times, 
                                                   topomap_args=topomap_args)
fig = mtg_a_epochs['event_2'].average().plot_joint(times=times, 
                                                   topomap_args=topomap_args)
fig = mtg_b_epochs['event_1'].average().plot_joint(times=times, 
                                                   topomap_args=topomap_args)
fig = mtg_b_epochs['event_2'].average().plot_joint(times=times, 
                                                   topomap_args=topomap_args)

# ###############################################################################
# # Compare tapping of left and right hands
# # ---------------------------------------
# #
# # Finally we generate topo maps for the left and right conditions to view
# # the location of activity. First we visualise the HbO activity.

times = np.arange(0.0, 1.0, 0.2)
mtg_a_epochs['event_1'].average().plot_topomap(times=times, title='Montage A Event 1', **topomap_args)
mtg_a_epochs['event_2'].average().plot_topomap(times=times, title='Montage A Event 2', **topomap_args)
mtg_b_epochs['event_1'].average().plot_topomap(times=times, title='Montage B Event 1', **topomap_args)
mtg_b_epochs['event_2'].average().plot_topomap(times=times, title='Montage B Event 2', **topomap_args)

# ###############################################################################
# # And we can plot the comparison at a single time point for two conditions.

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(9, 5),
                         gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]))
vmin, vmax, ts = -0.192, 0.992, 0.1
vmin = -20
vmax = 20

mtg_a_epochs['event_1'].average().plot_topomap(times=ts, 
                       axes=axes[0, 0], vmin=vmin, vmax=vmax, colorbar=False,
                       **topomap_args)

mtg_a_epochs['event_2'].average().plot_topomap(times=ts, 
                       axes=axes[1, 0], vmin=vmin, vmax=vmax, colorbar=False,
                       **topomap_args)

mtg_b_epochs['event_1'].average().plot_topomap(times=ts, 
                       axes=axes[0, 1], vmin=vmin, vmax=vmax, colorbar=False,
                       **topomap_args)

mtg_b_epochs['event_2'].average().plot_topomap(times=ts, 
                       axes=axes[1, 1], vmin=vmin, vmax=vmax, colorbar=False,
                       **topomap_args)


###can't compare events across montages, for this data set, since they
#don't have the same channel names
mtg_a_evoked_diff = mne.combine_evoked([mtg_a_epochs['event_1'].average(),
                                        -mtg_a_epochs['event_2'].average()],
                                       weights='equal')

mtg_b_evoked_diff = mne.combine_evoked([mtg_b_epochs['event_1'].average(),
                                        -mtg_b_epochs['event_2'].average()],
                                       weights='equal')

mtg_a_evoked_diff.plot_topomap(times=ts, axes=axes[0, 2:],
                          vmin=vmin, vmax=vmax, colorbar=True,
                          **topomap_args)
mtg_b_evoked_diff.plot_topomap(times=ts, axes=axes[1, 2:],
                          vmin=vmin, vmax=vmax, colorbar=True,
                          **topomap_args)

for column, condition in enumerate(
        ['Event 1', 'Event 2', 'Difference']):
    for row, chroma in enumerate(['Montage A', 'Montage B']):
        axes[row, column].set_title('{}: {}'.format(chroma, condition))
fig.tight_layout()

# ###############################################################################
# # Lastly, we can also look at the individual waveforms to see what is
# # driving the topographic plot above.

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
mne.viz.plot_evoked_topo(mtg_a_epochs['event_1'].average(), color='b',
                          axes=axes, legend=False)
mne.viz.plot_evoked_topo(mtg_a_epochs['event_2'].average(), color='r',
                          axes=axes, legend=False)

# Tidy the legend
leg_lines = [line for line in axes.lines if line.get_c() == 'b'][:1]
leg_lines.append([line for line in axes.lines if line.get_c() == 'r'][0])
fig.legend(leg_lines, ['Montage A Event 1', 'Montage A Event 2'], loc='lower right')


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
mne.viz.plot_evoked_topo(mtg_b_epochs['event_1'].average(), color='b',
                          axes=axes, legend=False)
mne.viz.plot_evoked_topo(mtg_b_epochs['event_2'].average(), color='r',
                          axes=axes, legend=False)

# Tidy the legend
leg_lines = [line for line in axes.lines if line.get_c() == 'b'][:1]
leg_lines.append([line for line in axes.lines if line.get_c() == 'r'][0])
fig.legend(leg_lines, ['Montage A Event 1', 'Montage A Event 2'], loc='lower right')
