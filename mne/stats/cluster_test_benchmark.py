import mne
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.stats import cluster_level # gives you all functions (also private functions)

# import and load dataset
path_to_p3 = Path("C:/Users/Carina/mne_data/ERP_CORE_P3")

# Define the range of participant IDs
participant_ids = range(15, 20)  # This will cover 015 to 019

evokeds_allsubs = []

# Loop over each participant ID and generate the corresponding filename
for pid in participant_ids:
    # Create the filename using an f-string, ensuring the participant ID is zero-padded to 3 digits
    filename_p3 = f"sub-{pid:03d}_ses-P3_task-P3_ave.fif"

    # Print the filename (or perform your desired operations on it)
    print(filename_p3)

    p3_file_path = Path(path_to_p3) / filename_p3

    evokeds = mne.read_evokeds(p3_file_path)

    # add to list
    evokeds_allsubs.append(evokeds)

target_only = [evoked[0] for evoked in evokeds_allsubs]
non_target_only = [evoked[1] for evoked in evokeds_allsubs]
contrast = [evoked[2] for evoked in evokeds_allsubs]

# plot the grand average
mne.grand_average(target_only).plot()
mne.grand_average(non_target_only).plot()
mne.grand_average(contrast).plot()

# create contrast from evokeds target and non-target
diff_evoked = [mne.combine_evoked([evokeds_a, evokeds_b], weights=[1, -1]) for evokeds_a, evokeds_b in zip(target_only, non_target_only)]

mne.grand_average(diff_evoked).plot()

# crop the evokeds in the post stimulus window
contrast = [evokeds.crop(tmin=-0.1, tmax=0.6) for evokeds in contrast]
target_only = [evokeds.crop(tmin=-0.1, tmax=0.6) for evokeds in target_only]
non_target_only = [evokeds.crop(tmin=-0.1, tmax=0.6) for evokeds in non_target_only]

# extract the data for each evoked and store in numpy array
data = np.array([evoked.data for evoked in contrast])

# shape should be (n_subjects, n_channels, n_times)
data.shape

# reshape to channels as last dimension
data = data.transpose(0, 2, 1)

data.shape

adjacency, _ = mne.channels.find_ch_adjacency(contrast[0].info, ch_type='eeg')

# We want a two-tailed test
tail = 0

# Set the number of permutations
n_permutations = 10000

# Run the analysis
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
    data,
    n_permutations=n_permutations,
    tail=0,
    adjacency=adjacency,
    verbose=True,
)

print(min(cluster_p_values))

lowest_p_cluster = np.argmin(cluster_p_values)

# configure variables for visualization
colors = {"target": "crimson", "non-target": "steelblue"}

# organize data for plotting
evokeds = {"target": target_only, "non-target": non_target_only}

# plot the cluster with the lowest p-value
time_inds, space_inds = np.squeeze(clusters[lowest_p_cluster])
ch_inds = np.unique(space_inds)
time_inds = np.unique(time_inds)

# get topography for F stat
t_map = T_obs[time_inds, ...].mean(axis=0)

# get signals at the sensors contributing to the cluster
sig_times = contrast[0].times[time_inds]

# create spatial mask
mask = np.zeros((t_map.shape[0], 1), dtype=bool)
mask[ch_inds, :] = True

# initialize figure
fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

# plot average test statistic and mark significant sensors
t_evoked = mne.EvokedArray(t_map[:, np.newaxis], contrast[0].info, tmin=0)
t_evoked.plot_topomap(
    times=0,
    mask=mask,
    axes=ax_topo,
    cmap="Reds",
    vlim=(np.min, np.max),
    show=False,
    colorbar=False,
    mask_params=dict(markersize=10),
)
image = ax_topo.images[0]

# remove the title that would otherwise say "0.000 s"
ax_topo.set_title("")

# create additional axes (for ERF and colorbar)
divider = make_axes_locatable(ax_topo)

# add axes for colorbar
ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(image, cax=ax_colorbar)
ax_topo.set_xlabel(
    "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
)

# add new axis for time courses and plot time courses
ax_signals = divider.append_axes("right", size="300%", pad=1.2)
title = f"Cluster #1, {len(ch_inds)} sensor"
if len(ch_inds) > 1:
    title += "s (mean)"
mne.viz.plot_compare_evokeds(
    evokeds,
    title=title,
    picks=ch_inds,
    axes=ax_signals,
    colors=colors,
    show=False,
    split_legend=True,
    truncate_yaxis="auto",
)

# plot temporal cluster extent
ymin, ymax = ax_signals.get_ylim()
ax_signals.fill_betweenx(
    (ymin, ymax), sig_times[0], sig_times[-1], color="green", alpha=0.3
)

plt.show()


events = epochs.event_id.keys()

metadata = epochs.metadata

# Create a new column 'condition' to mark long and short words, with NaN for median values
contrast = metadata[name].apply(
    lambda x: 1 if x > float(median_value) else (0 if x < float(median_value) else np.nan)
)

# paired t-test for subjects P300 datasets

# external function for MNE user
def setup_dataframe(data: mne.Epochs | mne.Evokeds , events: list, contrast: list):
    """
    prepare dataframe for input to cluster_test() function.

    """
    df = pd.DataFrame()

    # add a column to the dataframe called data that contains the data
    df["data"] = data
    # add a column containing the event_id for each epoch
    df["events"] = events


    return df

# internal function
import mne.stats.cluster_level

def cluster_test(df: pd.Dataframe, paired: bool):
    
    all_data = {}

    # separate epochs for n conditions
    for ci,cond in enumerate(conditions):
        # Separate the epochs based on the condition
        all_data[cond] = epochs.data[ci]

    if paired == True:
        # Ensure equal number of entries per condition if paired t-test
        num_long = len(long_words_epochs)
        num_short = len(short_words_epochs)

        if num_long > num_short:
            # Randomly select indices to drop from long words epochs
            drop_indices = np.random.choice(num_long, num_long - num_short, replace=True)
            long_words_epochs = long_words_epochs.drop(drop_indices, inplace=True)
        elif num_short > num_long:
            # Randomly select indices to drop from short words epochs
            drop_indices = np.random.choice(num_short, num_short - num_long, replace=True)
            short_words_epochs = short_words_epochs.drop(drop_indices, inplace=True)

        assert len(short_words_epochs) == len(long_words_epochs)

    # has to be list for current cluster function
    X = np.array(cond_a) - np.array(cond_b)

    # Run the analysis
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        threshold=t_threshold,
        n_permutations=n_permutations,
        tail=tail,
        adjacency=adjacency,
        verbose=True,
        seed=123
    )

    return cluster_out

print(min(cluster_p_values))

lowest_p_cluster = np.argmin(cluster_p_values)

# configure variables for visualization
colors = {"long": "crimson", "short": "steelblue"}

# organize data for plotting
evokeds = {"long": long_words_evoked, "short": short_words_evoked}

# plot the cluster with the lowest p-value
time_inds, space_inds = np.squeeze(clusters[lowest_p_cluster])
ch_inds = np.unique(space_inds)
time_inds = np.unique(time_inds)

# get topography for F stat
t_map = T_obs[time_inds, ...].mean(axis=0)

# get signals at the sensors contributing to the cluster
sig_times = short_words.times[time_inds]

# create spatial mask
mask = np.zeros((t_map.shape[0], 1), dtype=bool)
mask[ch_inds, :] = True

# initialize figure
fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3), layout="constrained")

# plot average test statistic and mark significant sensors
t_evoked = mne.EvokedArray(t_map[:, np.newaxis], epochs.info, tmin=0)
t_evoked.plot_topomap(
    times=0,
    mask=mask,
    axes=ax_topo,
    cmap="Reds",
    vlim=(np.min, np.max),
    show=False,
    colorbar=False,
    mask_params=dict(markersize=10),
)
image = ax_topo.images[0]

# remove the title that would otherwise say "0.000 s"
ax_topo.set_title("")

# create additional axes (for ERF and colorbar)
divider = make_axes_locatable(ax_topo)

# add axes for colorbar
ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(image, cax=ax_colorbar)
ax_topo.set_xlabel(
    "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
)

# add new axis for time courses and plot time courses
ax_signals = divider.append_axes("right", size="300%", pad=1.2)
title = f"Cluster #1, {len(ch_inds)} sensor"
if len(ch_inds) > 1:
    title += "s (mean)"
mne.viz.plot_compare_evokeds(
    evokeds,
    title=title,
    picks=ch_inds,
    axes=ax_signals,
    colors=colors,
    show=False,
    split_legend=True,
    truncate_yaxis="auto",
)

# plot temporal cluster extent
ymin, ymax = ax_signals.get_ylim()
ax_signals.fill_betweenx(
    (ymin, ymax), sig_times[0], sig_times[-1], color="green", alpha=0.3
)

plt.show()