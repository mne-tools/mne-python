from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne

# eventually we want to use the _permutation_cluster_test function

# import and load dataset
path_to_p3 = Path("C:/Users/Carina/mne_data/ERP_CORE_P3")


def prep_sample_data(plot_evokeds: bool = False):
    """
    Load the P3 dataset and extract the target, non-target and contrast evokeds.
    """
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

    if plot_evokeds:
        # plot the grand average
        mne.grand_average(target_only).plot()
        mne.grand_average(non_target_only).plot()
        mne.grand_average(contrast).plot()

    # create contrast from evokeds target and non-target
    diff_evoked = [
        mne.combine_evoked([evokeds_a, evokeds_b], weights=[1, -1])
        for evokeds_a, evokeds_b in zip(target_only, non_target_only)
    ]

    if plot_evokeds:
        mne.grand_average(diff_evoked).plot()

    # crop the evokeds in the post stimulus window
    contrast = [evokeds.crop(tmin=-0.1, tmax=0.6) for evokeds in contrast]
    target_only = [evokeds.crop(tmin=-0.1, tmax=0.6) for evokeds in target_only]
    non_target_only = [evokeds.crop(tmin=-0.1, tmax=0.6) for evokeds in non_target_only]

    return contrast, target_only, non_target_only


def old_api_cluster(n_permutations: int = 10000, seed: int = 1234):
    """
    Run the cluster test using the old API to get a bechmark result for the new API.
    Currently implementing a paired t-test with contrast between participants.
    """
    contrast, target_only, non_target_only = prep_sample_data()

    # extract the data for each evoked and store in numpy array
    data = np.array([evoked.data for evoked in contrast])

    # shape should be (n_subjects, n_channels, n_times)
    data.shape

    # reshape to channels as last dimension
    data = data.transpose(0, 2, 1)

    data.shape

    adjacency, _ = mne.channels.find_ch_adjacency(contrast[0].info, ch_type="eeg")

    stat_fun, threshold = mne.stats.cluster_level._check_fun(
        X=data, stat_fun=None, threshold=None, tail=0, kind="within"
    )

    # adjacency = mne.channels.find_ch_adjacency(contrast[0].info, ch_type='eeg')
    # Run the analysis
    T_obs, clusters, cluster_p_values, H0 = (
        mne.stats.cluster_level._permutation_cluster_test(
            [data],
            threshold=threshold,
            stat_fun=stat_fun,
            n_jobs=-1,  # takes all CPU cores
            max_step=1,  # maximum distance between samples (time points)
            exclude=None,  # exclude no time points or channels
            step_down_p=0,  # step down in jumps test
            t_power=1,  # weigh each location by its stats score
            out_type="indices",
            check_disjoint=False,
            buffer_size=None,  # block size for chunking the data
            n_permutations=n_permutations,
            tail=0,
            adjacency=adjacency,
            seed=seed,
        )
    )

    print(min(cluster_p_values))

    plot_cluster(
        contrast, target_only, non_target_only, T_obs, clusters, cluster_p_values
    )

    return T_obs, clusters, cluster_p_values, H0


# fit cluster test with dataframe as input
# create condition list that repeats 5times 1 and then 5 times 0
# 1 = target, 0 = non-target
# condition = 5 * [1] + 5 * [0]

# 1 = target, 0 = non-target
# contrast, target_only, non_target_only = prep_sample_data()

# evokeds_list = target_only + non_target_only


def create_random_evokeds_id_condition_list(evoked_data_a: list, evoked_data_b: list):
    """
    Create a list of shuffled participant IDs, conditions, and evoked data.
    # Keep the participant IDs and conditions paired but shuffle the order of the evoked data.
    """
    import random

    # Example participant IDs
    participant_ids = ["p1", "p2", "p3", "p4", "p5"] * 2

    # Combine the evoked data into a single list
    all_evoked_data = evoked_data_a + evoked_data_b

    # Create a corresponding list of conditions
    conditions = [1] * len(evoked_data_a) + [0] * len(evoked_data_b)

    # Combine the participant IDs, conditions, and evoked data into a list of tuples
    combined_list = list(zip(participant_ids, conditions, all_evoked_data))

    # Shuffle the combined list
    random.shuffle(combined_list)

    # Separate the shuffled list back into participant IDs, conditions, and evoked data
    shuffled_participant_ids, shuffled_conditions, shuffled_evoked_data = zip(
        *combined_list
    )

    # Convert the tuples back to lists
    shuffled_participant_ids = list(shuffled_participant_ids)
    shuffled_conditions = list(shuffled_conditions)
    shuffled_evoked_data = list(shuffled_evoked_data)

    return shuffled_participant_ids, shuffled_conditions, shuffled_evoked_data


def create_random_paired_evokeds_list(evoked_data_a: list, evoked_data_b: list):
    """
    Create a list of shuffled evoked data where each pair of target and non-target evoked data is shuffled together.
    """
    import random

    # Create a list of tuples where each tuple contains an evoked data and its corresponding label
    evoked_pairs = [(evoked, 1) for evoked in evoked_data_a] + [
        (evoked, 0) for evoked in evoked_data_b
    ]

    # Shuffle the list of tuples
    random.shuffle(evoked_pairs)

    # Separate the shuffled list back into evoked data and labels
    shuffled_evoked_data, shuffled_labels = zip(*evoked_pairs)

    # Convert the tuples back to lists
    shuffled_evoked_data = list(shuffled_evoked_data)

    return shuffled_evoked_data


# shuffle order of pairs
shuffled_evokeds_list = create_random_paired_evokeds_list(target_only, non_target_only)
# shouldn't change the results (p-value is different though?)

shuffled_participant_ids, shuffled_conditions, shuffled_evoked_data = (
    create_random_evokeds_id_condition_list(
        evoked_data_a=target_only, evoked_data_b=non_target_only
    )
)


def prepare_dataframe_for_cluster_function(
    contrast: bool = False,
    evokeds: list = None,
    condition: list = None,
    subject_index: list = None,
):
    """
    Prepare a dataframe for the cluster test function.

    Parameters
    ----------
    contrast : bool, optional
        If True, a contrast is calculated. Default is False.
    evokeds : list, optional
        List of evoked objects. Default is None.
    condition : list, optional
        List of conditions for each evoked object. Default is None.
    subject_index : list, optional
        List of subject IDs. Default is None.

    """
    # create an empty dataframe
    df = pd.DataFrame()

    if contrast == True:
        # check if evoked list is dividable by 2
        if len(evokeds) % 2 != 0:
            raise ValueError("evokeds list needs to be dividable by 2")
        if condition is not None:
            # Convert lists to DataFrame for easier manipulation
            df = pd.DataFrame(
                {
                    "evoked": evokeds,
                    "condition": condition,
                    "subject_index": subject_index,
                }
            )

        return df


def cluster_test(
    df: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int = 1234,
    contrast_weights: list = [1, -1],
):
    """
    Run the cluster test using the new API.
    # currently supports paired t-test with contrast or with list of conditions

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with evoked data, conditions and subject IDs.
    n_permutations : int, optional
        Number of permutations. Default is 10000.
    seed : int, optional
        Random seed. Default is 1234.

    Returns
    -------
    T_obs : array
        The observed test statistic.
    clusters : list
        List of clusters.
    cluster_p_values : array
        Array of cluster p-values.
    H0 : array
        The permuted test statistics.
    """
    if df.condition is not None:
        # Extract unique conditions
        unique_conditions = np.unique(df.condition)
        if len(unique_conditions) != 2:
            raise ValueError("Condition list needs to contain 2 unique values")
        if df.subject_index is not None:
            # Initialize a list to hold the combined evoked data
            evokeds_data = []

            # Process each subject's evoked data
            for sub_id in df.subject_index.unique():
                sub_df = df[df.subject_index == sub_id]

                # Split evokeds list based on condition list for this subject
                evokeds_a = sub_df[sub_df.condition == unique_conditions[0]][
                    "evoked"
                ].tolist()
                evokeds_b = sub_df[sub_df.condition == unique_conditions[1]][
                    "evoked"
                ].tolist()

                if len(evokeds_a) != 1 or len(evokeds_b) != 1:
                    raise ValueError(
                        f"Subject {sub_id}: Each subject must have exactly one evoked for each condition"
                    )

                # Calculate contrast based on condition list
                diff_evoked = mne.combine_evoked(
                    [evokeds_a[0], evokeds_b[0]], weights=contrast_weights
                )
                evokeds_data.append(diff_evoked)
        else:
            # calculate length of evokeds list
            n_evokeds = len(df.evokeds)
            # now split evokeds list in two lists
            evokeds_a = df.evokeds[: n_evokeds // 2]
            evokeds_b = df.evokeds[n_evokeds // 2 :]
            # create contrast from evokeds_a and evokeds_b
            diff_evoked = [
                mne.combine_evoked([evo_a, evo_b], weights=contrast_weights)
                for evo_a, evo_b in zip(evokeds_a, evokeds_b)
            ]
            evokeds_data = diff_evoked
    else:
        evokeds_data = df.evokeds

    # extract number of channels
    n_channels = evokeds_data[0].info["nchan"]

    # loop over rows and extract data from evokeds
    data_array = np.array([evoked.data for evoked in evokeds_data])

    # find the dimension that is equal to n_channels
    if data_array.shape[1] == n_channels:
        # reshape to channels as last dimension
        data = data_array.transpose(0, 2, 1)

    adjacency, _ = mne.channels.find_ch_adjacency(evokeds_data[0].info, ch_type="eeg")

    stat_fun, threshold = mne.stats.cluster_level._check_fun(
        X=data, stat_fun=None, threshold=None, tail=0, kind="within"
    )

    T_obs, clusters, cluster_p_values, H0 = (
        mne.stats.cluster_level._permutation_cluster_test(
            [data],
            threshold=threshold,
            stat_fun=stat_fun,
            n_jobs=-1,
            max_step=1,
            exclude=None,
            step_down_p=0.05,
            t_power=1,
            out_type="indices",
            check_disjoint=True,
            buffer_size=None,
            n_permutations=n_permutations,
            tail=0,
            adjacency=adjacency,
            seed=seed,
        )
    )

    print(min(cluster_p_values))

    # need to adjust plotting function for contrast only data
    contrast, evokeds_a, evokeds_b = prep_sample_data()

    # plot cluster
    plot_cluster(contrast, evokeds_a, evokeds_b, T_obs, clusters, cluster_p_values)

    return T_obs, clusters, cluster_p_values, H0


def plot_cluster(
    contrast, target_only, non_target_only, T_obs, clusters, cluster_p_values
):
    """
    Plot the cluster with the lowest p-value.

    Parameters
    ----------
    contrast : list
        List of contrast evoked objects.
    target_only : list
        List of target evoked objects.
    non_target_only : list
        List of non-target evoked objects.
    T_obs : array
        The observed test statistic.
    clusters : list
        List of clusters.
    cluster_p_values : array
        Array of cluster p-values.

    Returns
    -------
    None

    """
    # configure variables for visualization
    colors = {"target": "crimson", "non-target": "steelblue"}

    # organize data for plotting
    evokeds = {"target": target_only, "non-target": non_target_only}

    lowest_p_cluster = np.argmin(cluster_p_values)

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

    return None
