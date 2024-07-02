"""
.. _tut-new-cluster-test-api:

====================
New cluster test API
====================

This tutorial shows how to use the new API for cluster testing.
"""
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mne
from mne.utils import _soft_import

# TODO: test function and update docstrings

# import and load dataset
path_to_p3 = mne.datasets.misc.data_path() / "ERP_CORE" / "P3"


def prep_sample_data(plot_evokeds: bool = False):
    """Load the P3 dataset."""
    # Define the range of participant IDs
    participant_ids = range(15, 20)  # This will cover 015 to 019

    evokeds_allsubs = []

    # Loop over each participant ID and generate the corresponding filename
    for pid in participant_ids:
        # Create the filename using an f-string, ID is zero-padded to 3 digits
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
    Run the cluster test using the old API to get a benchmark result for the new API.

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


def create_random_evokeds_id_condition_list():
    """
    Create a list of shuffled participant IDs, conditions, and evoked data.

    # Keep the participant IDs and conditions paired but shuffle the order of the data.
    """
    import random

    _, evoked_data_a, evoked_data_b = prep_sample_data()

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


def create_random_paired_evokeds_list():
    """
    Create shuffled paired evoked data.

    Create a list of shuffled evoked data where each pair of target
    and non-target evoked data is shuffled together.
    """
    import random

    _, evoked_data_a, evoked_data_b = prep_sample_data()

    # Ensure evoked_data_a and evoked_data_b are of the same length
    assert len(evoked_data_a) == len(
        evoked_data_b
    ), "evoked_data_a and evoked_data_b must have the same length"

    # Create a list of participant indices
    participant_indices = list(range(len(evoked_data_a)))

    # Shuffle the list of participant indices
    random.shuffle(participant_indices)

    # Reorder evoked data according to the shuffled participant indices
    shuffled_evoked_data_a = [evoked_data_a[i] for i in participant_indices]
    shuffled_evoked_data_b = [evoked_data_b[i] for i in participant_indices]

    # Combine the shuffled evoked data into a single list
    shuffled_evoked_data = shuffled_evoked_data_a + shuffled_evoked_data_b

    # Combine the original evoked data into a single list
    original_evoked_data = evoked_data_a + evoked_data_b

    return original_evoked_data, shuffled_evoked_data


# shuffle order of pairs
original_evoked_data, shuffled_evoked_data = create_random_paired_evokeds_list()
# shouldn't change the results (p-value is different though?)

shuffled_participant_ids, shuffled_conditions, shuffled_evoked_data = (
    create_random_evokeds_id_condition_list()
)


def prepare_dataframe_for_cluster_function(
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

    Returns
    -------
    df : DataFrame
        The prepared DataFrame for the cluster test function.
    """
    # Initialize the DataFrame with evoked data
    df = pd.DataFrame(
        {
            "evoked": evokeds,
            "condition": condition if condition is not None else pd.NA,
            "subject_index": subject_index if subject_index is not None else pd.NA,
        }
    )

    return df


df = prepare_dataframe_for_cluster_function(
    evokeds=shuffled_evoked_data,
    condition=shuffled_conditions,
    subject_index=shuffled_participant_ids,
)


def cluster_test(
    df: pd.DataFrame,
    formula: str = None,  # Wilkinson notation formula for design matrix
    n_permutations: int = 10000,
    seed: None | int | np.random.RandomState = None,
    tail: int = 0,  # 0 for two-tailed, 1 for greater, -1 for less
    n_jobs: int = 1,  # how many cores to use
    adjacency: tuple = None,
    max_step: int = 1,  # maximum distance between samples (time points)
    exclude: list = None,  # exclude no time points or channels
    step_down_p: int = 0,  # step down in jumps test
    t_power: int = 1,  # weigh each location by its stats score
    out_type: str = "indices",
    check_disjoint: bool = False,
    buffer_size: int = None,  # block size for chunking the data
):
    """
    Run the cluster test using the new API.

    # currently supports paired t-test

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with evoked data, conditions and subject IDs.
    formula : str, optional
        Wilkinson notation formula for design matrix. Default is None.
    n_permutations : int, optional
        Number of permutations. Default is 10000.
    seed : None | int | np.random.RandomState, optional
        Seed for the random number generator. Default is None.
    tail : int, optional
        0 for two-tailed, 1 for greater, -1 for less. Default is 0.
    n_jobs : int, optional
        How many cores to use. Default is 1.
    adjacency : None, optional
        Adjacency matrix. Default is None.
    max_step : int, optional
        Maximum distance between samples (time points). Default is 1.
    exclude : np.Array, optional
        Exclude no time points or channels. Default is None.
    step_down_p : int, optional
        Step down in jumps test. Default is 0.
    t_power : int, optional
        Weigh each location by its stats score. Default is 1.
    out_type : str, optional
        Output type. Default is "indices".
    check_disjoint : bool, optional
        Check if clusters are disjoint. Default is False.
    buffer_size : int, optional
        Block size for chunking the data. Default is None.
    seed : int, optional
        Seed for the random number generator. Default is None.

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
    # for now this assumes a dataframe with a column for evoked data
    # add a data column to the dataframe (numpy array)
    df["data"] = [evoked.data for evoked in df.evoked]

    # extract number of channels and timepoints
    # (eventually should also allow for frequency)
    n_channels, n_timepoints = df["data"][0].shape

    # convert wide format to long format for formulaic
    df_long = unpack_time_and_channels(df)

    # Pivot the DataFrame
    pivot_df = df_long.pivot_table(
        index=["subject_index", "channel", "timepoint"],
        columns="condition",
        values="value",
    ).reset_index()

    # if not 2 unique conditions raise error
    if len(pd.unique(df.condition)) != 2:
        raise ValueError("Condition list needs to contain 2 unique values")

    # Compute the difference (assuming there are only 2 conditions)
    pivot_df["y"] = pivot_df[0] - pivot_df[1]

    # Optional: Clean up the DataFrame
    pivot_df = pivot_df[["subject_index", "channel", "timepoint", "y"]]

    # check if formula is present
    if formula is not None:
        formulaic = _soft_import(
            "formulaic", purpose="set up Design Matrix"
        )  # soft import (not a dependency for MNE)

        # for the paired t-test y is the difference between conditions
        # X is the design matrix with a column with 1s and 0s for each participant
        # Create the design matrix using formulaic
        y, X = formulaic.model_matrix(formula, pivot_df)
    else:
        raise ValueError(
            "Formula is required and needs to be a string in Wilkinson notation."
        )

    # now prep design matrix outcome variable for input into MNE cluster function
    # we initially had first channels, then timepoints,
    # now we need first timepoints, then channels
    y_for_cluster = y.values.reshape(-1, n_channels, n_timepoints).transpose(0, 2, 1)

    adjacency, _ = mne.channels.find_ch_adjacency(df["evoked"][0].info, ch_type="eeg")

    # define stat function and threshold
    stat_fun, threshold = mne.stats.cluster_level._check_fun(
        X=y_for_cluster, stat_fun=None, threshold=None, tail=0, kind="within"
    )

    # Run the cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = (
        mne.stats.cluster_level._permutation_cluster_test(
            [y_for_cluster],
            n_permutations=10000,
            threshold=threshold,
            stat_fun=stat_fun,
            tail=tail,
            n_jobs=n_jobs,
            adjacency=adjacency,
            max_step=max_step,  # maximum distance between samples (time points)
            exclude=exclude,  # exclude no time points or channels
            step_down_p=step_down_p,  # step down in jumps test
            t_power=t_power,  # weigh each location by its stats score
            out_type=out_type,
            check_disjoint=check_disjoint,
            buffer_size=buffer_size,  # block size for chunking the data
            seed=seed,
        )
    )

    print(min(cluster_p_values))

    # need to adjust plotting function for contrast only data
    contrast, evokeds_a, evokeds_b = prep_sample_data()

    # plot cluster
    plot_cluster(contrast, evokeds_a, evokeds_b, T_obs, clusters, cluster_p_values)

    return T_obs, clusters, cluster_p_values, H0


def unpack_time_and_channels(df):
    """
    Extract the time and channel data from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame in wide format.
    """
    # Extracting all necessary data using list comprehensions for better performance
    long_format_data = [
        {
            "condition": row["condition"],
            "subject_index": row["subject_index"],
            "channel": channel,
            "timepoint": timepoint,
            "value": row["data"][channel, timepoint],
        }
        for idx, row in df.iterrows()
        for channel in range(row["data"].shape[0])
        for timepoint in range(row["data"].shape[1])
    ]

    # Creating the long format DataFrame
    df_long = pd.DataFrame(long_format_data)

    return df_long


# Example usage
# Sample wide format DataFrame
df_wide = pd.DataFrame(
    {
        "condition": ["A", "B"],
        "subject_index": [1, 2],
        "data": [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])],
    }
)


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


# translated the limo permutation ttest from matlab to python
def limo_ttest_permute(Data, n_perm=None):
    """
    Pseudo one-sample t-test using sign-test with permutations.

    Parameters
    ----------
    Data (numpy.ndarray): A matrix of data for the one-sample t-test.
                          Shape can be (n_channels, n_var, n_obs) or
                          (n_var, n_obs).
                        n_perm (int, optional): Number of permutations to perform.
    If None, it defaults based on the number of observations.

    Returns
    -------
    t_vals (numpy.ndarray): t-values under H0.
    p_vals (numpy.ndarray): p-values under H0.
    dfe (int): Degrees of freedom.
    """
    # Check inputs and reshape if necessary
    if Data.ndim == 3:
        n_channels, n_var, n_obs = Data.shape
    else:
        n_channels = 1
        n_var, n_obs = Data.shape
        Data = Data[np.newaxis, ...]

    # Warn if the number of observations is very small
    if n_obs < 7:
        n_psbl_prms = 2**n_obs
        print(
            f"Due to the very limited number of observations, "
            f"the total number of possible permutations is small ({n_psbl_prms}). "
            "Thus, only a limited number of p-values are possible "
            "and the test might be overly conservative."
        )

    # Set up permutation test
    if n_obs <= 12:
        n_perm = 2**n_obs  # total number of possible permutations
        exact = True
        print(
            "Due to the limited number of observations, all possible permutations "
            "of the data will be computed instead of random permutations."
        )
    else:
        exact = False
        if n_perm is None:
            n_perm = 1000

    print(f"Executing permutation test with {n_perm} permutations...")

    # Initialize variables
    t_vals = np.full(
        (n_channels, n_var, n_perm), np.nan
    )  # Array to store t-values for each permutation
    sqrt_nXnM1 = np.sqrt(
        n_obs * (n_obs - 1)
    )  # Precompute constant for t-value calculation
    dfe = n_obs - 1  # Degrees of freedom

    if exact:
        # Use all possible permutations
        for perm in range(n_perm):
            # Set sign of each trial / participant's data
            temp = np.array(
                [int(x) for x in bin(perm)[2:].zfill(n_obs)]
            )  # Convert perm index to binary array
            sn = np.where(temp == 0, -1, 1)  # Map 0 to -1 and 1 to 1
            sn_mtrx = np.tile(sn, (n_var, 1)).T  # Repeat sn for each variable

            for c in range(n_channels):
                data = Data[c, :, :]
                d_perm = data * sn_mtrx  # Apply sign flip to data

                # Compute t-score of permuted data
                sm = np.sum(d_perm, axis=1)  # Sum of permuted data
                mn = sm / n_obs  # Mean of permuted data
                sm_sqrs = (
                    np.sum(d_perm**2, axis=1) - (sm**2) / n_obs
                )  # Sum of squares for standard error
                stder = np.sqrt(sm_sqrs) / sqrt_nXnM1  # Standard error
                t_vals[c, :, perm] = mn / stder  # Compute t-values

    else:
        # Use random permutations
        for perm in range(n_perm):
            # Randomly set sign of each trial / participant's data
            sn = (np.random.rand(n_obs) > 0.5) * 2 - 1  # Generate random sign flips
            sn_mtrx = np.tile(sn, (n_var, 1))  # Repeat sn for each variable

            for c in range(n_channels):
                data = Data[c, :, :]
                d_perm = data * sn_mtrx  # Apply sign flip to data

                # Compute t-score of permuted data
                sm = np.sum(d_perm, axis=1)  # Sum of permuted data
                mn = sm / n_obs  # Mean of permuted data
                sm_sqrs = (
                    np.sum(d_perm**2, axis=1) - (sm**2) / n_obs
                )  # Sum of squares for standard error
                stder = np.sqrt(sm_sqrs) / sqrt_nXnM1  # Standard error
                t_vals[c, :, perm] = mn / stder  # Compute t-values

    # Compute p-values from t-values
    p_vals = 2 * scipy.stats.cdf(-np.abs(t_vals), dfe)

    return t_vals, p_vals, dfe
