"""
.. _tut-new-cluster-test-api:

===============================================================
New cluster test API that allows for Wilkinson style formulas
===============================================================

This tutorial shows how to use the new API for cluster testing.
The new API allows for Wilkinson style formulas and allows for more flexibility in
the design of the test. Here we will demonstrate how to use the new API for
a standard paired t-test on evoked data from multiple subjects.
It uses a non-parametric statistical procedure based on permutations and
cluster level statistics.

The procedure consists of:

  - loading evoked data from multiple subjects
  - construct a dataframe that contains the difference between conditions
  - run the new cluster test function with formula in Wilkinson notation
  - plot the results with the new ClusterResults API

Here, the unit of observation are evokeds from multiple subjects (2nd level analysis).

For more information on cluster-based permutation testing in MNE-Python,
see also: :ref:`tut-cluster-one-samp-tfr`.
"""
# Authors: Carina Forster <carinaforster0611@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %% Load the required packages

from pathlib import Path

import pandas as pd

import mne

# %% Load the P3 dataset

# Set parameters
# --------------
# Define the path to the P3 dataset
path_to_p3 = mne.datasets.misc.data_path() / "ERP_CORE" / "P3"

# Define the range of participant IDs (we only have 5 participants in the dataset)
participant_ids = range(15, 20)  # This will cover participant 15 to 19

# store the evoked data of all subjects
evokeds_allsubs = []

# Loop over each participant ID and generate the corresponding filename
# to load the evoked data
for pid in participant_ids:
    # Create the filename using an f-string, ID is zero-padded to 3 digits
    filename_p3 = f"sub-{pid:03d}_ses-P3_task-P3_ave.fif"

    # Create the full path to the file
    p3_file_path = Path(path_to_p3) / filename_p3

    # load the evoked data
    evokeds = mne.read_evokeds(p3_file_path)

    # add single subjects evoked data to a list
    evokeds_allsubs.append(evokeds)

# the P3b dataset is part of the freely available ERP CORE dataset
# participants were presented with a visual oddball task
# and the P3b component was analyzed
# the conditions of interest are the target (rare visual stimuli)
# and non-target stimuli (frequent visual stimuli)

# %% visually inspect the evoked data for each condition

# let's extract the target and non-target evokeds
target_only = [evoked[0] for evoked in evokeds_allsubs]
non_target_only = [evoked[1] for evoked in evokeds_allsubs]

# let's first have a look at the data

# create contrast target - non-target
diff_evoked = [
    mne.combine_evoked([evokeds_a, evokeds_b], weights=[1, -1])
    for evokeds_a, evokeds_b in zip(target_only, non_target_only)
]

# plot the grand average of the difference signal
mne.grand_average(diff_evoked).plot()
# plot the topography of the difference signal
mne.grand_average(diff_evoked).plot_topomap()

# we can see that the strongest difference is around 400 ms in
# central-parietal channels with a stronger evoked signal for target stimuli

# %% Prepare the dataframe for the new cluster test API

# the dataframe should contain the contrast evoked data and the subject index
# each row in the dataframe should represent one observation (evoked data)

# save the evoked data for both conditions in one list
evokeds_conditions = target_only + non_target_only

# create a list that defines the condition for each evoked data
# this will be used to create the conditions column in the dataframe
conditions = ["target"] * len(target_only) + ["non-target"] * len(non_target_only)

# finally add a column that defines the subject index
# this will be used to create the subject_index column in the dataframe
# we multiply the participant_ids by 2 to account for the two conditions
subject_index = list(participant_ids) * 2

# create the dataframe containing the evoked data, the condition and the subject index
df = pd.DataFrame(
    {
        "evoked": evokeds_conditions,
        "condition": conditions,
        "subject_index": subject_index,
    }
)

# %% run the cluster test function with formulaic input

# we will use the new API that allows for Wilkinson style formulas
# the formula should be a string in Wilkinson notation

# we want to test whether there is a significant difference between
# target and non-target stimuli in the post-stimulus window
# we will use a cluster-based permutation paired t-test for this

# let's first define the formula based on Wilkinson notation
# we want to predict the evoked difference signal based on the subject
# the cluster test randomly permutes the subject label
# the 1 in the formula represents the intercept which is always included
# C is a categorical variable that will be dummy coded
formula = "evoked ~ condition"

# run the new cluster test API and return the new cluster_result object
cluster_result = mne.stats.cluster_level.cluster_test(
    df=df, formula=formula, within_id="subject_index"
)
# TODO: add n_permutations to cluster_result

# print the lowest cluster p-value
print(f"The lowest cluster p-value is: {cluster_result.cluster_p_values.min()}")

# note that we ran an exact test due to the small sample size
# (only 15 permutations)

# %% plot the results

# set up conditions dictionary for cluster plots
# this is necessary for plotting the evoked data and the cluster result on top
conditions_dict = {"target": target_only, "non-target": non_target_only}

# finally let's plot the results using the ClusterResults class

# we plot the cluster with the lowest p-value
cluster_result.plot_cluster_time_sensor(condition_labels=conditions_dict, ci=True)
# we can see that there is something going on around 400 ms
# with a stronger signal for target trials in right central-parietal channels

# however the cluster is not significant which is unsurprising
# given the small sample size (only 5 subjects)
