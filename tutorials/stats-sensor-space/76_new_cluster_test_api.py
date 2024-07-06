"""
.. _tut-new-cluster-test-api:

===============================================================
New cluster test API that allows for Wilkinson style formulas
===============================================================

This tutorial shows how to use the new API for cluster testing.
This script shows how to estimate significant clusters in
evoked contrast data of multiple subjects.
It uses a non-parametric statistical procedure based on permutations and
cluster level statistics.

The procedure consists of:

  - loading evoked data from multiple subjects
  - construct a dataframe that contains the difference between conditions
  - run the new cluster test function with formula in Wilkinson notation
  - plot the results with the ClusterResults Class

Here, the unit of observation are evokeds from multiple subjects (2nd level analysis).

For more information on cluster-based permutation testing in MNE-Python,
see also: :ref:`tut-cluster-one-samp-tfr`.
"""
# Authors: Carina Forster <carinaforster0611@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

from pathlib import Path

import pandas as pd

import mne

# Set parameters
# --------------
# Define the path to the P3 dataset
path_to_p3 = mne.datasets.misc.data_path() / "ERP_CORE" / "P3"

# Define the range of participant IDs
participant_ids = range(15, 20)  # This will cover 015 to 019

# store the evoked data of all subjects
evokeds_allsubs = []

# Loop over each participant ID and generate the corresponding filename
for pid in participant_ids:
    # Create the filename using an f-string, ID is zero-padded to 3 digits
    filename_p3 = f"sub-{pid:03d}_ses-P3_task-P3_ave.fif"

    # Create the full path to the file
    p3_file_path = Path(path_to_p3) / filename_p3

    # load the evoked data
    evokeds = mne.read_evokeds(p3_file_path)

    # add subjects evoked data to list
    evokeds_allsubs.append(evokeds)

# the P3b dataset is part of the freely available ERP CORE dataset
# participants were presented with a visual oddball task
# and the P3b component was analyzed
# the conditions of interest are the target (rare visual stimuli)
# and non-target stimuli (frequency visual stimuli)

# let's extract the target and non-target evokeds
target_only = [evoked[0] for evoked in evokeds_allsubs]
non_target_only = [evoked[1] for evoked in evokeds_allsubs]

# let's first have a look at the data
# create contrast from target and non-target evokeds
diff_evoked = [
    mne.combine_evoked([evokeds_a, evokeds_b], weights=[1, -1])
    for evokeds_a, evokeds_b in zip(target_only, non_target_only)
]

# plot the grand average of the difference signal
mne.grand_average(diff_evoked).plot()
# plot the topography of the difference signal
mne.grand_average(diff_evoked).plot_topomap()

# we can see that the strongest difference is around 400 ms in
# visual channels (occipital region)

# Next we prepare a dataframe for the cluster test function
# the dataframe should contain the contrast evoked data and the subject index
# each row in the dataframe should represent one observation (evoked data)

# save the evoked data for both conditions in one list
evokeds_conditions = target_only + non_target_only

# set up a list that defines the condition for each evoked data
# this will be used to create the conditions column in the dataframe
conditions = ["target"] * len(target_only) + ["non-target"] * len(non_target_only)

# finally add a column that defines the subject index
# this will be used to create the subject_index column in the dataframe
# we multiply the participant_ids by 2 to account for the two conditions
subject_index = list(participant_ids) * 2

# create the dataframe
df = pd.DataFrame(
    {
        "evoked": evokeds_conditions,
        "condition": conditions,
        "subject_index": subject_index,
    }
)

# now we can run the cluster test function
# we will use the new API that allows for Wilkinson style formulas
# the formula should be a string in Wilkinson notation

# we want to test whether there is a significant difference between
# target and non-target stimuli in the post-stimulus window
# we will use a cluster-based permutation paired t-test for this

# let's first define the formula based on Wilkinson notation
formula = "evoked ~ 1 + C(subject_index)"

# run the cluster test and return the cluster_result object
cluster_result = mne.stats.cluster_level.cluster_test(df=df, formula=formula)

# note that we ran an exact test due to the small sample size (only 15 permutations)

# set up conditions dictionary for cluster plots
conditions_dict = {"target": target_only, "non-target": non_target_only}

# finally let's plot the results using the ClusterResults class

# we plot the cluster with the lowest p-value

# we can see that there is something going on around 400 ms
# in the visual channels (topomap on the left)
# however the cluster is not significant which is unsurprising
# given the small sample size (only 5 subjects)
cluster_result.plot_cluster(cond_dict=conditions_dict)
