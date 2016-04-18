"""
Rejecting bad data (channels and segments)
==========================================

Bad channels
Bad segments (annotations)
Reject param in epochs

Link to interpolation

"""
# .. _marking_bad_channels:

# Marking bad channels
# --------------------

# Sometimes some MEG or EEG channels are not functioning properly
# for various reasons. These channels should be excluded from
# analysis by marking them bad as::

#     >>> raw.info['bads'] = ['MEG2443']

# Especially if a channel does not show
# a signal at all (flat) it is important to exclude it from the
# analysis, since its noise estimate will be unrealistically low and
# thus the current estimate calculations will give a strong weight
# to the zero signal on the flat channels and will essentially vanish.
# It is also important to exclude noisy channels because they can
# possibly affect others when signal-space projections or EEG average electrode
# reference is employed. Noisy bad channels can also adversely affect
# averaging and noise-covariance matrix estimation by causing
# unnecessary rejections of epochs.

# Recommended ways to identify bad channels are:

# - Observe the quality of data during data
#     acquisition and make notes of observed malfunctioning channels to
#     your measurement protocol sheet.

# - View the on-line averages and check the condition of the channels.

# - Compute preliminary off-line averages with artifact rejection,
#     SSP/ICA, and EEG average electrode reference computation
#     off and check the condition of the channels.

# - View raw data with :func:`mne.io.Raw.plot` without SSP/ICA
#     enabled and identify bad channels.

# .. note::
#     It is strongly recommended that bad channels are identified and
#     marked in the original raw data files. If present in the raw data
#     files, the bad channel selections will be automatically transferred
#     to averaged files, noise-covariance matrices, forward solution
#     files, and inverse operator decompositions.

# The actual removal happens using :func:`pick_types <mne.pick_types>` with
# `exclude='bads'` option (see :ref:`picking_channels`).

# Instead of removing the bad channels, you can also interpolate the data from
# other channels, to correct the bad channels.

#     >>> # compute interpolation (also works with Raw and Epochs objects)
#     >>> evoked.interpolate_bads(reset_bads=False)

# .. figure:: ../../../../_images/sphx_glr_plot_interpolate_bad_channels_001.png
#     :target: ../../auto_examples/preprocessing/plot_interpolate_bad_channels.html
#     :scale: 30%
# .. figure:: ../../../../_images/sphx_glr_plot_interpolate_bad_channels_002.png
#     :target: ../../auto_examples/preprocessing/plot_interpolate_bad_channels.html
#     :scale: 30%

# .. topic:: Examples:

#     * :ref:`sphx_glr_auto_examples_time_frequency_pplot_interpolate_bad_channels.py`
