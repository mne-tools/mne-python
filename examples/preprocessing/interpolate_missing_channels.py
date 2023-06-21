"""
.. _ex-interpolate-missing-channels:

================================================================
Reconstruct and interpolate a missing channel for MEG recordings
================================================================

Sometimes, a noisy bad MEG channel can be completely turned off during
an experimental recording on purpose. This is mainly done inorder to stop
bad channel noises to spread over other channels. As a result, physical
location of the channel is completely missing from the recording data.
Therefore, the bad channel can't be interpolated anymore.

This example shows how to:

- Reconstruct a MEG channel which is turned off during the recording.
- Interpolate the reconstruct channel using field interpolation.

In this example, one gradiometer channel will be dropped on purpose to
mimic a missing channel scenario in practice. Only the data in that channel
is reconstructed and later replaced by interpolation.
"""
# Authors: Diptyajit Das <bmedasdiptyajit@gmail.com>
#          co-author name if required.
#
# License: BSD-3-Clause

# %%

import mne
from mne.datasets import sample
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as grd

print(__doc__)

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
fname = meg_path / "sample_audvis-ave.fif"
evoked = mne.read_evokeds(fname, condition="Left Auditory", baseline=(None, 0))
# pick only gradiometer channels
evoked = evoked.copy().pick_types(meg="grad")

# %%
# Lets drop one channel (we choose 'MEG 0212') to create a missing channel scenario.
# In practice, this can be any bad channel which is turned off during recording.
ch_names = evoked.info["ch_names"]
drop_ch = "MEG 0212"
drop_ch_pos = ch_names.index(drop_ch)
# Our missing channel data
evoked_drop = evoked.copy().drop_channels(ch_names[drop_ch_pos])

# %%
# Lets create a flat gradiometer channel to reconstruct the missing channel.
# We will use the EvokedArray class constructor to do this.
sampling_freq = evoked_drop.info["sfreq"]
ch_types = "grad"
info = mne.create_info([drop_ch], ch_types=ch_types, sfreq=sampling_freq)
times = evoked.times
data = np.arange(len(times)).reshape(1, len(times))
flat_channel_data = np.zeros_like(data)
flat_evoked = mne.EvokedArray(
    flat_channel_data,
    info,
    tmin=times[0],
    nave=flat_channel_data.shape[0],
    comment="reconstructed channel",
)
evoked_recon = evoked_drop.add_channels([flat_evoked], force_update_info=True)
# plot evoked with reconstructed channel
evoked_recon.plot()

# %%
# Now, we have to update two channel field parameters (i.e., channel description and
# channel order). We will replace the reconstructed channel description with the
# original one. This can also be added from the built-in MEG layout, currently available
# in MNE.
evoked_recon.info["chs"][-1] = evoked.info["chs"][drop_ch_pos]
# Reorder the channels
evoked_recon.reorder_channels(ch_names)
# Since our reconstructed is still a flat channel, we add the channel as a bad
# channel manually to perform interpolation.
evoked_recon.info["bads"] = [drop_ch]
evoked_interp = evoked_recon.interpolate_bads(reset_bads=False)

# %%
# Plot original vs interpolated channels
fig = plt.figure(figsize=(10, 8), constrained_layout=True)
gs = grd.GridSpec(ncols=1, nrows=1, figure=fig)
ax = fig.add_subplot(gs[0, 0])
conditions = ("Original", "Interpolated")
evokeds = dict(zip(conditions, [evoked, evoked_interp]))
x_fills = [50, 150]  # 50-150 ms (highlight auditory activity)
mne.viz.plot_compare_evokeds(
    evokeds, axes=ax, picks=drop_ch, legend=True, show=False, time_unit="ms"
)
ax.axvspan(x_fills[0], x_fills[1], alpha=0.5, color="grey")
plt.margins(y=0.2)
plt.show()
