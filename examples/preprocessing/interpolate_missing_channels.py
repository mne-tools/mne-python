"""
.. _ex-interpolate-missing-channels:
=========================================================
Reconstruct and interpolate missing channels for MEG data
=========================================================
Sometimes, noisy bad MEG channels can be completely turned off during
an experimental recording on purpose. This is mainly done inorder to stop
noise signal spreading towards other channels. As a result, physical
locations of those channels are completely missing from the recording data.
Therefore, bad channels can't be interpolated anymore.

This example shows how to:
- Reconstruct MEG channels which are turned off during the recording.
- Interpolate the reconstruct channels using field interpolation.

In this example, two MEG channels (i.e., gradiometer and magnetometer) are dropped
on purpose to mimic the missing channels scenario in practice. Only those channels
are reconstructed and later the channel data are replaced using interpolation.
"""
# Author: Diptyajit Das <bmedasdiptyajit@gmail.com>
#
# License: BSD-3-Clause

# %%

import mne
from mne.datasets import sample
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec as grd


print(__doc__)

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
fname = meg_path / "sample_audvis-ave.fif"
evoked = mne.read_evokeds(fname, condition="Left Auditory", baseline=(None, 0))
# pick only meg channels
evoked = evoked.copy().pick_types(meg=True)

# %%
# Lets drop two MEG channels to create missing channel's scenario.
# Here, we choose "MEG 0212" and "MEG 1321" as the missing channels. In practice,
# this can be any noisy channel that was turned off during recording.
ch_names = evoked.info["ch_names"]
chs_drop = ["MEG 0212", "MEG 1321"]

# evoked with missing channels
grad_evoked_mis = evoked.copy().drop_channels(chs_drop)


# %%
# Lets create two flat channels to reconstruct the missing channels.
# We will use the EvokedArray class constructor to do this.
sampling_freq = evoked.info["sfreq"]
ch_types = ["grad", "mag"]

info = mne.create_info(chs_drop, ch_types=ch_types, sfreq=sampling_freq)
times = evoked.times
flat_channel_data = np.zeros((len(chs_drop), len(times)))

flat_evoked = mne.EvokedArray(
    flat_channel_data,
    info,
    tmin=times[0],
    nave=flat_channel_data.shape[0],
    comment="reconstructed channel",
)

# Now we have to update channels location. For this, we will use MNE's
# in-built MEG layout configuration

file = (
    "/home/dip_linux/PycharmProjects/mne-python/mne/io/fiff/"
    "VectorView_mne_loc.json"
)

with open(file, "r") as f:
    chs_loc = json.load(f)

for itype, ch in enumerate(chs_drop):
    flat_evoked.info["chs"][itype]["loc"] = np.array(chs_loc[ch])

evoked_recon = grad_evoked_mis.add_channels([flat_evoked], force_update_info=True)

# plot evoked with reconstructed channels
evoked_recon.plot()

# Reorder the channel naming
evoked_recon.reorder_channels(ch_names)

# Since our reconstructed channels are still flat channels, we first add them as bad
# channels and then perform interpolation.
evoked_recon.info["bads"] = chs_drop
evoked_interp = evoked_recon.interpolate_bads(reset_bads=True)

# %%
# Plot original vs interpolated channels
fig = plt.figure(figsize=(10, 8), constrained_layout=True)
conditions = ("Original", "Interpolated")
evokeds = dict(zip(conditions, [evoked, evoked_interp]))
gs = grd.GridSpec(ncols=1, nrows=2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax = [ax1, ax2]

for itype, i in enumerate(ax):
    x_fills = [75, 125]  # 75-125 ms (highlight auditory activity)
    mne.viz.plot_compare_evokeds(
        evokeds,
        axes=i,
        picks=chs_drop[itype],
        legend=True,
        linestyles=dict(Original="solid", Interpolated="dashed"),
        show=False,
        time_unit="ms",
    )
    i.axvspan(x_fills[0], x_fills[1], alpha=0.5, color="grey")

plt.margins(y=0.2)
plt.show()
