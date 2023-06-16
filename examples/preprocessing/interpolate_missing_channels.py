"""
==================================================================================
Problem statement:
sometimes a few MEG channels can be completely turned off during an experimental
recording on purpose (i.e., bad sensors contain high noise).
This is done inorder to stop noise spreading towards the other sensors.
Therefore, physical location of the sensor is completely missing from
the recording data. Therefore, the sensor can't be interpolated anymore.
==================================================================================

@author: diptyajit das <bmedasdiptyajit@gmail.com>
Created on Jun 16, 2023

"""

# import some packages
import mne
from mne.datasets import sample
from mne import read_evokeds
import matplotlib.pyplot as plt
from matplotlib import gridspec as grd

print(__doc__)

# define meg data path
data_path = sample.data_path()
evk_fname = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"

# read sample evoked data
evk = read_evokeds(evk_fname)[0]

# pick only grad channels and crop the data
evk_grad = evk.copy().pick_types(meg='grad').crop(-0.05, 0.4)

# correct baseline
evk_grad.apply_baseline(baseline=(None, 0))

# get the original channel names
grad_ch_names = evk_grad.info['ch_names']

# we specify one sensor (left auditory),
# later we want to drop this sensor to create a missing sensor data file
mark_channel = 'MEG 0212'
pos = grad_ch_names.index(mark_channel)  # get sensor position

# print the location // 'MEG 0212'
print(evk_grad.info['chs'][pos]['loc'])

# now lets drop the mark channel (in this case we choose a grad channel:'MEG 0212')
# to create a missing channel scenario. this can be any bad sensor in practice
mis_channel = 'MEG 0212'
mis_channel_evk = evk_grad.copy().drop_channels(grad_ch_names[pos])

# using MNE function to add an extra reference channel
# it's a dummy reference channel ('flat' channel) inorder to retrieve our missing channel
evk_recon = mne.add_reference_channels(mis_channel_evk, ref_channels=[mis_channel])

# now change the channel type to grad
evk_recon.set_channel_types({mis_channel: 'grad'})

# check the info file that contains channel order
# newly added channel name can be seen at the end of the channel order
print('check the order of the channel names: ', evk_recon.info['ch_names'])

# let's check our channel description that has the location of the sensor
print('newly added channel location', evk_recon.info['chs'][-1]['loc'])

# we replace the channel description with the original one
evk_recon.info['chs'][-1] = evk_grad.info['chs'][pos]  # original the grad channel description

# let's reorder channels now as it was for original case
evk_recon.reorder_channels(grad_ch_names)

# since our newly added channel is still a flat channel,
# we add the channel as a bad channel manually to perform interpolation
evk_recon.info['bads'] = [mis_channel]
evk_recon.interpolate_bads()

# compare original vs reconstructed/interpolated evoked
# create figure
fig = plt.figure(figsize=(10, 8), constrained_layout=False)
gs = grd.GridSpec(ncols=1, nrows=1, figure=fig)

# set axis
ax = fig.add_subplot(gs[0, 0])
conds = ('Original channel', 'interpolated channel')
evks = dict(zip(conds, [evk_grad, evk_recon]))
x_fills = [50, 150]  # 50-150 ms (highlight auditory activity for our sample data)
mne.viz.plot_compare_evokeds(evks, axes=ax, picks=mis_channel, legend=True, show=False, time_unit='ms')
ax.axvspan(x_fills[0], x_fills[1], alpha=0.5, color='grey')
plt.tight_layout()
plt.margins(y=0.2)
plt.show()
