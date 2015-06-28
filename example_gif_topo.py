import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
from mne.viz.topomap import _prepare_topo_plot, _plot_update_evoked_topomap
from mne import report

path = mne.datasets.sample.data_path()
fname = path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory', baseline=(None, 0))

fig, ax = plt.subplots(1, figsize=[2, 3])
picks, pos, merge_grads, _, ch_type = _prepare_topo_plot(evoked, 'mag', None)


def animate(nframe):
    for time in np.linspace(0., .200, 5):
        time_idx = np.where(evoked.times > time)[0][0]
        data = evoked.data[picks, time_idx]
        plot_topomap(data, pos, axis=ax, vmin=-1e-13, vmax=1e-13,
                     sensors=False, contours=False, show=False)

anim = animation.FuncAnimation(fig, animate, frames=5)
fig.show()

report = report.Report()
report.add_anims_to_section(anim, 'section', 'example')
report.save()
