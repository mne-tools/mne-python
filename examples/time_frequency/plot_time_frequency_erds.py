"""
===============================
Compute and visualize ERDS maps
===============================

This example calculates and displays ERDS maps of event-related EEG data. ERDS
(sometimes also written as ERD/ERS) is short for event-related
desynchronization (ERD) and event-related synchronization (ERS) [1]_.
Conceptually, ERD corresponds to a decrease in power in a specific frequency
band relative to a baseline. Similarly, ERS corresponds to an increase in
power. An ERDS map is a time/frequency representation of ERD/ERS over a range
of frequencies [2]_. ERDS maps are also known as ERSP (event-related spectral
perturbation) [3]_.

We use a public EEG BCI data set containing two different motor imagery tasks
available at PhysioNet. The two tasks are imagined hand and feet movement. Our
goal is to generate ERDS maps for each of the two tasks.

First, we load the data and create epochs of 5s length. The data sets contain
multiple channels, but we will only consider the three channels C3, Cz, and C4.
We compute maps containing frequencies ranging from 2 to 35Hz. We map ERD to
red color and ERS to blue color, which is the convention in many ERDS
publications. Finally, we perform cluster-based permutation tests to estimate
significant ERDS values (corrected for multiple comparisons within channels).

References
----------

.. [1] G. Pfurtscheller, F. H. Lopes da Silva. Event-related EEG/MEG
       synchronization and desynchronization: basic principles. Clinical
       Neurophysiology 110(11), 1842-1857, 1999.
.. [2] B. Graimann, J. E. Huggins, S. P. Levine, G. Pfurtscheller.
       Visualization of significant ERD/ERS patterns in multichannel EEG and
       ECoG data. Clinical Neurophysiology 113(1), 43-47, 2002.
.. [3] S. Makeig. Auditory event-related dynamics of the EEG spectrum and
       effects of exposure to tones. Electroencephalography and Clinical
       Neurophysiology 86(4), 283-293, 1993.
"""
# Authors: Clemens Brunner <clemens.brunner@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap


# load and preprocess data ####################################################
subject = 1  # use data from subject 1
runs = [6, 10, 14]  # use only hand and feet motor imagery runs

fnames = eegbci.load_data(subject, runs)
raws = [read_raw_edf(f, preload=True) for f in fnames]
raw = concatenate_raws(raws)

raw.rename_channels(lambda x: x.strip('.'))  # remove dots from channel names

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])

# epoch data ##################################################################
tmin, tmax = -1, 4  # define epochs around events (in s)
event_ids = dict(hands=2, feet=3)  # map event IDs to tasks

epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)

# compute ERDS maps ###########################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
