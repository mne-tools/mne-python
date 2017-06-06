"""
==================================================
n-way non parametric cluster tests at sensor level
==================================================

This will compute an n-way non-parametric analysis
based on Threshold Free Cluster Statistics (TFCE).
"""
# Authors: Jean-Remi king <jeanremi.king@gmail.com>
#
# License: Simplified BSD

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io
from mne.channels import read_ch_connectivity
from mne.stats import spatio_temporal_cluster_1samp_test
# XXX as it's WIP, I'm putting in a separate file
from mne.stats.nway_anova import set_contrasts, compute_contrast

from mne.datasets import sample

# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
subjects_dir = data_path + '/subjects'

# Preprocessing
raw = io.Raw(raw_fname)
raw.info['bads'] += ['MEG 2443']
picks = mne.pick_types(raw.info, meg=True, eog=True, exclude='bads')
events = mne.read_events(event_fname)
event_id = {'audio/left': 1, 'audio/right': 2,
            'visual/left': 3, 'visual/right': 4}
reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id, -0.2, 0.3, picks=picks,
                    baseline=(None, 0), reject=reject, preload=True)

# XXX necessary in sensor space?
epochs.equalize_event_counts(event_id, copy=False)

# Subsample for speed
epochs.resample(50)
epochs.crop(0, None)

# ############################################################################
# Stats on magnetometers only
epochs.pick_types(meg='mag')
connectivity, ch_names = read_ch_connectivity('neuromag306mag')

# Prepare ANOVA contrasts
contrasts, contrasts_labels = set_contrasts(
    [['left', 'right'], ['visual', 'audio']], level=2)

X_list = compute_contrast(epochs, contrasts)

# For each contrast
for (X, contrast_label) in zip(X_list, contrasts_labels):
    # Simulate 10 subjects
    n_subjects = 10
    X = np.tile(X, [n_subjects, 1, 1])
    X = X.transpose(0, 2, 1)  # subject x time x space
    X += np.random.randn(*X.shape) * np.std(epochs._data)  # add noise

    # Run TFCE
    T_obs, clusters, p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(
            X, connectivity=connectivity, n_jobs=-1,
            threshold=dict(start=1, step=.2), out_type='mask',
            n_permutations=128, buffer_size=None)

    # Create topomap mask from sig cluster
    sig_clusters = np.array(clusters)[p_values < .01, :, :]
    mask = np.sum(sig_clusters, axis=0).transpose()

    # plot average contrast
    evoked = epochs.average()
    evoked.data = np.mean(X, axis=0).T
    evoked.plot_topomap(mask=mask, sensors=False, vmin=-600, vmax=600,
                        contours=False, show=False, title=contrast_label)
plt.show()
