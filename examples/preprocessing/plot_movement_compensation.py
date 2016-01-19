"""
==============================================
Maxwell filter data with movement compensation
==============================================

Demonstrate movement compensation on simulated data.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.transforms import rot_to_quat
from mne.simulation import simulate_raw
from mne.preprocessing import maxwell_filter

print(__doc__)

data_path = mne.datasets.sample.data_path()

###############################################################################
# Simulate some head movement (typically recorded data could be used instead)

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
subjects_dir = data_path + '/subjects'
bem_fname = subjects_dir + '/sample/bem/sample-5120-bem-sol.fif'
src_fname = subjects_dir + '/sample/bem/sample-oct-6-src.fif'

# let's make rotation matrices about each axis, plus some compound rotations
phi = np.deg2rad(30)
x_rot = np.array([[1, 0, 0],
                  [0, np.cos(phi), -np.sin(phi)],
                  [0, np.sin(phi), np.cos(phi)]])
y_rot = np.array([[np.cos(phi), 0, np.sin(phi)],
                  [0, 1, 0],
                  [-np.sin(phi), 0, np.cos(phi)]])
z_rot = np.array([[np.cos(phi), -np.sin(phi), 0],
                  [np.sin(phi), np.cos(phi), 0],
                  [0, 0, 1]])
xz_rot = np.dot(x_rot, z_rot)
xmz_rot = np.dot(x_rot, z_rot.T)
yz_rot = np.dot(y_rot, z_rot)
mymz_rot = np.dot(y_rot.T, z_rot.T)


# Create different head rotations, one per second
rots = [x_rot, y_rot, z_rot, xz_rot, xmz_rot, yz_rot, mymz_rot]
# The transpose of a rotation matrix is a rotation in the opposite direction
rots += [rot.T for rot in rots]

raw = mne.io.Raw(raw_fname).crop(0, len(rots))
raw.load_data().pick_types(meg=True, stim=True, copy=False)
raw.add_proj([], remove_existing=True)
center = (0., 0., 0.04)  # a bit lower than device origin
raw.info['dev_head_t']['trans'] = np.eye(4)
raw.info['dev_head_t']['trans'][:3, 3] = center
pos = np.zeros((len(rots), 10))
for ii in range(len(pos)):
    pos[ii] = np.concatenate([[ii], rot_to_quat(rots[ii]), center, [0] * 3])
pos[:, 0] += raw.first_samp / raw.info['sfreq']  # initial offset

# Let's activate a vertices bilateral auditory cortices
src = mne.read_source_spaces(src_fname)
labels = mne.read_labels_from_annot('sample', 'aparc.a2009s', 'both',
                                    regexp='G_temp_sup-Plan_tempo',
                                    subjects_dir=subjects_dir)
assert len(labels) == 2  # one left, one right
vertices = [np.intersect1d(l.vertices, s['vertno'])
            for l, s in zip(labels, src)]
data = np.zeros([sum(len(v) for v in vertices), int(raw.info['sfreq'])])
activation = np.hanning(int(raw.info['sfreq'] * 0.2)) * 1e-9  # nAm
t_offset = int(np.ceil(0.2 * raw.info['sfreq']))  # 200 ms in (after baseline)
data[:, t_offset:t_offset + len(activation)] = activation
stc = mne.SourceEstimate(data, vertices, tmin=-0.2,
                         tstep=1. / raw.info['sfreq'])

# Simulate the movement
raw = simulate_raw(raw, stc, trans_fname, src, bem_fname,
                   head_pos=pos, interp='zero', n_jobs=-1)
raw_stat = simulate_raw(raw, stc, trans_fname, src, bem_fname,
                        head_pos=None, n_jobs=-1)

##############################################################################
# Process our simulated raw data taking into account head movements

# extract our resulting events
events = mne.find_events(raw, stim_channel='STI 014')
assert len(events) == len(pos)  # make sure we did this right
assert np.array_equal(events[:, 2], np.arange(len(pos)) + 1)
events[:, 2] = 1
raw.plot(events=events)

topo_kwargs = dict(times=[0, 0.1, 0.2], ch_type='mag', vmin=-500, vmax=500)

# 0. Take average of stationary data (bilateral auditory patterns)
evoked_stat = mne.Epochs(raw_stat, events, 1, -0.2, 0.8).average()
evoked_stat.plot_topomap(title='stationary', **topo_kwargs)

# 1. Take a naive average (smears activity)
evoked = mne.Epochs(raw, events, 1, -0.2, 0.8).average()
assert evoked.nave == len(pos)
evoked.plot_topomap(title='naive average', **topo_kwargs)

# 2. Use raw movement compensation (works well)
raw_sss = maxwell_filter(raw, pos=pos, regularize=None, bad_condition='ignore',
                         verbose=True)
evoked_raw_mc = mne.Epochs(raw_sss, events, 1, -0.2, 0.8).average()
assert evoked_raw_mc.nave == len(pos)
evoked_raw_mc.plot_topomap(title='raw movement compensation', **topo_kwargs)
