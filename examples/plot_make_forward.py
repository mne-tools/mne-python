"""
======================================================
Create a forward operator and display sensitivity maps
======================================================
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

print(__doc__)

import mne
from mne.datasets import sample
data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
mri = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
src = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
bem = data_path + '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'
subjects_dir = data_path + '/subjects'

fwd = mne.make_forward_solution(raw_fname, mri=mri, src=src, bem=bem,
                                fname=None, meg=True, eeg=True, mindist=5.0,
                                n_jobs=2, overwrite=True)

# convert to surface orientation for better visualization
fwd = mne.convert_forward_solution(fwd, surf_ori=True)
leadfield = fwd['sol']['data']

print("Leadfield size : %d x %d" % leadfield.shape)

grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivity map

import matplotlib.pyplot as plt
picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False)
picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=14)
for ax, picks, ch_type in zip(axes, [picks_meg, picks_eeg], ['meg', 'eeg']):
    im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto')
    ax.set_title(ch_type.upper())
    ax.set_xlabel('sources')
    ax.set_ylabel('sensors')
    plt.colorbar(im, ax=ax, cmap='RdBu_r')
plt.show()

plt.figure()
plt.hist([grad_map.data.ravel(), mag_map.data.ravel(), eeg_map.data.ravel()],
         bins=20, label=['Gradiometers', 'Magnetometers', 'EEG'],
         color=['c', 'b', 'k'])
plt.legend()
plt.title('Normal orientation sensitivity')
plt.xlabel('sensitivity')
plt.ylabel('count')
plt.show()

args = dict(fmin=0.1, fmid=0.5, fmax=0.9, smoothing_steps=7)
grad_map.plot(subject='sample', time_label='Gradiometer sensitivity',
              subjects_dir=subjects_dir, **args)
