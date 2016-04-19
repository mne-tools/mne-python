"""
================================================
Display sensitivity maps for EEG and MEG sensors
================================================

Sensitivity maps can be produced from forward operators that
indicate how well different sensor types will be able to detect
neural currents from different regions of the brain.

To get started with forward modeling see ref:`tut_forward`.

"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
import matplotlib.pyplot as plt

print(__doc__)

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'

subjects_dir = data_path + '/subjects'

# Read the forward solutions with surface orientation
fwd = mne.read_forward_solution(fwd_fname, surf_ori=True)
leadfield = fwd['sol']['data']
print("Leadfield size : %d x %d" % leadfield.shape)

###############################################################################
# Compute sensitivity maps

grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivity map

picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False)
picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=14)
for ax, picks, ch_type in zip(axes, [picks_meg, picks_eeg], ['meg', 'eeg']):
    im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto',
                   cmap='RdBu_r')
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

grad_map.plot(time_label='Gradiometer sensitivity', subjects_dir=subjects_dir,
              clim=dict(lims=[0, 50, 100]))
