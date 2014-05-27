import nibabel as nib
import numpy as np
import mne
from mne.datasets import spm_face
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import matplotlib.pyplot as plt
from scipy import stats

# get the data paths
data_path = spm_face.data_path()
subjects_dir = data_path + '/subjects'

# get the segmentation, bem, and transformation files
aseg_fname = subjects_dir + '/spm/mri/aseg.mgz'
mri = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
bem = subjects_dir + '/spm/bem/spm-5120-5120-5120-bem-sol.fif'

# Read the epoch data
epo_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D_epochs.fif'
epochs = mne.read_epochs(epo_fname)

# load segment info using nibabel
aseg = nib.load(aseg_fname)
aseg_data = aseg.get_data()
ix = aseg_data == 54  # index for the right amygdala

# get the indices in x, y, z space
iix = []
for i in range(ix.shape[0]):
    for j in range(ix.shape[1]):
        for k in range(ix.shape[2]):
            if ix[i, j, k]:
                iix.append([i, j, k])

iix = np.array(iix)  # convert to array

# get the header information
aseg_hdr = aseg.get_header()

# get the transformation matrix
trans = aseg_hdr.get_vox2ras_tkr()

# convert using the transformation matrix
xyz = np.dot(iix, trans[:3, :3].T)+trans[:3, 3]

# convert to meters
xyz /= 1000.

# generate random orientations
ori = np.random.randn(xyz.shape[0], xyz.shape[1])

# create the pos dictionary
pos = dict(rr=xyz, nn=ori)

# estimate noise covarariance
noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))

# setup the source space
src = mne.setup_volume_source_space('spm', pos=pos)

# setup the forward model
forward = mne.make_forward_solution(epochs.info, mri=mri, src=src, bem=bem)
forward = mne.convert_forward_solution(forward, surf_ori=True)

# Compute inverse solution
snr = 5.0
lambda2 = 1.0 / snr ** 2
method = 'dSPM'

inverse_operator = make_inverse_operator(epochs.info, forward, noise_cov,
                                         loose=0.2, depth=0.8)

# Apply inverse solution to both stim types
stc_faces = apply_inverse_epochs(epochs['faces'], inverse_operator, lambda2,
                                 method)
stc_scrambled = apply_inverse_epochs(epochs['scrambled'], inverse_operator,
                                     lambda2, method)

# compare face vs. scrambled trials
X = np.zeros((len(epochs.events), len(epochs.times)))

for i, s in enumerate(stc_faces):
    x = s.data.mean(0)
    X[i] = x
for i, s in enumerate(stc_scrambled):
    x = s.data.mean(0)
    X[i+83] = x

t, p = stats.ttest_ind(X[:83], X[83:])
sig, p = mne.stats.fdr_correction(p)  # apply fdr correction

# plot the results
t = epochs.times
s1 = X[:83]
s2 = X[83:]

ax = plt.axes()

l1, = ax.plot(t, s1.mean(0), 'b')  # faces
l2, = ax.plot(t, s2.mean(0), 'g')  # scrambled

ylim = ax.get_ylim()
ax.fill_between(t, ylim[1]*np.ones(t.shape), ylim[0]*np.ones(t.shape), sig,
                facecolor='k', alpha=0.3)
ax.set_xlim((t.min(), t.max()))
ax.set_xlabel('Time (s)')
ax.set_title('Right Amygdala Activation')
ax.legend((l1, l2), ('Faces', 'Scrambled'))
ax.set_ylim(ylim)

plt.show()
