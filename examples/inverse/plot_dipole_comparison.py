# -*- coding: utf-8 -*-
"""
================================================
Compare dipole fits between MNE-C and mne-python
================================================

This creates a small grid of source locations and performs dipole
fits using both MNE-C and mne-python.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from os import path as op
import matplotlib.pyplot as plt
import tempfile
import shutil

import mne

print(__doc__)

meg = True
eeg = True

data_path = mne.datasets.testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_evo = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem',
                    'sample-1280-1280-1280-bem-sol.fif')
fname_dip = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
fname_mri = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-trans.fif')

# Store temporary files somewhere
temp_dir = tempfile.mkdtemp()
fname_dip = op.join(temp_dir, 'test.dip')
fname_src = op.join(temp_dir, 'test-src.fif')
fname_sim = op.join(temp_dir, 'test-ave.fif')

#
# Simulate data on a 4 cm grid @ SNR=20 with random orientations
#
amp = 10e-9
rng = np.random.RandomState(0)
src = mne.setup_volume_source_space('sample', fname_src, pos=40,
                                    bem=fname_bem, overwrite=True)
nn = rng.randn(src[0]['np'], 3)
nn /= np.sqrt(np.sum(nn * nn, axis=1)[:, np.newaxis])
src[0]['nn'] = nn
evoked = mne.read_evokeds(fname_evo)[0]
fwd = mne.make_forward_solution(evoked.info, fname_mri, fname_src,
                                fname_bem, n_jobs=2)
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
cov = mne.read_cov(fname_cov)
stc = mne.VolSourceEstimate(amp * np.eye(fwd['src'][0]['nuse']),
                            fwd['src'][0]['vertno'], 0, 0.001)
evoked = mne.simulation.generate_evoked(fwd, stc, evoked, cov, snr=20,
                                        random_state=rng)
if eeg:
    evoked.add_proj(mne.proj.make_eeg_average_ref_proj(evoked.info))
mne.write_evokeds(fname_sim, evoked)
picks = mne.pick_types(evoked.info, meg=meg, eeg=eeg)
evoked.pick_channels([evoked.ch_names[k] for k in picks])

#
# Run MNE-C version
#
mne.utils.run_subprocess([
    'mne_dipole_fit', '--meas', fname_sim, '--meg', '--eeg',
    '--bem', fname_bem, '--noise', fname_cov, '--dip', fname_dip,
    '--mri', fname_mri, '--reg', '0'] +
    (['--meg'] if meg else []) + (['--eeg'] if eeg else []))
dip_c = mne.read_dipole(fname_dip)
shutil.rmtree(temp_dir)

#
# Run mne-python version
#
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_mri, n_jobs=2)

#
# Compare to original points
#
trans = mne.read_trans(fname_mri)
mne.transform_surface_to(src[0], 'head', trans)
src_rr = src[0]['rr'][src[0]['vertno']]
src_nn = fwd['source_nn']

# MNE-C skips the last "time" point :(
dip.crop(dip_c.times[0], dip_c.times[-1])
src_rr = src_rr[:-1]
src_nn = src_nn[:-1]

fig = plt.figure()
ax = fig.add_subplot(111)
orig = src_rr * 1000.
pts = ax.plot(orig[:, 0], orig[:, 1],
              color='k', markerfacecolor='none', marker='o', markersize=10,
              linestyle='none', label='True location')[0]
algs = ['MNE-C', 'mne-python']
gc_dist = 180 / np.pi * np.mean(np.arccos(np.sum(dip_c.ori * dip.ori,
                                                 axis=1)))
print('  Average orientation error: %s deg' % round(gc_dist, 1))
for ii, (d, color, alg) in enumerate(zip((dip_c, dip), ('b', 'r'), algs)):
    new = d.pos * 1000.
    diffs = new - orig
    print('\n%s:' % alg)
    corr = np.corrcoef(orig.ravel(), new.ravel())[0, 1]
    print('  Position corr.:  %s' % round(corr, 4))
    dists_sq = np.sum(diffs * diffs, axis=1)
    dist = np.sqrt(np.mean(dists_sq))
    print('  RMS distance:    %s mm' % round(dist, 1))
    misses = np.round(np.sort(np.sqrt(dists_sq))[::-1][:5], 1)
    print('  Worst misses:    %s' % misses)
    # great circle angle
    gc_dist = 180 / np.pi * np.mean(np.arccos(np.sum(src_nn * d.ori,
                                                     axis=1)))
    print('  Average orientation error: %s deg' % round(gc_dist, 1))
    amp_err = 1e9 * np.sqrt(np.mean((amp - d.amplitude) ** 2))
    print('  RMS amplitude error:       %s nA' % round(amp_err, 1))
    avg_gof = np.mean(d.gof)
    print('  Average goodness of fit:   %s%%' % round(avg_gof, 1))
    q = ax.quiver(orig[:, 0], orig[:, 1], *diffs.T[:2],
                  headwidth=4, headlength=4,
                  facecolor=color, scale=1., units='xy', alpha=0.5)
    ax.quiverkey(q, 0.25 + 0.5 * ii, 0.95, 10, alg, labelpos='S',
                 color=color)
ax.set_ylim([-80, 90])
ax.set_xlim([-50, 40])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
plt.legend([pts], ('Original locations',), loc=8)
plt.show()
