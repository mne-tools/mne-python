"""
=========================
Use source space morphing
=========================

This example shows how to use source space morphing (as opposed to
SourceEstimate morphing) to create data that can be compared between
subjects.

.. warning:: Source space morphing will likely lead to source spaces that are
             less evenly sampled than source spaces created for individual
             subjects. Use with caution and check effects on localization
             before use.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Eric larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem',
                    'sample-5120-bem-sol.fif')
fname_src_fs = op.join(subjects_dir, 'fsaverage', 'bem',
                       'fsaverage-ico-5-src.fif')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')

# Get relevant channel information
info = mne.io.read_info(raw_fname)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False,
                                          exclude=[]))

# Morph fsaverage's source space to sample
src_fs = mne.read_source_spaces(fname_src_fs)
src_morph = mne.morph_source_spaces(src_fs, subject_to='sample',
                                    subjects_dir=subjects_dir)

# Compute the forward with our morphed source space
fwd = mne.make_forward_solution(info, trans=fname_trans,
                                src=src_morph, bem=fname_bem)
mag_map = mne.sensitivity_map(fwd, ch_type='mag')

# Return this SourceEstimate (on sample's surfaces) to fsaverage's surfaces
mag_map_fs = mag_map.return_to_original_src(src_fs, subjects_dir=subjects_dir)

# Plot the result, which tracks the sulcal-gyral folding
kwargs = dict(clim=dict(kind='percent', lims=[0, 50, 100]), smoothing_steps=5,
              hemi='both', views=['med', 'lat'])
brain = mag_map_fs.plot(time_label=None, subjects_dir=subjects_dir, **kwargs)
