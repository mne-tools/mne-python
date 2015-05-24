import os.path as op
import numpy as np

import mne
from mne.forward import make_field_map, transform_instances


path = mne.datasets.sample.data_path()
evoked_fname = op.join(path, 'MEG', 'sample', 'sample_audvis-ave.fif')
trans_fname = op.join(path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
subjects_dir = op.join(path, 'subjects')

condition = 'Left Auditory'
evoked = mne.read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))

# Compute the field maps to project MEG and EEG data to MEG helmet
# and scalp surface
maps = make_field_map(evoked, trans_fname, subject='sample',
                      subjects_dir=subjects_dir,  meg_surf='head')
p_field = evoked.plot_field(maps, time=0.11)


p_helmet = mne.viz.plot_trans(evoked.info, trans_fname, 'sample', subjects_dir,
                              ch_type='meg', coord_frame='meg')


# let's assume you have multiple runs and your subject moved :(
evoked_rot = evoked.copy()
trans_rot = evoked_rot.info['dev_head_t']['trans']

rot = mne.transforms.rotation(x=0, y=np.pi/16, z=0)
trans_rot[:] = rot.dot(trans_rot)

maps_rot = make_field_map(evoked_rot, trans_fname, subject='sample',
                          subjects_dir=subjects_dir, meg_surf='head')
p_field_rot = evoked_rot.plot_field(maps_rot, time=0.11)

p_helmet_rot = mne.viz.plot_trans(evoked_rot.info, trans_fname, 'sample',
                                  subjects_dir, ch_type='meg',
                                  coord_frame='meg')

# we can see that the dipoles are now more pronounced on the rotated head
# bc the magnetic field has changed given the head position
# let's try to rotate the head
# we start with the same data in both evokeds
# the infos are different because of the rotation to the dev_head_t
# this will result in our simulated rotated data.
# now same dev_head_t, different data

# let's see if we can return to our original data
# we apply a rotation in the opposite direction to return to original evoked

trans_rot = evoked.copy().info['dev_head_t']['trans']
rot = mne.transforms.rotation(x=0, y=-np.pi/16, z=0)
trans_rot = rot.dot(trans_rot)
evoked_rot.info['dev_head_t']['trans'] = trans_rot

transform_instances([evoked, evoked_rot])
maps_mod = make_field_map(evoked_rot, trans_fname, subject='sample',
                          subjects_dir=subjects_dir, meg_surf='head')
p_field_mod = evoked_rot.plot_field(maps_rot, time=0.11)
picks = mne.pick_types(evoked.info, meg=True)

# # values are different
# np.testing.assert_allclose(evoked.data[picks], evoked_rot.data[picks])
