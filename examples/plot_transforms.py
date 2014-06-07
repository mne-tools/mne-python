from mne.io import read_evokeds
from mne.datasets import sample
from mne.viz import plot_trans


data_path = sample.data_path()

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
# If trans_fname is set to None then only MEG estimates can be visualized

condition = 'Left Auditory'
evoked = read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))
plot_trans(evoked.info, trans_fname=trans_fname, subject='sample',
           subjects_dir=subjects_dir)
