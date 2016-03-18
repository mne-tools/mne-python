"""
=========================================
Plotting head in helmet from a trans file
=========================================

In this example, the head is shown in the
MEG helmet along with the EEG electrodes in MRI
coordinate system. This allows assessing the
MEG <-> MRI coregistration quality.

"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from mne import read_evokeds
from mne.datasets import sample
from mne.viz import plot_trans

print(__doc__)


data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

condition = 'Left Auditory'
evoked = read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))
plot_trans(evoked.info, trans_fname, subject='sample', dig=True,
           meg_sensors=True, subjects_dir=subjects_dir)
