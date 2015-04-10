"""
=======================
Remap MEG channel types
=======================

In this example, M/EEG data are remapped from one
channel type to another. This process can be
computationally intensive.
"""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>

# License: BSD (3-clause)

from mne.datasets import sample
from mne import read_evokeds
from mne.forward import compute_virtual_evoked

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'

from_type, to_type = 'mag', 'grad'
condition = 'Left Auditory'
evoked = read_evokeds(evoked_fname, condition=condition, baseline=(-0.2, 0.0))

virtual_evoked = compute_virtual_evoked(evoked, subject='sample',
                                        subjects_dir=subjects_dir,
                                        from_type=from_type, to_type=to_type)

evoked.plot_topomap(ch_type=from_type)
virtual_evoked.plot_topomap(ch_type=to_type)
