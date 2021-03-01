"""
=============================================
Interpolate bad channels for MEG/EEG channels
=============================================

This example shows how to interpolate bad MEG/EEG channels

- Using spherical splines from :footcite:`PerrinEtAl1989` for EEG data.
- Using field interpolation for MEG and EEG data.

In this example, the bad channels will still be marked as bad.
Only the data in those channels is replaced.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_number = 2

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory',
                          baseline=(None, 0))

# plot with bads
evoked.plot(exclude=[], picks=('grad', 'eeg'))

###############################################################################
# Compute interpolation (also works with Raw and Epochs objects)
evoked_interp = evoked.copy().interpolate_bads(reset_bads=False)
evoked_interp.plot(exclude=[], picks=('grad', 'eeg'))

###############################################################################
# You can also use minimum-norm for EEG as well as MEG
evoked_interp_mne = evoked.copy().interpolate_bads(
    reset_bads=False, method=dict(eeg='MNE'), verbose=True)
evoked_interp_mne.plot(exclude=[], picks=('grad', 'eeg'))

###############################################################################
# References
# ----------
# .. footbibliography::
