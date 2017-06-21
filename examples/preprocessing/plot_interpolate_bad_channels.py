"""
=============================================
Interpolate bad channels for MEG/EEG channels
=============================================

This example shows how to interpolate bad MEG/EEG channels

    - Using spherical splines as described in [1]_ for EEG data.
    - Using field interpolation for MEG data.

The bad channels will still be marked as bad. Only the data in those channels
is removed.

References
----------
.. [1] Perrin, F., Pernier, J., Bertrand, O. and Echallier, JF. (1989)
       Spherical splines for scalp potential and current density mapping.
       Electroencephalography and Clinical Neurophysiology, Feb; 72(2):184-7.
"""
# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evoked = mne.read_evokeds(fname, condition='Left Auditory',
                          baseline=(None, 0))

# plot with bads
evoked.plot(exclude=[])

# compute interpolation (also works with Raw and Epochs objects)
evoked.interpolate_bads(reset_bads=False)

# plot interpolated (previous bads)
evoked.plot(exclude=[])
