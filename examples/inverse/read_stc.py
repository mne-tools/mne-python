# -*- coding: utf-8 -*-
"""
.. _ex-read-stc:

===================
Reading an STC file
===================

STC files contain activations on cortex ie. source
reconstructions
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
fname = meg_path / 'sample_audvis-meg'

stc = mne.read_source_estimate(fname)

n_vertices, n_samples = stc.data.shape
print("stc data size: %s (nb of vertices) x %s (nb of samples)"
      % (n_vertices, n_samples))

# View source activations
plt.plot(stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('Source amplitude')
plt.show()
