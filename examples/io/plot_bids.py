"""
===================================
Create a BIDS-compatible MEG folder
===================================

Brain Imaging Data Structure (BIDS) MEG is a new standard for
storing MEG files. This example demonstrates how to convert
your existing files into a BIDS-compatible folder.
"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

# License: BSD (3-clause)

import os.path as op
from mne.datasets import sample
from mne.io import folder_to_bids

data_path = sample.data_path()

fnames = dict(events='sample_audvis_raw-eve.fif',
              raw='sample_audvis_raw.fif')

input_path = op.join(data_path, 'MEG', 'sample')
output_path = op.join(data_path, '..', 'MNE-sample-data-bids')
folder_to_bids(input_path=input_path, output_path=output_path,
               fnames=fnames, subject='01', run='01', task='audiovisual')
