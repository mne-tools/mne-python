"""
.. _ex-plot-meg-sensors:

======================================
Plotting sensor layouts of MEG systems
======================================

Show sensor layouts of different MEG systems.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne.io import read_raw_fif, read_raw_ctf, read_raw_bti, read_raw_kit
from mne.io import read_raw_artemis123
from mne.datasets import sample, spm_face, testing
from mne.viz import plot_alignment, set_3d_title

print(__doc__)

bti_path = op.abspath(op.dirname(mne.__file__)) + '/io/bti/tests/data/'
kit_path = op.abspath(op.dirname(mne.__file__)) + '/io/kit/tests/data/'
raws = {
    'Neuromag': read_raw_fif(sample.data_path() +
                             '/MEG/sample/sample_audvis_raw.fif'),
    'CTF 275': read_raw_ctf(spm_face.data_path() +
                            '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'),
    'Magnes 3600wh': read_raw_bti(op.join(bti_path, 'test_pdf_linux'),
                                  op.join(bti_path, 'test_config_linux'),
                                  op.join(bti_path, 'test_hs_linux')),
    'KIT': read_raw_kit(op.join(kit_path, 'test.sqd')),
    'Artemis123': read_raw_artemis123(op.join(
        testing.data_path(), 'ARTEMIS123',
        'Artemis_Data_2017-04-14-10h-38m-59s_Phantom_1k_HPI_1s.bin')),
}

for system, raw in sorted(raws.items()):
    meg = ['helmet', 'sensors']
    # We don't have coil definitions for KIT refs, so exclude them
    if system != 'KIT':
        meg.append('ref')
    fig = plot_alignment(raw.info, eeg=False, meg=('helmet', 'sensors'),
                         coord_frame='meg', show_axes=True, verbose=True)
    set_3d_title(figure=fig, title=system)
