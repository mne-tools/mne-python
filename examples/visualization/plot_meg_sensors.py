"""
======================================
Plotting sensor layouts of MEG systems
======================================

In this example, sensor layouts of different MEG systems
are shown.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

from mayavi import mlab

import mne
from mne.io import read_raw_fif, read_raw_ctf, read_raw_bti, read_raw_kit
from mne.datasets import sample, spm_face
from mne.viz import plot_trans

print(__doc__)

bti_path = op.abspath(op.dirname(mne.__file__)) + '/io/bti/tests/data/'
kit_path = op.abspath(op.dirname(mne.__file__)) + '/io/kit/tests/data/'
raws = dict(
    Neuromag=read_raw_fif(sample.data_path() +
                          '/MEG/sample/sample_audvis_raw.fif'),
    CTF_275=read_raw_ctf(spm_face.data_path() +
                         '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'),
    Magnes_3600wh=read_raw_bti(op.join(bti_path, 'test_pdf_linux'),
                               op.join(bti_path, 'test_config_linux'),
                               op.join(bti_path, 'test_hs_linux')),
    KIT=read_raw_kit(op.join(kit_path, 'test.sqd')),
)

for system, raw in raws.items():
    # We don't have coil definitions for KIT refs, so exclude them
    ref_meg = False if system == 'KIT' else True
    fig = plot_trans(raw.info, trans=None, dig=False, eeg_sensors=False,
                     meg_sensors=True, coord_frame='meg', ref_meg=ref_meg)
    mlab.title(system)
