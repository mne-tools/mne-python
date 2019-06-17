# -*- coding: utf-8 -*-
"""
.. _plot_digitization:

digitization
============

this is an example to keep discussing :class:`mne.digitization.digitization`
stuff.
"""  # noqa: d205, d400
# authors: alexandre gramfort <alexandre.gramfort@inria.fr>
#          teon brooks <teon.brooks@gmail.com>
#          eric larson <larson.eric.d@gmail.com>
#          joan massich <mailsik@gmail.com>
#
# license: bsd style.

from mayavi import mlab
import os.path as op
import numpy as np

import mne
from mne.datasets import fetch_fsaverage
from mne.datasets import testing
from mne.viz import plot_alignment


subjects_dir = op.dirname(fetch_fsaverage())

testing_data_path = testing.data_path(download=False)
fif_dig_montage_fname = op.join(testing_data_path, 'montage', 'eeganes07.fif')
egi_dig_montage_fname = op.join(testing_data_path,
                                'montage', 'coordinates.xml')
locs_montage_fname = op.join(testing_data_path, 'eeglab', 'test_chans.locs')

kit_data_dir = op.join(op.dirname(mne.__file__), 'io', 'kit', 'tests', 'data')
kit_hsp_fname = op.join(kit_data_dir, 'test_hsp.txt')
kit_elp_fname = op.join(kit_data_dir, 'test_elp.txt')


###############################################################################
# i should be able to visualize all those with digitization without using
#

valid_inputs = {
    # digitizations
    # 'kit_montage': {'hsp': kit_hsp_fname, 'elp': kit_elp_fname},
    # 'fif_montage': fif_dig_montage_fname,
    # 'egi_montage': egi_dig_montage_fname,
    # 'eeglab_montage': locs_montage_fname,

    # build-in montages
    'EGI_256': 'EGI_256',
    'GSN-HydroCel-128': 'GSN-HydroCel-128',
    'GSN-HydroCel-129': 'GSN-HydroCel-129',
    'GSN-HydroCel-256': 'GSN-HydroCel-256',
    'GSN-HydroCel-257': 'GSN-HydroCel-257',
    'GSN-HydroCel-32': 'GSN-HydroCel-32',
    'GSN-HydroCel-64_1.0': 'GSN-HydroCel-64_1.0',
    'GSN-HydroCel-65_1.0': 'GSN-HydroCel-65_1.0',
    'biosemi128': 'biosemi128',
    'biosemi16': 'biosemi16',
    'biosemi160': 'biosemi160',
    'biosemi256': 'biosemi256',
    'biosemi32': 'biosemi32',
    'biosemi64': 'biosemi64',
    'easycap-M1': 'easycap-M1',
    'easycap-M10': 'easycap-M10',
    'mgh60': 'mgh60',
    'mgh70': 'mgh70',
    'standard_1005': 'standard_1005',
    'standard_1020': 'standard_1020',
    'standard_alphabetic': 'standard_alphabetic',
    'standard_postfixed': 'standard_postfixed',
    'standard_prefixed': 'standard_prefixed',
    'standard_primed': 'standard_primed',
}

###############################################################################
# This is what should be in mne code
#


def _from_file_to_digitization(name, value):
    """this is just for illustrative purposes."""
    def _montage_to_digitization(my_montage):
        return my_montage

    def _digmontage_to_digitization(my_montage):
        return my_montage

    if name in ['EGI_256', 'GSN-HydroCel-128', 'GSN-HydroCel-129',
                'GSN-HydroCel-256', 'GSN-HydroCel-257', 'GSN-HydroCel-32',
                'GSN-HydroCel-64_1.0', 'GSN-HydroCel-65_1.0', 'biosemi128',
                'biosemi16', 'biosemi160', 'biosemi256', 'biosemi32',
                'biosemi64', 'easycap-M1', 'easycap-M10', 'mgh60', 'mgh70',
                'standard_1005', 'standard_1020', 'standard_alphabetic',
                'standard_postfixed', 'standard_prefixed', 'standard_primed']:

        montage = mne.channels.read_montage(value, unit='auto',
                                            transform=False)
        n_channels = len(montage.ch_names) if montage.ch_names else 0
        return _montage_to_digitization(montage), n_channels

    elif name == 'fif_montage':
        montage = mne.channels.montage.read_dig_montage(fif=value)
        return None, 0


###############################################################################
# User code:
# ----------
#

for kk, vv in valid_inputs.items():
    dig, n_channels = _from_file_to_digitization(name=kk, value=vv)

    info = mne.create_info(ch_names=n_channels, sfreq=1, ch_types='eeg')
    raw = mne.io.RawArray(data=np.empty((info['nchan'], 0)),
                          info=info)

    raw.set_montage(dig)

    fig = plot_alignment(info, trans=None,
                         subject='fsaverage',
                         subjects_dir=subjects_dir,
                         eeg=['projected'],
                         )
    mlab.view(135, 80)
    mlab.title(kk, figure=fig)
