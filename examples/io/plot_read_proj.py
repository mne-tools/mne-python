"""
==============================================
Read and visualize SSP (and other) projections
==============================================

This example shows how to read and visualize SSP vector correcting for ECG.
"""
# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)


from mne import read_proj 

from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()

subjects_dir = data_path + '/subjects'
fname = data_path + '/MEG/sample/sample_audvis.fif'
ecg_fname = data_path + '/MEG/sample/sample_audvis_ecg-proj.fif'

ssp_projs = read_proj(ecg_fname)

###############################################################################
# Show a single projection
ssp_projs[0].plot_topomap()


###############################################################################
# Show all projections within a raw file



