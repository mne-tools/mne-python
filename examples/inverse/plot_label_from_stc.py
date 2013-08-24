"""
====================================
Generate label from source estimates
====================================

Threshold a source estimate and produce a label after smoothing.

"""

# Author: Luke Bloy <luke.bloy@gmail.com>
#         Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD (3-clause)

import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse
from mne.fiff import Evoked
import pylab as pl
import numpy as np
from mne.stats.cluster_level import _find_clusters


def generateFunctionalLabel(tot_stc, aparc_label_name, src, subject, subjects_dir=None,
                            pct_thresh=0.2, smoothing=1):
    # ok we want to find a cluster centered on the max Power in aparc_label_name
    labels, label_colors = mne.labels_from_parc(
            subject, parc='aparc', subjects_dir=subjects_dir, regexp=aparc_label_name)
    tmp_label = labels[0]
    tmp_label_stc = tot_stc.in_label(tmp_label)
    idx = label_src_vertno_sel(tmp_label, src)

    # ok we want the max vertex so we can look find it in the tot_data...
    totMax_tmp = tmp_label_stc.data.max()
    totMax_ind = tmp_label_stc.data.argmax()

    if totMax_ind < len(tmp_label_stc.vertno[0]):
        totMax_vert = tmp_label_stc.vertno[0][totMax_ind]
    else:
        totMax_vert = tmp_label_stc.vertno[1][totMax_ind - len(
            tmp_label_stc.vertno[0])]

    # find the index in tot_stc.data the corresponds to totMax_vert...
    if totMax_vert in tot_stc.vertno[0]:
        totMaxIndex = np.nonzero(totMax_vert == tot_stc.vertno[0])[0][0]
    elif totMax_vert in tot_stc.vertno[1]:
        totMaxIndex = len(tot_stc.vertno[0]) + np.nonzero(
            totMax_vert == tot_stc.vertno[1])[0][0]
    else:
        print "Error finding vertexs"
        return

    # lets make sure the we found what we though we did...
    if (tot_stc.data[totMaxIndex, 0] != totMax_tmp):
        print "Big Problems finding correct index"
        return

    src_conn = mne.spatial_src_connectivity(src)

    # what should we use as the threshold for seperating clusters
    thresh = pct_thresh * totMax_tmp
    clusters, sums = _find_clusters(
        tot_stc.data[:, 0], thresh, connectivity=src_conn)

    # find the cluster with the totMaxVertex in it.
    bCluster = None
    for c in clusters:
        if totMaxIndex in c:
            bCluster = c
            break

    if bCluster == None:
        print "clustering didn't work!"
        return

    # make a label from bCluster
    tmpData = np.zeros(tot_stc.data.shape)
    tmpData[bCluster] = 1
    tmpStc = mne.SourceEstimate(
         tmpData, vertices=tot_stc.vertno, tmin=0, tstep=1000, subject=subject)
    func_label = mne.stc_to_label(
        tmpStc, src=src, smooth=smoothing, subjects_dir=subjects_dir)[0]
    return func_label


from mne.datasets import sample

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
subjects_dir = data_path + '/subjects'
subject = 'sample'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# The purpose of this example is to show how to compute labels based on seed
# growing activity.
# we'll compute an ROI based on the peak power between 80 and 120 ms.
# and we'll use the bankssts-lh as the anatomical seed ROI
aparc_label_name = 'bankssts-lh'
tmin, tmax = 0.080, 0.120

# Load data
evoked = Evoked(fname_evoked, setno=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']  # get the source space

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_normal=True)

stc.crop(tmin, tmax)

# Make a summary stc file with total power between tmin and tmax.
tot_data = np.sum(np.abs(stc.data), axis=1)[:, np.newaxis]
tot_stc = mne.SourceEstimate(tot_data, vertices=stc.vertno, tmin=0,
                             tstep=1000, subject=subject)

# use the tot_stc to generate a functional label
# region growing is halted at 60% of the peak value (of tot_stc) within the
# anatomical roi specified by aparc_label_name
func_label = generateFunctionalLabel(tot_stc, aparc_label_name, src, subject,
    pct_thresh=0.6, smoothing=5, subjects_dir=subjects_dir)
func_label.name = "%s_%s" % ('Active', aparc_label_name)

# load the anatomical ROI for comparison
labels, label_colors = mne.labels_from_parc(subject, parc='aparc',
                            subjects_dir=subjects_dir, regexp=aparc_label_name)
anat_label = labels[0]

# extract the anatomical time course for each label
stc_anat_label = stc.in_label(anat_label)
pcaAnat = stc.extract_label_time_course(anat_label, src, mode='pca_flip')[0]

stc_func_label = stc.in_label(func_label)
pcaFunc = stc.extract_label_time_course(func_label, src, mode='pca_flip')[0]

# flip the pca so that the max power between tmin and tmax is positive
pcaAnat *= np.sign(pcaAnat[np.argmax(np.abs(pcaAnat))])
pcaFunc *= np.sign(pcaFunc[np.argmax(np.abs(pcaAnat))])

###############################################################################
# plot the time courses....
pl.figure()
pl.plot(1e3 * stc_anat_label.times, pcaAnat, 'k',
        label='Anatomical %s' % aparc_label_name)
pl.plot(1e3 * stc_func_label.times, pcaFunc, 'b',
        label='Active %s' % aparc_label_name)
pl.legend()
pl.show()

###############################################################################
# Plot brain in 3D with PySurfer if available. Note that the subject name
# is already known by the SourceEstimate stc object.
brain = tot_stc.plot(surface='inflated', hemi='lh', subjects_dir=subjects_dir)
brain.scale_data_colormap(fmin=0, fmid=350, fmax=700, transparent=True)
brain.show_view('lateral')

# show both labels
brain.add_label(anat_label, borders=True, color='k')
brain.add_label(func_label, borders=True, color='b')
