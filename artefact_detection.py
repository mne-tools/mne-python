#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:40:37 2019

@author: an512
"""

from scipy.stats import zscore
import mne
import numpy as np

from mne.annotations import Annotations
from scipy.ndimage.measurements import label
from mne.io.ctf.trans import _quaternion_align
from mne.chpi import _apply_quat
from itertools import compress


def detect_bad_channels(raw, zscore_v=4, method='both',
                    neigh_max_distance=.035):
    """ zscore_v = zscore threshold"""
   
    
    # set recording length
    Fs = raw.info['sfreq']
    t1x = 30
    t2x = 220
    t2 = min(raw.last_samp/Fs, t2x)
    t1 = max(0, t1x + t2-t2x)  # Start earlier if recording is shorter

    # Get data
    raw_copy = raw.copy().crop(t1, t2).load_data()
    raw_copy = raw_copy.pick_types(meg=True, ref_meg=False)\
        .filter(1, 45).resample(150, npad='auto')
    data_chans = raw_copy.get_data()

    # Get channel distances matrix
    chns_locs = np.asarray([x['loc'][:3] for x in
                            raw_copy.info['chs']])
    chns_dist = np.linalg.norm(chns_locs - chns_locs[:, None],
                               axis=-1)
    chns_dist[chns_dist > neigh_max_distance] = 0

    # Get avg channel uncorrelation between neighbours
    chns_corr = np.abs(np.corrcoef(data_chans))
    weig = np.array(chns_dist, dtype=bool)
    chn_nei_corr = np.average(chns_corr, axis=1, weights=weig)
    chn_nei_uncorr_z = zscore(1-chn_nei_corr)  # l ower corr higer Z

    # Get channel magnitudes
    max_Pow = np.sqrt(np.sum(data_chans ** 2, axis=1))
    max_Z = zscore(max_Pow)

    if method == 'corr':  # Based on local uncorrelation
        feat_vec = chn_nei_uncorr_z
        max_th = feat_vec > zscore_v
    elif method == 'norm':  # Based on magnitude
        feat_vec = max_Z
        max_th = feat_vec > zscore_v
    elif method == 'both':  # Combine uncorrelation with magnitude
        feat_vec = (chn_nei_uncorr_z+max_Z)/2
        max_th = (feat_vec) > zscore_v

    bad_chns = list(compress(raw_copy.info['ch_names'], max_th))
    return bad_chns


def detect_movement(info, pos, thr_mov=.005):
    
    time = pos[:, 0]
    quats = pos[:, 1:7]

    # Get static head pos from file, used to convert quat to cartesian
    chpi_locs_dev = sorted([d for d in info['hpi_results'][-1]
                            ['dig_points']], key=lambda x: x['ident'])
    chpi_locs_dev = np.array([d['r'] for d in chpi_locs_dev])
    # chpi_locs_dev[0]-> LPA, chpi_locs_dev[1]-> NASION, chpi_locs_dev[2]-> RPA
    # Get head pos changes during recording
    chpi_mov_head = np.array([_apply_quat(quat, chpi_locs_dev, move=True)
                              for quat in quats])

    # get median position across all recording
    chpi_mov_head_f = chpi_mov_head.reshape([-1, 9])  # always 9 chans
    chpi_med_head_tmp = np.median(chpi_mov_head_f, axis=0).reshape([3, 3])

    # get movement displacement from median
    hpi_disp = chpi_mov_head - np.tile(chpi_med_head_tmp, (len(time), 1, 1))
    # get positions above threshold distance
    disp = np.sqrt((hpi_disp ** 2).sum(axis=2))
    disp_exes = np.any(disp > thr_mov, axis=1)

    # Get median head pos during recording under threshold distance
    weights = np.append(time[1:] - time[:-1], 0)
    weights[disp_exes] = 0
    weights /= sum(weights)
    tmp_med_head = weighted_median(chpi_mov_head, weights)
    # Get closest real pos to estimated median
    hpi_disp_th = chpi_mov_head - np.tile(tmp_med_head, (len(time), 1, 1))
    hpi_dist_th = np.sqrt((hpi_disp_th.reshape(-1, 9) ** 2).sum(axis=1))
    chpi_median_pos = chpi_mov_head[hpi_dist_th.argmin(), :, :]

    # Compute displacements from final median head pos
    hpi_disp = chpi_mov_head - np.tile(chpi_median_pos, (len(time), 1, 1))
    hpi_disp = np.sqrt((hpi_disp**2).sum(axis=-1))
    
    art_mask_mov = np.any(hpi_disp > thr_mov, axis=-1)  
    annot = Annotations([], [], [])
    annot += _annotations_from_mask(time, art_mask_mov,
                                    'Bad-motion-dist>%0.3f' % thr_mov)

    # Compute new dev->head transformation from median
    dev_head_t = _quaternion_align(info['dev_head_t']['from'],
                                        info['dev_head_t']['to'],
                                        chpi_locs_dev, chpi_median_pos)
    
    return annot, hpi_disp, dev_head_t

def detect_muscle(raw, thr=1.5, t_min=1):
    """Find and annotate mucsle artifacts - by Luke Bloy"""
    
    raw.pick_types(meg=True, ref_meg=False)
    raw.notch_filter(np.arange(60, 241, 60), fir_design='firwin')
    raw.filter(110, 140, fir_design='firwin')
    raw.apply_hilbert(envelope=True)
    sfreq = raw.info['sfreq']
    art_scores = zscore(raw._data, axis=1)
    # band pass filter the data
    art_scores_filt = mne.filter.filter_data(art_scores.mean(axis=0),
                                             sfreq, None, 4)
    art_mask = art_scores_filt > thr
    # remove artifact free periods shorter than t_min
    idx_min = t_min * sfreq
    comps, num_comps = label(art_mask == 0)
    for l in range(1, num_comps+1):
        l_idx = np.nonzero(comps == l)[0]
        if len(l_idx) < idx_min:
            art_mask[l_idx] = True
    mus_annot = _annotations_from_mask(raw.times, art_mask,
                                       'Bad-muscle')
    return mus_annot, art_scores_filt
        
def weighted_median(data, weights):
    """ by tinybike
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    dims = data.shape
    w_median = np.zeros((dims[1], dims[2]))
    for d1 in range(dims[1]):
        for d2 in range(dims[2]):
            data_dd = np.array(data[:, d1, d2]).squeeze()
            s_data, s_weights = map(np.array, zip(*sorted(zip(
                                                        data_dd, weights))))
            midpoint = 0.5 * sum(s_weights)
            if any(s_weights > midpoint):
                w_median[d1, d2] = (data[weights == np.max(weights)])[0]
            else:
                cs_weights = np.cumsum(s_weights)
                idx = np.where(cs_weights <= midpoint)[0][-1]
                if cs_weights[idx] == midpoint:
                    w_median[d1, d2] = np.mean(s_data[idx:idx+2])
                else:
                    w_median[d1, d2] = s_data[idx+1]
    return w_median

def _annotations_from_mask(times, art_mask, art_name):
    # make annotations - by Luke Bloy
    comps, num_comps = label(art_mask)
    onsets = []
    durations = []
    desc = []
    n_times = len(times)
    for l in range(1, num_comps+1):
        l_idx = np.nonzero(comps == l)[0]
        onsets.append(times[l_idx[0]])
        # duration is to the time after the last labeled time
        # or to the end of the times.
        if 1+l_idx[-1] < n_times:
            durations.append(times[1+l_idx[-1]] - times[l_idx[0]])
        else:
            durations.append(times[l_idx[-1]] - times[l_idx[0]])
        desc.append(art_name)
    return Annotations(onsets, durations, desc)
