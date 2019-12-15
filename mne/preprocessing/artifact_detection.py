# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)


from scipy.stats import zscore
from scipy.ndimage.measurements import label
import numpy as np
from itertools import compress
from scipy import linalg

from mne.filter import filter_data
from mne.annotations import Annotations, _annotations_starts_stops
from mne.chpi import _apply_quat
from mne.transforms import (quat_to_rot)
from mne import Transform
from mne.utils import _mask_to_onsets_offsets


def detect_bad_channels(raw, zscore_v=4, method='both', tmin=30, tmax=220,
                        neigh_max_distance=.035):
    """Detect bad channels.

    Detection can be based on z-score amplitude deviation or/and decreased
    local correlation with other channels.

    Notes
    -----
    This helps in detecting suspicious channels, visual inspection warranted.

    Parameters
    ----------
    raw : instance of Raw
        The data.
    zscore : int
        The z-score value to reject channels if exceeding it.
    method : 'corr | 'norm' | 'both'
        The criteria used to detect bad channels.
        'corr' - correlation with all neighbours not exceeding
               ``neigh_max_distance``.
        'norm' - averaged magnitude.
        'both' - mean of the z-scored 'corr' and 'norm' methods.
    tmin : int
        Start time in seconds of the signal segment used for bad chn detection.
    tmax : int
        End time in seconds of the signal segment used to detect bad channels.
    neigh_max_distance : float
        Maximum channel distance in meters for local correlation.

    Returns
    -------
    bad_chs : list
        List of detected bad channels.
    """
    # set recording length
    sfreq = raw.info['sfreq']
    t2x = min(raw.last_samp / sfreq, tmax)
    t1x = max(0, tmin + t2x - tmax)  # Start earlier if recording is shorter

    # Get data
    raw_copy = raw.copy().crop(t1x, t2x).load_data()
    raw_copy = raw_copy.pick_types(meg=True, ref_meg=False)\
        .filter(1, 45).resample(150, npad='auto')
    data_chans = raw_copy.get_data()

    # Get channel distances matrix
    ch_locs = np.asarray([x['loc'][:3] for x in raw_copy.info['chs']])
    chns_dist = np.linalg.norm(ch_locs - ch_locs[:, None],
                               axis=-1)
    chns_dist[chns_dist > neigh_max_distance] = 0

    if method == 'corr' or method == 'both':
        # Get avg channel uncorrelation between neighbours
        chns_corr = np.abs(np.corrcoef(data_chans))
        weig = np.array(chns_dist, dtype=bool)
        chn_nei_corr = np.average(chns_corr, axis=1, weights=weig)
        chn_nei_uncorr_z = zscore(1 - chn_nei_corr)  # lower corr higher Z

    # Get channel magnitudes
    max_pow = np.sqrt(np.sum(data_chans ** 2, axis=1))
    max_Z = zscore(max_pow)

    if method == 'corr':  # Based on local uncorrelation
        feat_vec = chn_nei_uncorr_z
        max_th = feat_vec > zscore_v
    elif method == 'norm':  # Based on magnitude
        feat_vec = max_Z
        max_th = feat_vec > zscore_v
    elif method == 'both':  # Combine uncorrelation with magnitude
        feat_vec = (chn_nei_uncorr_z + max_Z) / 2
        max_th = (feat_vec) > zscore_v

    bad_chs = list(compress(raw_copy.info['ch_names'], max_th))
    return bad_chs


def annotate_movement(raw, pos, rotation_limit=None,
                      translation_vel_limit=None, displacement_limit=None):
    """Detect segments with movement velocity or displacement from mean.

    First, the cHPI is calculated relative to the default head position, then
    segments beyond the threshold are discarded and the median head pos is
    calculated. Time points further thr_mov from the median are annotated and
    a new device to head transformation is calculated only with the good
    segments.

    Parameters
    ----------
    info : structure
        From raw.info
    pos : array, shape (N, 10)
        The position and quaternion parameters from cHPI fitting.
    thr_mov : int
        in meters, the maximal distance allowed from the median cHPI

    Returns
    -------
    annot : mne.Annotations
        periods where head position was too far
    hpi_disp : array
        head position over time w.r.t the median head pos
    dev_head_t : array
        new trans matrix using accepted head pos
    """

    sfreq = raw.info['sfreq']
    hp_ts = pos[:, 0]
    hp_ts -= raw.first_samp / sfreq
    dt = np.diff(hp_ts)
    seg_good = np.append(dt, 1. / sfreq)
    hp_ts = np.concatenate([hp_ts, [hp_ts[-1] + 1. / sfreq]])

    annot = Annotations([], [], [], orig_time=None)  # rel to data start

    # Mark down times that are bad according to annotations
    onsets, ends = _annotations_starts_stops(raw, 'bad')
    for onset, end in zip(onsets, ends):
        seg_good[onset:end] = 0

    # Annotate based on rotational velocity
    t_tot = raw.times[-1]
    if rotation_limit is not None:
        from mne.transforms import _angle_between_quats
        assert rotation_limit > 0
        # Rotational velocity (radians / sec)
        r = _angle_between_quats(pos[:-1, 1:4], pos[1:, 1:4])
        r /= dt
        bad_mask = (r >= np.deg2rad(rotation_limit))
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        print(u'Omitting %5.1f%% (%3d segments): '
              u'ω >= %5.1f°/s (max: %0.1f°/s)'
              % (bad_pct, len(onsets), rotation_limit,
                 np.rad2deg(r.max())))
        annot += _annotations_from_mask(hp_ts, bad_mask, 'BAD_rotat_vel')

    # Annotate based on translational velocity
    if translation_vel_limit is not None:
        assert translation_vel_limit > 0
        v = np.linalg.norm(np.diff(pos[:, 4:7], axis=0), axis=-1)
        v /= dt
        bad_mask = (v >= translation_vel_limit)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        print(u'Omitting %5.1f%% (%3d segments): '
              u'v >= %5.4fm/s (max: %5.4fm/s)'
              % (bad_pct, len(onsets), translation_vel_limit, v.max()))
        for onset, offset in zip(onsets, offsets):
            annot.append(onset, offset - onset, 'BAD_trans_vel')
        annot += _annotations_from_mask(hp_ts, bad_mask, 'BAD_trans_vel')

    # Annotate based on displacement from mean
    disp = []
    if displacement_limit is not None:
        assert displacement_limit > 0
        # Get static head pos from file, used to convert quat to cartesian
        chpi_pos = sorted([d for d in raw.info['hpi_results'][-1]
                          ['dig_points']], key=lambda x: x['ident'])
        chpi_pos = np.array([d['r'] for d in chpi_pos])
        # CTF: chpi_pos[0]-> LPA, chpi_pos[1]-> NASION, chpi_pos[2]-> RPA
        # Get head pos changes during recording
        chpi_pos_mov = np.array([_apply_quat(quat, chpi_pos, move=True)
                                for quat in pos[:, 1:7]])

        # get average position
        chpi_pos_avg = np.average(chpi_pos_mov, axis=0, weights=seg_good)

        # get movement displacement from mean pos
        hpi_disp = chpi_pos_mov - np.tile(chpi_pos_avg, (len(seg_good), 1, 1))
        # get positions above threshold distance
        disp = np.sqrt((hpi_disp ** 2).sum(axis=2))
        bad_mask = np.any(disp > displacement_limit, axis=1)
        onsets, offsets = _mask_to_onsets_offsets(bad_mask)
        onsets, offsets = hp_ts[onsets], hp_ts[offsets]
        bad_pct = 100 * (offsets - onsets).sum() / t_tot
        print(u'Omitting %5.1f%% (%3d segments): '
              u'disp >= %5.4fm/s (max: %5.4fm)'
              % (bad_pct, len(onsets), displacement_limit, disp.max()))
        annot += _annotations_from_mask(hp_ts, bad_mask, 'BAD_mov_disp')
    return annot, disp


def compute_average_dev_head_t(raw, pos):
    '''Get new device to head transform based on averaged recording

    It excludes pos that have a BAD_ annotation'''

    import warnings

    sfreq = raw.info['sfreq']
    seg_good = np.ones(len(raw.times))
    trans_pos = np.zeros(3)
    hp = pos
    hp_ts = hp[:, 0] - raw._first_time

    # Check rounding issues at 0 time
    if hp_ts[0] < 0:
        hp_ts[0] = 0
        assert hp_ts[1] > 1. / sfreq

    # Mask out segments if beyond scan time
    mask = hp_ts <= raw.times[-1]
    if not mask.all():
        warnings.warn(
            '          Removing %0.1f%% time points > raw.times[-1] (%s)'
            % ((~mask).sum() / float(len(mask)), raw.times[-1]))
        hp = hp[mask]
    del mask, hp_ts

    # Get time indices
    ts = np.concatenate((hp[:, 0], [(raw.last_samp + 1) / sfreq]))
    assert (np.diff(ts) > 0).all()
    ts -= raw.first_samp / sfreq
    idx = raw.time_as_index(ts, use_rounding=True)
    del ts
    if idx[0] == -1:  # annoying rounding errors
        idx[0] = 0
        assert idx[1] > 0
    assert (idx >= 0).all()
    assert idx[-1] == len(seg_good)
    assert (np.diff(idx) > 0).all()

    # Mark times bad that are bad according to annotations
    onsets, ends = _annotations_starts_stops(raw, 'bad')
    for onset, end in zip(onsets, ends):
        seg_good[onset:end] = 0
    dt = np.diff(np.cumsum(np.concatenate([[0], seg_good]))[idx])
    assert (dt >= 0).all()
    dt = dt / sfreq
    del seg_good, idx

    # Get weighted head pos trans and rot
    trans_pos += np.dot(dt, hp[:, 4:7])
    rot_qs = hp[:, 1:4]
    res = 1 - np.sum(rot_qs * rot_qs, axis=-1, keepdims=True)
    assert (res >= 0).all()
    rot_qs = np.concatenate((rot_qs, np.sqrt(res)), axis=-1)
    assert np.allclose(np.linalg.norm(rot_qs, axis=1), 1)
    rot_qs *= dt[:, np.newaxis]
    # rank 1 update method
    # https://arc.aiaa.org/doi/abs/10.2514/1.28949?journalCode=jgcd
    # https://github.com/tolgabirdal/averaging_quaternions/blob/master/wavg_quaternion_markley.m  # noqa: E501
    # qs.append(rot_qs)
    outers = np.einsum('ij,ik->ijk', rot_qs, rot_qs)
    A = outers.sum(axis=0)
    dt_sum = dt.sum()
    assert dt_sum >= 0
    norm = dt_sum
    if norm <= 0:
        raise RuntimeError('No good segments found (norm=%s)' % (norm,))
    A /= norm

    best_q = linalg.eigh(A)[1][:, -1]  # largest eigenvector is the wavg
    # Same as the largest eigenvector from the concatenation of all
    # best_q = linalg.svd(np.concatenate(qs).T)[0][:, 0]
    best_q = best_q[:3] * np.sign(best_q[-1])
    trans = np.eye(4)
    trans[:3, :3] = quat_to_rot(best_q)
    trans[:3, 3] = trans_pos / norm
    assert np.linalg.norm(trans[:3, 3]) < 1  # less than 1 meter is sane
    dev_head_t = Transform('meg', 'head', trans)
    return dev_head_t


def annotate_muscle(raw, thr=1.5, t_min=1, notch=[60, 120, 180]):
    """Find and annotate mucsle artifacts based on high frequency activity.

    Data is band pass filtered at high frequencies, smoothed with the envelope,
    z-scored, channel averaged and low-pass filtered to remove transient peaks.

    Parameters
    ----------
    raw : instance of Raw
        The data.
    thr : integer
        threshold where a segment is marked as artifactual. The thr represent
        the channel averaged z-score values of the data band pass filtered
        betweehn 110 and 140 Hz.
    t_min = integer
        Min. time between annotated muscle artifacts, below will be annotated
        as muscle artifact.
    notch = List
        The frequencies at which to apply a notch filter

    Returns
    -------
    annot : mne.Annotations
        periods with muscle artifacts
    art_scores_filt : array
        the z-score values of the filtered data
    """
    raw.pick_types(meg=True, ref_meg=False)
    if notch is not None:
        raw.notch_filter(notch, fir_design='firwin')
    raw.filter(110, 140, fir_design='firwin')
    raw.apply_hilbert(envelope=True)
    sfreq = raw.info['sfreq']
    art_scores = zscore(raw._data, axis=1)
    # band pass filter the data
    art_scores_filt = filter_data(art_scores.mean(axis=0), sfreq, None, 4)
    art_mask = art_scores_filt > thr
    # remove artifact free periods shorter than t_min
    idx_min = t_min * sfreq
    comps, num_comps = label(art_mask == 0)
    for l in range(1, num_comps + 1):
        l_idx = np.nonzero(comps == l)[0]
        if len(l_idx) < idx_min:
            art_mask[l_idx] = True
    mus_annot = _annotations_from_mask(raw.times, art_mask,
                                       'BAD_muscle')
    return mus_annot, art_scores_filt


def _annotations_from_mask(times, art_mask, art_name):
    """Construct annotations from boolean mask of the data."""
    comps, num_comps = label(art_mask)
    onsets, durations, desc = [], [], []
    n_times = len(times)
    for l in range(1, num_comps + 1):
        l_idx = np.nonzero(comps == l)[0]
        onsets.append(times[l_idx[0]])
        # duration is to the time after the last labeled time
        # or to the end of the times.
        if 1 + l_idx[-1] < n_times:
            durations.append(times[1 + l_idx[-1]] - times[l_idx[0]])
        else:
            durations.append(times[l_idx[-1]] - times[l_idx[0]])
        desc.append(art_name)
    return Annotations(onsets, durations, desc)
