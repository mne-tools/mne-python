# Authors: Adonay Nunes <adonay.s.nunes@gmail.com>
#          Luke Bloy <luke.bloy@gmail.com>
# License: BSD (3-clause)


from scipy.ndimage.measurements import label
import numpy as np
from scipy import linalg
from mne.annotations import Annotations, _annotations_starts_stops
from mne.chpi import _apply_quat
from mne.transforms import (quat_to_rot)
from mne import Transform
from mne.utils import _mask_to_onsets_offsets


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
    hp_ts = pos[:, 0].copy()
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
    """Get new device to head transform based on averaged recording.

    It excludes pos that have a BAD_ annotation.
    """
    import warnings

    sfreq = raw.info['sfreq']
    seg_good = np.ones(len(raw.times))
    trans_pos = np.zeros(3)
    hp = pos.copy()
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
