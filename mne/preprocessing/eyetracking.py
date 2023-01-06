import numpy as np

from ..annotations import Annotations


def find_blinks(inst, pad=.01, min_overlap=.03, min_dur=.05, max_dur=2., verbose=True):
    """
    this function detects blinks from gaze postion data.
    all values equal to either [nan, 0, -1] will be interpreted as missing.

    """

    from ..preprocessing import annotate_nan
    from ..utils import warn
    from ..io.constants import FIFF

    # find eyetrack channels
    try:
        et_data = inst.copy().pick_types(eyetrack=True)
    except ValueError:
        raise ValueError('no eyetrack channels present in the data. '
                         'if this is wrong, label the channels using '
                         '"mne.channels.set_channel_types()" ')
    et_info = et_data.info

    # find names of relevant channels
    Xids, Yids = [], []
    for ch in et_info['chs']:
        if (ch['coil_type'] == FIFF.FIFFV_COIL_EYETRACK_POS):
            if (ch['loc'][4] == -1):  # X
                Xids.append(ch['ch_name'])
                if (ch['loc'][3] == -1):  # L
                    LXid = ch['ch_name']
                elif (ch['loc'][3] == 1):  # R
                    RXid = ch['ch_name']
            elif (ch['loc'][4] == 1):  # Y
                Yids.append(ch['ch_name'])
                if (ch['loc'][3] == -1):  # L
                    LYid = ch['ch_name']
                elif (ch['loc'][3] == 1):  # R
                    RYid = ch['ch_name']
    if len(Xids) == len(Yids) == 0:
        warn('no information on eyetrack channels was found '
             '(l/r, x/y). if you have binocular recordings consider providing '
             'this information using "set_channelinfo_eyetrack()" as the '
             'detection method is more sensitive then.')
    if verbose:
        print('X position channels: {}'.format(Xids))
        print('Y position channels: {}'.format(Yids))


    # check if data is binocular
    if len(Xids) > 2 or len(Yids) > 2:
        raise ValueError('more then 2 channels detected for X or Y gaze '
                         'position. check your info structure and consider '
                         'fixing it using "set_channelinfo_eyetrack()"')
    is_bino = True if (len(Xids) == len(Yids) == 2) else False
    if verbose:
        print('binocular data: {}'.format(is_bino))

    # assert that we have correct ids
    id_detection = ['RYid' not in locals(), 'LYid' not in locals(),
            'RXid' not in locals(), 'LXid' not in locals()]
    Xids.sort()
    Yids.sort()
    if is_bino and (sum(id_detection) != 0):
        RXid, LXid = Xids
        RYid, LYid = Yids
    elif (not is_bino) and (sum(id_detection) > 2):
        RXid, RYid = Xids[0], Yids[0]
        del LXid, LYid

    # transcribe zeros and -1 to nan, as
    et_data.apply_function(lambda arr: np.where(arr == 0., np.nan, arr))
    et_data.apply_function(lambda arr: np.where(arr == -1., np.nan, arr))

    # find nan periods (blinks) -
    # if binocular, use 2-step procedure (find overlap btw eyes first,
    # then btw gaze directions
    nan_annot = annotate_nan(et_data)
    # filter annotations
    mask = (nan_annot.duration < max_dur) & (nan_annot.duration > min_dur)
    nan_annot = nan_annot[mask]

    if is_bino:
        blink_annot_X = overlapping_annotations(
            nan_annot, nan_annot, 'BLINK_X',
            min_overlap=min_overlap, pad=pad,
            max_dur=max_dur, ch_set=(LXid, RXid))
        blink_annot_Y = overlapping_annotations(
            nan_annot, nan_annot, 'BLINK_Y',
            min_overlap=min_overlap, pad=pad,
            max_dur=max_dur, ch_set=([LYid, RYid]))
        blink_annot = overlapping_annotations(
            blink_annot_X, blink_annot_Y, 'BLINK',
            min_overlap=min_overlap, pad=0,
            max_dur=max_dur, ch_set=(LXid, LYid))
    else:
        blink_annot = overlapping_annotations(
            nan_annot, nan_annot, 'BLINK',
            min_overlap=min_overlap, pad=pad,
            max_dur=max_dur, ch_set=([LXid, RXid, LYid, RYid]))

    if verbose:
        print('identified {} data segments as blinks.'.format(len(blink_annot)))

    return blink_annot


def find_saccades(inst):
    raise NotImplementedError()


def overlapping_annotations(annot1, annot2, desc,
                            min_overlap=0., max_dur=5., pad=.02,
                            ch_set=('LX', 'RX', 'LY', 'RY')):

    assert(annot1.orig_time == annot2.orig_time)

    # adapt ch_set to data
    ch_set_tmp = [ch for ch in ch_set if (
        any([ch in b for b in list(annot1.ch_names)])
    )]
    [ch_set_tmp.append(ch) for ch in ch_set if (
        any([ch in b for b in list(annot2.ch_names)]) and
        (ch not in ch_set_tmp))]
    ch_set = ch_set_tmp

    # find overlapping annotations
    idx_pairs, interval_pairs, overlaps = _find_overlap_within_ch(
        annot1, annot2, ch_set=ch_set)

    # filter overlap duration
    filter_min_overlap = []
    for i, overlap in enumerate(overlaps):
        if overlap >= min_overlap:
            filter_min_overlap.append(i)
    interval_pairs = [interval_pairs[i] for i in filter_min_overlap]

    # merge remaining
    intervals = [_merge_interval(pair[0], pair[1]) for pair in interval_pairs]

    # filter maximum nan duration
    filter_max_duration = []
    for i, interval in enumerate(intervals):
        dur = interval[1] - interval[0]
        if dur <= max_dur:
            filter_max_duration.append(i)
    intervals = [intervals[i] for i in filter_max_duration]

    # create annotations with padding
    n_remaining = len(intervals)
    onset = []
    duration = []
    description = [desc] * n_remaining
    ch_names = [ch_set] * n_remaining
    #ch_names = [None] * n_remaining

    for i in range(n_remaining):
        onset.append(intervals[i][0] - pad)
        duration.append(intervals[i][1] - intervals[i][0] + 2 * pad)

    return Annotations(onset, duration, description,
                       orig_time=annot1.orig_time, ch_names=ch_names)


def _overlapping_interval(interval1, interval2):
    return ((interval1[0] <= interval2[0] <= interval1[1]) or
            (interval1[0] <= interval2[1] <= interval1[1]) or
            (interval2[0] <= interval1[0] <= interval2[1]) or
            (interval2[0] <= interval1[1] <= interval2[1]))


def _merge_interval(interval1, interval2):
    return [min([min(interval1), min(interval2)]),
            max([max(interval1), max(interval2)])]


def _find_overlap_within_ch(
        annot1, annot2,
        ch_set=('LX', 'RX', 'LY', 'RY')):
    pairs = []
    intervals = []
    overlap = []
    # restrict comparison to predefined sets of channels
    for i1, an1 in enumerate(annot1):
        ch1 = an1['ch_names'][0]
        if ch1 not in ch_set:
            continue
        interval1 = [an1['onset'], an1['onset'] + an1['duration']]
        for i2, an2 in enumerate(annot2):
            ch2 = an2['ch_names'][0]
            if ch2 not in ch_set:
                continue
            elif ch1 == ch2:
                continue
            elif (annot1 == annot2) and (i1 >= i2):
                continue
            interval2 = [an2['onset'], an2['onset'] + an2['duration']]
            if _overlapping_interval(interval1, interval2):
                pairs.append([i1, i2])
                intervals.append([interval1, interval2])
                overlap.append(min([interval1[1], interval2[1]]) -
                               max([interval1[0], interval2[0]]))

    return pairs, intervals, overlap

####
#def _merge_annotations_to_intervals(annot1, annot2):
#    intervalmerged = []
#    for an1 in annot1:
#        interval1 = [an1['onset'], an1['onset'] + an1['duration']]
#        for an2 in annot2:
#            interval2 = [an2['onset'], an2['onset'] + an2['duration']]
#            if _overlapping_interval(interval1, interval2):
#                intervalmerged.append(_merge_interval(interval1, interval2))
#    return intervalmerged
