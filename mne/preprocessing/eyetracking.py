from ..annotations import Annotations


def find_blinks(inst, pad=.01, min_overlap=.02, max_dur=5.):
    from ..preprocessing import annotate_nan

    # find eyetrack channels
    et_data = inst.copy().pick_types(eyetrack=True)
    et_info = et_data.info

    # make sure there's only annotations for channels in the data


    # check if data is binocular
    loc_eye = [ch['loc'][3] for ch in et_info['chs']]
    is_bino = True if (-1 and 1 in loc_eye) else False

    # find blinks -
    # if binocular, use 2-step procedure (find overlap btw eyes first,
    # then btw gaze directions
    nan_annot = annotate_nan(et_data)
    if is_bino:
        blink_annot_X = overlapping_annotations(
            nan_annot, nan_annot, 'BLINK_X',
            min_overlap=min_overlap, pad=pad,
            max_dur=max_dur, ch_set=('LX', 'RX'))
        blink_annot_Y = overlapping_annotations(
            nan_annot, nan_annot, 'BLINK_Y',
            min_overlap=min_overlap, pad=pad,
            max_dur=max_dur, ch_set=(['LY', 'RY']))
        blink_annot = overlapping_annotations(
            blink_annot_X, blink_annot_Y, 'BLINK',
            min_overlap=min_overlap, pad=0,
            max_dur=max_dur, ch_set=('LX', 'LY'))
    else:
        blink_annot = overlapping_annotations(
            nan_annot, nan_annot, 'BLINK',
            min_overlap=min_overlap, pad=pad,
            max_dur=max_dur, ch_set=(['LX', 'RX', 'LY', 'RY']))

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
