from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.utils import logger
from mne.io import Raw
from mne.datasets import sample
from mne.io.pick import channel_type
from mne.channels import layout
from mne.defaults import _handle_default
from mne.viz import tight_layout

data_path = sample.data_path()

raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

tmin, tmax = -.1, .6

raw = Raw(raw_fname, preload=True)
raw.filter(.5, None, method="iir")
events = mne.read_events(event_fname)
event_id = dict(audio_l=1, audio_r=2, visual_l=3, visual_r=4)

raw.apply_proj()
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=None)

# stuff the evoked objects into a dictionary with conditions for keys
evokeds = dict((cond, epochs[cond].average()) for cond in event_id.keys())


def _percentiles(ch_names=None, pos=None,
                 rows=None, columns=None,
                 column_names=None, row_names=None):
    """find rois by x/y percentiles of rectangle positions"""

    arr = np.zeros(pos[:, :2].shape)

    pctl_x = [v / float(columns) * 100 for v in range(1, columns)]
    pctl_y = [v / float(rows) * 100 for v in range(1, rows)]

    bounds = np.percentile(pos[:, 0], pctl_x)
    for bound in bounds:
        arr[np.where(pos[:, 0] < bound), 0] += 1
    bounds = np.percentile(pos[:, 1], pctl_y)
    for bound in bounds:
        arr[np.where(pos[:, 1] < bound), 1] += 1

    roi_list = [(column_names[x] + " " + row_names[y])
                for (x, y) in arr.astype(int)]

    roi_to_chan = defaultdict(list)
    for ch, roi in zip(ch_names, roi_list):
        roi_to_chan[roi].append(ch)

    return roi_to_chan


def plot_roi(evokeds,
             column_names=["Left", "Midline", "Right"],
             row_names=["Frontal", "Central", "Posterior"],
             roi_to_chan="percentiles", ch_type='eeg',
             conds=None):
    """Plot evoked data per ROI, multiple conditions

    By assigning channels to regions of interest/ROIs, this function can
    summarize multi-channel data for multiple conditions. ROIs can be
    constructed automatically, or supplied as a dictionary. ROIs are
    assigned in a rectangular fashion.

    Note: in this form, gradiometers may prove problematic.
    Also, top left/right and bottom left/right ROIs will often
    feature few channels.

    Parameters
    ----------

    evokeds : dict
        A dictionary where the keys are condition names and the values
        Evoked objects.
    column_names, row_names : list of str
        Define the length and naming of the horizontal and vertical ROI
        selections, going through channels from left to right and from
        frontal to posterior. Names are arbitrary - the automatic ROI
        constructor does not understand what e.g. "frontal" means.
        If too many ROIs are asked for, or a problematic asymmetric
        split requested, warnings are raised.
    roi_to_chan : str | dict
        If `str`, must be 'percentiles' (for percentile-based ROI
        construction). Else, must be a dict that exhaustively maps ROIs
        (keys) to channels (values).
    ch_type : str
        Channel type to be picked from the evoked instances.
    conds : None | list
        If None, all members of `evokeds` are plotted. Else, the keys in
        `conds` (should be `str`) will be used to choose a subset of the
        values in `evokeds`.

    Returns
    -------
    f : Instance of matplotlib.figure.Figure
        Images of evoked responses per condition, averaged by ROI.
        f.rois holds the roi-to-channel dict.
    """

    evokeds = dict((cond, evoked.copy())
                   for cond, evoked in evokeds.items())
    info = list(evokeds.values())[0].info

    picks = [ch_name for idx, ch_name in enumerate(info["ch_names"])
             if channel_type(info, idx) == ch_type]

    for evoked in evokeds.values():
        evoked.pick_channels(picks)
    if conds is not None:
        evokeds = dict((k, v) for k, v in evokeds.items() if k in conds)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    columns, rows = len(column_names), len(row_names)

    if not columns % 2:
        w = ("An even number of coronal cuts is requested. "
             "This will lead to strange results if midline electrodes "
             "are used.")
        logger.warning(w)
        warnings.warn(w)

    ch_names = info["ch_names"]
    pos = layout.find_layout(info).pos
    for p in pos:
        p[0] += p[2] / 2
        p[1] += p[3] / 2

    # we construct dicts and lists to map between electrodes, ROIs, etc & back
    # and check these to know what ROI an electrode ended up in
    if not isinstance(roi_to_chan, dict):
        roi_to_chan = _percentiles(ch_names=ch_names, pos=pos,
                                   rows=rows, columns=columns,
                                   row_names=row_names,
                                   column_names=column_names)
    roi_sel = ", ".join([": ".join((k, ", ".join(v)))
                        for k, v in roi_to_chan.items()])
    logger.info("ROIs: " + roi_sel)

    if len(roi_to_chan) < (columns * rows):
        error = "Sensor coverage too sparse for requested layout."
        raise ValueError(error)

    if any([len(v) == 1 for v in roi_to_chan.values()]):
        w = ("Sensor coverage sparse compared to requested layout; "
             "some ROIs only contain one channel.")
        logger.warning(w)
        warnings.warn(w)

    # now the actual plot ...
    f, splts = plt.subplots(rows, columns, sharex=True, sharey=True)

    s = _handle_default('scalings')[ch_type]
    tsx, tsy = None, None
    max_y, c_ps = [], []
    for r, p_line in zip(row_names, splts):
        for c, p in zip(column_names, p_line):
            roi = c + " " + r
            ch = roi_to_chan[roi]
            for cond, color in zip(evokeds.keys(), colors):
                if len(roi_to_chan[roi]) > 1:
                    d = evokeds[cond].pick_channels(ch, copy=True).data
                    p.plot(evokeds[cond].times, np.mean(d, axis=0) * s,
                           color=color)
                elif len(roi_to_chan[roi]) == 1:
                    d = evokeds[cond].pick_channels(ch, copy=True).data.T
                    p.plot(evokeds[cond].times, d * s, color=color)
            if tsx is None:
                old_x = [x for x in p.get_xticks() if x > 0]
                tsx = (0, old_x[int((len(old_x) - 1) / 2)])

            if r in row_names[-1]:
                p.set_xticks(tsx)
            if r in row_names[0]:
                p.set_xlabel(c)
                p.xaxis.set_label_position("top")
            if c in column_names[-1]:
                p.set_ylabel(r, labelpad=15)
                p.yaxis.set_label_position("right")
                p.yaxis.get_label().set_rotation(270)
            plt.suptitle(ch_type.upper() + " channel ROIs", y=.95)

            max_y.append(max(abs(p.get_yticks())))
            if c in column_names[0]:
                c_ps.append(p)
            if r in row_names[-1] and c in column_names[int(rows / 2)]:
                p.set_xlabel("time (s)")
            if c in column_names[0] and r in row_names[int(columns / 2)]:
                p.set_ylabel(_handle_default('units')[ch_type])
            if c == column_names[-1] and r == row_names[-1]:
                p.legend(p.get_lines(), evokeds.keys(), loc='lower right',
                         frameon=True, ncol=4)

    max_y.sort()
    y_max = max_y[-1]
    tsy = (y_max / -2, 0, y_max / 2)
    for p in c_ps:
        p.set_yticks(tsy)

    f.rois = roi_to_chan

    return f

column_names = ["Left", "Left-mid",  "Midline", "Right-mid", "Right"]
row_names = ["Frontal", "Central", "Posterior", "Occipital"]

fs = plot_roi(evokeds, column_names=column_names, row_names=row_names,
              roi_to_chan="percentiles")

fs.set_size_inches((12, 12))
plt.show()
tight_layout()
