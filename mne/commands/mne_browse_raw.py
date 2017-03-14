#!/usr/bin/env python
r"""Browse raw data.

You can do for example:

$ mne browse_raw --raw sample_audvis_raw.fif \
                 --proj sample_audvis_ecg-proj.fif \
                 --eve sample_audvis_raw-eve.fif
"""

# Authors : Eric Larson, PhD

import sys
import mne


def run():
    """Run command."""
    import matplotlib.pyplot as plt

    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("--raw", dest="raw_in",
                      help="Input raw FIF file", metavar="FILE")
    parser.add_option("--proj", dest="proj_in",
                      help="Projector file", metavar="FILE",
                      default='')
    parser.add_option("--eve", dest="eve_in",
                      help="Events file", metavar="FILE",
                      default='')
    parser.add_option("-d", "--duration", dest="duration", type="float",
                      help="Time window for plotting (sec)",
                      default=10.0)
    parser.add_option("-t", "--start", dest="start", type="float",
                      help="Initial start time for plotting",
                      default=0.0)
    parser.add_option("-n", "--n_channels", dest="n_channels", type="int",
                      help="Number of channels to plot at a time",
                      default=20)
    parser.add_option("-o", "--order", dest="order",
                      help="Order for plotting ('type' or 'original')",
                      default='type')
    parser.add_option("-p", "--preload", dest="preload",
                      help="Preload raw data (for faster navigaton)",
                      default=False)
    parser.add_option("-s", "--show_options", dest="show_options",
                      help="Show projection options dialog",
                      default=False)
    parser.add_option("--allowmaxshield", dest="maxshield",
                      help="Allow loading MaxShield processed data",
                      action="store_true")
    parser.add_option("--highpass", dest="highpass", type="float",
                      help="Display high-pass filter corner frequency",
                      default=-1)
    parser.add_option("--lowpass", dest="lowpass", type="float",
                      help="Display low-pass filter corner frequency",
                      default=-1)
    parser.add_option("--filtorder", dest="filtorder", type="int",
                      help="Display filtering IIR order",
                      default=4)
    parser.add_option("--clipping", dest="clipping",
                      help="Enable trace clipping mode, either 'clip' or "
                      "'transparent'", default=None)
    parser.add_option("--filterchpi", dest="filterchpi",
                      help="Enable filtering cHPI signals.", default=None)

    options, args = parser.parse_args()

    raw_in = options.raw_in
    duration = options.duration
    start = options.start
    n_channels = options.n_channels
    order = options.order
    preload = options.preload
    show_options = options.show_options
    proj_in = options.proj_in
    eve_in = options.eve_in
    maxshield = options.maxshield
    highpass = options.highpass
    lowpass = options.lowpass
    filtorder = options.filtorder
    clipping = options.clipping
    filterchpi = options.filterchpi

    if raw_in is None:
        parser.print_help()
        sys.exit(1)

    raw = mne.io.read_raw_fif(raw_in, preload=preload,
                              allow_maxshield=maxshield)
    if len(proj_in) > 0:
        projs = mne.read_proj(proj_in)
        raw.info['projs'] = projs
    if len(eve_in) > 0:
        events = mne.read_events(eve_in)
    else:
        events = None

    if filterchpi:
        if not preload:
            raise RuntimeError(
                'Raw data must be preloaded for chpi, use --preload')
        raw = mne.chpi.filter_chpi(raw)

    highpass = None if highpass < 0 or filtorder <= 0 else highpass
    lowpass = None if lowpass < 0 or filtorder <= 0 else lowpass
    filtorder = 4 if filtorder <= 0 else filtorder
    raw.plot(duration=duration, start=start, n_channels=n_channels,
             order=order, show_options=show_options, events=events,
             highpass=highpass, lowpass=lowpass, filtorder=filtorder,
             clipping=clipping)
    plt.show(block=True)


is_main = (__name__ == '__main__')
if is_main:
    run()
