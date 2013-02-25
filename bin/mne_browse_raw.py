#!/usr/bin/env python
"""Browse raw data

You can do for example:

$mne_browse_raw.py -i sample_audvis_raw.fif
"""

# Authors : Eric Larson, PhD

import sys
import mne


if __name__ == '__main__':

    from optparse import OptionParser
    import pylab as pl

    parser = OptionParser()
    parser.add_option("-i", "--in", dest="raw_in",
                      help="Input raw FIF file", metavar="FILE")
    parser.add_option("-d", "--duration", dest="duration", type="float",
                      help="Time window for plotting (sec)",
                      default=10.0)
    parser.add_option("-t", "--start", dest="start", type="float",
                      help="Initial start time for plotting",
                      default=0.0)
    parser.add_option("-n", "--n_row", dest="n_row", type="int",
                      help="Number of rows to plot at a time",
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
    options, args = parser.parse_args()

    raw_in = options.raw_in
    duration = options.duration
    start = options.start
    n_row = options.n_row
    order = options.order
    preload = options.preload
    show_options = options.show_options

    if raw_in is None:
        parser.print_help()
        sys.exit(-1)

    raw = mne.fiff.Raw(raw_in, preload=preload)
    fig = raw.plot(duration=duration, start=start, n_row=n_row, order=order,
                   show_options=show_options)
    pl.show(block=True)
