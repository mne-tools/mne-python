#!/usr/bin/env python
"""Compare FIFF files.

You can do for example:

$ mne compare_fiff test_raw.fif test_raw_sss.fif
"""

# Authors : Eric Larson, PhD

import sys
import mne


def run():
    """Run command."""
    parser = mne.commands.utils.get_optparser(
        __file__, usage='mne compare_fiff <file_a> <file_b>')
    options, args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    mne.viz.compare_fiff(args[0], args[1])


is_main = (__name__ == '__main__')
if is_main:
    run()
