#!/usr/bin/env python
"""Compare FIFF files

You can do for example:

$ mne compare_fiff test_raw.fif test_raw_sss.fif
"""

# Authors : Eric Larson, PhD

import mne


def run():
    parser = mne.commands.utils.get_optparser(__file__)
    options, args = parser.parse_args()
    if len(args) != 2:
        raise ValueError('two arguments required')
    mne.viz.compare_fiff(args[0], args[1])


is_main = (__name__ == '__main__')
if is_main:
    run()
