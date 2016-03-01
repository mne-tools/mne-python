#!/usr/bin/env python
# Authors: Christian Brodbeck  <christianbrodbeck@nyu.edu>

"""Open the coregistration GUI.

example usage:  $ mne coreg

"""

import os
import sys

import mne


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)
    options, args = parser.parse_args()

    os.environ['ETS_TOOLKIT'] = 'qt4'
    mne.gui.coregistration()
    if is_main:
        sys.exit(0)

is_main = (__name__ == '__main__')
if is_main:
    run()
