#!/usr/bin/env python
# Authors: Christian Brodbeck  <christianbrodbeck@nyu.edu>

""" Open the coregistration GUI.

example usage:  $ mne coreg

"""

import os
import sys

import mne


if __name__ == '__main__':
    os.environ['ETS_TOOLKIT'] = 'qt4'
    mne.gui.coregistration()
    sys.exit(0)
