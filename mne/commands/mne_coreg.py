#!/usr/bin/env python
# Authors: Christian Brodbeck  <christianbrodbeck@nyu.edu>

"""Open the coregistration GUI.

example usage:  $ mne coreg
"""

import sys

import mne
from mne.utils import ETSContext


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      default=None, help="Subjects directory")
    parser.add_option("-s", "--subject", dest="subject", default=None,
                      help="Subject name")
    parser.add_option("-f", "--fiff", dest="inst", default=None,
                      help="FIFF file with digitizer data for coregistration")
    parser.add_option("-t", "--tabbed", dest="tabbed", action="store_true",
                      default=False, help="Option for small screens: Combine "
                      "the data source panel and the coregistration panel "
                      "into a single panel with tabs.")
    parser.add_option("--no-guess-mri", dest="guess_mri_subject",
                      action='store_false', default=True,
                      help="Prevent the GUI from automatically guessing and "
                      "changing the MRI subject when a new head shape source "
                      "file is selected.")
    parser.add_option("--head-opacity", type=float, default=1.,
                      dest="head_opacity",
                      help="The opacity of the head surface, in the range "
                      "[0, 1].")
    parser.add_option("--low-res-head",
                      action='store_true', default=False, dest="low_res_head",
                      help="Default to using a low-resolution head surface.")
    parser.add_option('--verbose', action='store_true', dest='verbose',
                      help='Turn on verbose mode.')

    options, args = parser.parse_args()

    with ETSContext():
        mne.gui.coregistration(options.tabbed, inst=options.inst,
                               subject=options.subject,
                               subjects_dir=options.subjects_dir,
                               guess_mri_subject=options.guess_mri_subject,
                               head_opacity=options.head_opacity,
                               head_high_res=not options.low_res_head,
                               verbose=options.verbose)
    if is_main:
        sys.exit(0)

is_main = (__name__ == '__main__')
if is_main:
    run()
