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
                      action='store_false', default=None,
                      help="Prevent the GUI from automatically guessing and "
                      "changing the MRI subject when a new head shape source "
                      "file is selected.")
    parser.add_option("--head-opacity", type=float, default=None,
                      dest="head_opacity",
                      help="The opacity of the head surface, in the range "
                      "[0, 1].")
    parser.add_option("--high-res-head",
                      action='store_true', default=False, dest="high_res_head",
                      help="Use a high-resolution head surface.")
    parser.add_option("--low-res-head",
                      action='store_true', default=False, dest="low_res_head",
                      help="Use a low-resolution head surface.")
    parser.add_option('--trans', dest='trans', default=None,
                      help='Head<->MRI transform FIF file ("-trans.fif")')
    parser.add_option('--verbose', action='store_true', dest='verbose',
                      help='Turn on verbose mode.')

    options, args = parser.parse_args()

    if options.low_res_head:
        if options.high_res_head:
            raise ValueError("Can't specify --high-res-head and "
                             "--low-res-head at the same time.")
        head_high_res = False
    elif options.high_res_head:
        head_high_res = True
    else:
        head_high_res = None

    with ETSContext():
        mne.gui.coregistration(options.tabbed, inst=options.inst,
                               subject=options.subject,
                               subjects_dir=options.subjects_dir,
                               guess_mri_subject=options.guess_mri_subject,
                               head_opacity=options.head_opacity,
                               head_high_res=head_high_res,
                               trans=options.trans,
                               verbose=options.verbose)
    if is_main:
        sys.exit(0)

is_main = (__name__ == '__main__')
if is_main:
    run()
