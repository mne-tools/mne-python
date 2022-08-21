#!/usr/bin/env python
# Authors: Christian Brodbeck  <christianbrodbeck@nyu.edu>

"""Open the coregistration GUI.

Examples
--------
.. code-block:: console

    $ mne coreg

"""

import os.path as op

import mne


def run():
    """Run command."""
    from mne.commands.utils import get_optparser, _add_verbose_flag

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
    parser.add_option('--interaction',
                      type=str, default=None, dest='interaction',
                      help='Interaction style to use, can be "trackball" or '
                      '"terrain".')
    parser.add_option('--scale',
                      type=float, default=None, dest='scale',
                      help='Scale factor for the scene.')
    parser.add_option('--simple-rendering', action='store_false',
                      dest='advanced_rendering',
                      help='Use simplified OpenGL rendering')
    _add_verbose_flag(parser)

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

    # expanduser allows ~ for --subjects-dir
    subjects_dir = options.subjects_dir
    if subjects_dir is not None:
        subjects_dir = op.expanduser(subjects_dir)
    trans = options.trans
    if trans is not None:
        trans = op.expanduser(trans)
    import faulthandler
    faulthandler.enable()
    mne.gui.coregistration(
        options.tabbed, inst=options.inst, subject=options.subject,
        subjects_dir=subjects_dir,
        guess_mri_subject=options.guess_mri_subject,
        head_opacity=options.head_opacity, head_high_res=head_high_res,
        trans=trans, scrollable=True,
        interaction=options.interaction,
        scale=options.scale,
        advanced_rendering=options.advanced_rendering,
        show=True, block=True,
        verbose=options.verbose)


mne.utils.run_command_if_main()
