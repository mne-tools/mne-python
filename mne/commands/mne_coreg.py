# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

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
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "-d",
        "--subjects-dir",
        dest="subjects_dir",
        default=None,
        help="Subjects directory",
    )
    parser.add_option(
        "-s", "--subject", dest="subject", default=None, help="Subject name"
    )
    parser.add_option(
        "-f",
        "--fiff",
        dest="inst",
        default=None,
        help="FIFF file with digitizer data for coregistration",
    )
    parser.add_option(
        "--head-opacity",
        type=float,
        default=None,
        dest="head_opacity",
        help="The opacity of the head surface, in the range [0, 1].",
    )
    parser.add_option(
        "--high-res-head",
        action="store_true",
        default=False,
        dest="high_res_head",
        help="Use a high-resolution head surface.",
    )
    parser.add_option(
        "--low-res-head",
        action="store_true",
        default=False,
        dest="low_res_head",
        help="Use a low-resolution head surface.",
    )
    parser.add_option(
        "--trans",
        dest="trans",
        default=None,
        help='Head<->MRI transform FIF file ("-trans.fif")',
    )
    parser.add_option(
        "--interaction",
        type=str,
        default=None,
        dest="interaction",
        help='Interaction style to use, can be "trackball" or "terrain".',
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()

    if options.low_res_head:
        if options.high_res_head:
            raise ValueError(
                "Can't specify --high-res-head and --low-res-head at the same time."
            )
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
        inst=options.inst,
        subject=options.subject,
        subjects_dir=subjects_dir,
        head_opacity=options.head_opacity,
        head_high_res=head_high_res,
        trans=trans,
        interaction=options.interaction,
        show=True,
        block=True,
        verbose=options.verbose,
    )


mne.utils.run_command_if_main()
