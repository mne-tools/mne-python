# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Open the dipole fitting GUI.

Examples
--------
.. code-block:: console

    $ mne dipolefit

"""

import os.path as op

import mne


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "-e",
        "--evoked",
        default=None,
        help='The evoked file ("-ave.fif") containing the data to fit dipoles to.',
    )
    parser.add_option(
        "--condition",
        default=0,
        help="The condition to use.",
    )
    parser.add_option(
        "--baseline-from",
        default=None,
        type=float,
        help="The earliest timepoint to use as baseline.",
    )
    parser.add_option(
        "--baseline-to",
        default=None,
        type=float,
        help="The latest timepoint to use as baseline.",
    )
    parser.add_option(
        "-c",
        "--cov",
        default=None,
        help='The noise covariance ("-cov.fif") to use.',
    )
    parser.add_option(
        "-b",
        "--bem",
        default=None,
        help='The BEM model ("-bem-sol.fif") to use.',
    )
    parser.add_option(
        "-t",
        "--initial-time",
        default=None,
        type=float,
        help="The initial time to show",
    )
    parser.add_option(
        "--trans",
        default=None,
        help='Head<->MRI transform FIF file ("-trans.fif")',
    )
    parser.add_option(
        "--stc",
        default=None,
        help="An optional distributed source estimate to show during dipole fitting.",
    )
    parser.add_option(
        "-s", "--subject", dest="subject", default=None, help="Subject name"
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        default=None,
        help="Subjects directory",
    )
    parser.add_option(
        "--hide-density",
        action="store_true",
        default=False,
        help="Prevent showing the magnetic field density as blobs of color.",
    )
    parser.add_option(
        "--channel-type",
        default=None,
        help=(
            'Restrict channel types to either "meg" or "eeg". By default both are used '
            "if present."
        ),
    )
    parser.add_option(
        "-j", "--cpus", default=-1, type=int, help="Number of CPUs to use."
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()

    # expanduser allows ~ for paths
    subjects_dir = options.subjects_dir
    if subjects_dir is not None:
        subjects_dir = op.expanduser(subjects_dir)
    bem = options.bem
    if bem is not None:
        bem = op.expanduser(bem)
    trans = options.trans
    if trans is not None:
        trans = op.expanduser(trans)
    stc = options.stc
    if stc is not None:
        stc = op.expanduser(stc)
    import faulthandler

    # Condition can be specified as integer index or string comment.
    if options.condition is not None:
        try:
            condition = int(options.condition)
        except ValueError:
            condition = options.condition
    else:
        condition = None

    faulthandler.enable()
    mne.gui.dipolefit(
        evoked=options.evoked,
        condition=condition,
        baseline=(options.baseline_from, options.baseline_to),
        cov=options.cov,
        bem=bem,
        subject=options.subject,
        subjects_dir=subjects_dir,
        stc=stc,
        ch_type=options.channel_type,
        initial_time=options.initial_time,
        trans=trans,
        n_jobs=options.cpus,
        show_density=not options.hide_density,
        show=True,
        block=True,
        verbose=options.verbose,
    )


mne.utils.run_command_if_main()
