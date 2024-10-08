# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""Create BEM surfaces using the watershed algorithm included with FreeSurfer.

Examples
--------
.. code-block:: console

    $ mne watershed_bem -s sample

"""

import sys

import mne
from mne.bem import make_watershed_bem
from mne.utils import _check_option


def run():
    """Run command."""
    from mne.commands.utils import _add_verbose_flag, get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "-s", "--subject", dest="subject", help="Subject name (required)", default=None
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        dest="subjects_dir",
        help="Subjects directory",
        default=None,
    )
    parser.add_option(
        "-o",
        "--overwrite",
        dest="overwrite",
        help="Write over existing files",
        action="store_true",
    )
    parser.add_option(
        "-v", "--volume", dest="volume", help="Defaults to T1", default="T1"
    )
    parser.add_option(
        "-a",
        "--atlas",
        dest="atlas",
        help="Specify the --atlas option for mri_watershed",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "-g",
        "--gcaatlas",
        dest="gcaatlas",
        help="Specify the --brain_atlas option for mri_watershed",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "-p",
        "--preflood",
        dest="preflood",
        help="Change the preflood height",
        default=None,
    )
    parser.add_option(
        "--copy",
        dest="copy",
        help="Use copies instead of symlinks for surfaces",
        action="store_true",
    )
    parser.add_option(
        "-t",
        "--T1",
        dest="T1",
        help="Whether or not to pass the -T1 flag "
        "(can be true, false, 0, or 1). "
        "By default it takes the same value as gcaatlas.",
        default=None,
    )
    parser.add_option(
        "-b",
        "--brainmask",
        dest="brainmask",
        help="The filename for the brainmask output file "
        "relative to the "
        "$SUBJECTS_DIR/$SUBJECT/bem/watershed/ directory.",
        default="ws",
    )
    _add_verbose_flag(parser)

    options, args = parser.parse_args()

    if options.subject is None:
        parser.print_help()
        sys.exit(1)

    subject = options.subject
    subjects_dir = options.subjects_dir
    overwrite = options.overwrite
    volume = options.volume
    atlas = options.atlas
    gcaatlas = options.gcaatlas
    preflood = options.preflood
    copy = options.copy
    brainmask = options.brainmask
    T1 = options.T1
    if T1 is not None:
        T1 = T1.lower()
        _check_option("--T1", T1, ("true", "false", "0", "1"))
        T1 = T1 in ("true", "1")
    verbose = options.verbose

    make_watershed_bem(
        subject=subject,
        subjects_dir=subjects_dir,
        overwrite=overwrite,
        volume=volume,
        atlas=atlas,
        gcaatlas=gcaatlas,
        preflood=preflood,
        copy=copy,
        T1=T1,
        brainmask=brainmask,
        verbose=verbose,
    )


mne.utils.run_command_if_main()
