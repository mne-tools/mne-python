"""Create 3-layer BEM model from Flash MRI images.

Examples
--------
.. code-block:: console

    $ mne flash_bem --subject=sample
    $ mne flash_bem -s sample -n --registered -5 sample/mri/mef05.mgz -3 sample/mri/mef30.mgz
    $ mne flash_bem -s sample -n --registered -5 sample/mri/flash/mef05_*.mgz -3 sample/mri/flash/mef30_*.mgz

Notes
-----
This program assumes that FreeSurfer and MNE are installed and
sourced properly.

This function extracts the BEM surfaces (outer skull, inner skull, and
outer skin) from multiecho FLASH MRI data with spin angles of 5 and 30
degrees. The multiecho FLASH data can be input as .mgz or .nii files.
This function assumes that the Freesurfer segmentation of the subject
has been completed. In particular, the T1.mgz and brain.mgz MRI volumes
should be, as usual, in the subject's mri directory.

"""  # noqa E501

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne
from mne.bem import convert_flash_mris, make_flash_bem


def _vararg_callback(option, opt_str, value, parser):
    assert value is None
    del opt_str  # required for input but not used
    value = []

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        if arg[:1] == "-" and len(arg) > 1:
            break
        value.append(arg)

    del parser.rargs[: len(value)]
    setattr(parser.values, option.dest, value)


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option(
        "-s", "--subject", dest="subject", help="Subject name", default=None
    )
    parser.add_option(
        "-d",
        "--subjects-dir",
        dest="subjects_dir",
        help="Subjects directory",
        default=None,
    )
    parser.add_option(
        "-3",
        "--flash30",
        "--noflash30",
        dest="flash30",
        action="callback",
        callback=_vararg_callback,
        help=(
            "The 30-degree flip angle data. If no argument do "
            "not use flash30. If arguments are given, them as "
            "file names."
        ),
    )
    parser.add_option(
        "-5",
        "--flash5",
        dest="flash5",
        action="callback",
        callback=_vararg_callback,
        help=("Path to the multiecho flash 5 images. Can be one file or one per echo."),
    )
    parser.add_option(
        "-r",
        "--registered",
        dest="registered",
        action="store_true",
        default=False,
        help=(
            "Set if the Flash MRI images have already "
            "been registered with the T1.mgz file."
        ),
    )
    parser.add_option(
        "-u",
        "--unwarp",
        dest="unwarp",
        action="store_true",
        default=False,
        help=(
            "Run grad_unwarp with -unwarp <type> "
            "option on each of the converted data sets"
        ),
    )
    parser.add_option(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Write over existing .surf files in bem folder",
    )
    parser.add_option(
        "-v",
        "--view",
        dest="show",
        action="store_true",
        help="Show BEM model in 3D for visual inspection",
        default=False,
    )
    parser.add_option(
        "--copy",
        dest="copy",
        help="Use copies instead of symlinks for surfaces",
        action="store_true",
    )

    options, _ = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir
    flash5 = options.flash5
    if flash5 is None or len(flash5) == 0:
        flash5 = True
    flash30 = options.flash30
    if flash30 is None:
        flash30 = True
    elif len(flash30) == 0:
        flash30 = False
    register = not options.registered
    unwarp = options.unwarp
    overwrite = options.overwrite
    show = options.show
    copy = options.copy

    if options.subject is None:
        parser.print_help()
        raise RuntimeError("The subject argument must be set")

    flash5_img = convert_flash_mris(
        subject=subject,
        subjects_dir=subjects_dir,
        flash5=flash5,
        flash30=flash30,
        unwarp=unwarp,
        verbose=True,
    )
    make_flash_bem(
        subject=subject,
        subjects_dir=subjects_dir,
        overwrite=overwrite,
        show=show,
        copy=copy,
        register=register,
        flash5_img=flash5_img,
        verbose=True,
    )


mne.utils.run_command_if_main()
