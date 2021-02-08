#!/usr/bin/env python
# Authors: Lorenzo De Santis
#          Eric Larson <larson.eric.d@gmail.com>

"""Create BEM surfaces using FSL.

Examples
--------
.. code-block:: console

    $ mne fsl_bem -s sample

"""

import sys

import mne
from mne.bem import make_fsl_bem


def run():
    """Run command."""
    from mne.commands.utils import get_optparser, _add_verbose_flag

    parser = get_optparser(__file__)

    parser.add_option("-s", "--subject", dest="subject",
                      help="Subject name (required)", default=None)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=None)
    parser.add_option("-o", "--overwrite", dest="overwrite",
                      help="Write over existing files", action="store_true")
    parser.add_option("-v", "--volume", dest="volume",
                      help="Defaults to T1", default='T1')
    parser.add_option("-f", "--fraction", dest="fraction", default=0.5,
                      help="Write over existing files", type="float")
    parser.add_option("-b", "--brainmask", dest="brainmask",
                      help="Brainmask image to use instead of bet",
                      default=None)
    parser.add_option("-t", "--talairach", dest='talairach',
                      help="Use FreeSurfer's Talairach transform rather than "
                      "FLIRT", action="store_true")
    _add_verbose_flag(parser)

    options, args = parser.parse_args()

    if options.subject is None:
        parser.print_help()
        sys.exit(1)

    subject = options.subject
    subjects_dir = options.subjects_dir
    overwrite = options.overwrite
    volume = options.volume
    fraction = options.fraction
    brainmask = options.brainmask
    talairach = options.talairach
    verbose = options.verbose

    make_fsl_bem(subject=subject, subjects_dir=subjects_dir,
                 overwrite=overwrite, volume=volume, fraction=fraction,
                 brainmask=brainmask, talairach=talairach,
                 verbose=verbose)


mne.utils.run_command_if_main()
