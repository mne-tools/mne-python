#!/usr/bin/env python
# Authors: Lorenzo De Santis
"""Create BEM surfaces using the watershed algorithm included with FreeSurfer.

Examples
--------
.. code-block:: console

    $ mne watershed_bem -s sample

"""

import sys

from mne.bem import make_watershed_bem


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-s", "--subject", dest="subject",
                      help="Subject name (required)", default=None)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=None)
    parser.add_option("-o", "--overwrite", dest="overwrite",
                      help="Write over existing files", action="store_true")
    parser.add_option("-v", "--volume", dest="volume",
                      help="Defaults to T1", default='T1')
    parser.add_option("-a", "--atlas", dest="atlas",
                      help="Specify the --atlas option for mri_watershed",
                      default=False, action="store_true")
    parser.add_option("-g", "--gcaatlas", dest="gcaatlas",
                      help="Use the subcortical atlas", default=False,
                      action="store_true")
    parser.add_option("-p", "--preflood", dest="preflood",
                      help="Change the preflood height", default=None)
    parser.add_option("--copy", dest="copy",
                      help="Use copies instead of symlinks for surfaces",
                      action="store_true")
    parser.add_option("--verbose", dest="verbose",
                      help="If not None, override default verbose level",
                      default=None)

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
    verbose = options.verbose

    make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                       overwrite=overwrite, volume=volume, atlas=atlas,
                       gcaatlas=gcaatlas, preflood=preflood, copy=copy,
                       verbose=verbose)

is_main = (__name__ == '__main__')
if is_main:
    run()
