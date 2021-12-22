#!/usr/bin/env python
r"""Convert surface to BEM FIF file.

Examples
--------
.. code-block:: console

    $ mne surf2bem --surf ${SUBJECTS_DIR}/${SUBJECT}/surf/lh.seghead \
        --fif ${SUBJECTS_DIR}/${SUBJECT}/bem/${SUBJECT}-head.fif \
        --id=4

"""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause

import sys

import mne


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-s", "--surf", dest="surf",
                      help="Surface in Freesurfer format", metavar="FILE")
    parser.add_option("-f", "--fif", dest="fif",
                      help="FIF file produced", metavar="FILE")
    parser.add_option("-i", "--id", dest="id", default=4,
                      help=("Surface Id (e.g. 4 for head surface)"))

    options, args = parser.parse_args()

    if options.surf is None:
        parser.print_help()
        sys.exit(1)

    print("Converting %s to BEM FIF file." % options.surf)
    surf = mne.bem._surfaces_to_bem([options.surf], [int(options.id)],
                                    sigmas=[1])
    mne.write_bem_surfaces(options.fif, surf)


mne.utils.run_command_if_main()
