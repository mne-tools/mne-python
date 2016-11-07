#!/usr/bin/env python
r"""Convert surface to BEM FIF file.

Example usage

mne surf2bem --surf ${SUBJECTS_DIR}/${SUBJECT}/surf/lh.seghead --fif \
${SUBJECTS_DIR}/${SUBJECT}/bem/${SUBJECT}-head.fif --id=4
"""
from __future__ import print_function
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

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
                      help=("Surface Id (e.g. 4 sur head surface)"))

    options, args = parser.parse_args()

    if options.surf is None:
        parser.print_help()
        sys.exit(1)

    print("Converting %s to BEM FIF file." % options.surf)
    surf = mne.bem._surfaces_to_bem([options.surf], [int(options.id)],
                                    sigma=[1])
    mne.write_bem_surfaces(options.fif, surf)


is_main = (__name__ == '__main__')
if is_main:
    run()
