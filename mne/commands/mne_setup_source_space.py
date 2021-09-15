#!/usr/bin/env python
"""Set up bilateral hemisphere surface-based source space with subsampling.

Examples
--------
.. code-block:: console

    $ mne setup_source_space --subject sample


 .. note : Only one of --ico, --oct or --spacing options can be set at the same
           time. Default to oct6.

"""

import sys

import mne
from mne.utils import _check_option


def run():
    """Run command."""
    from mne.commands.utils import get_optparser, _add_verbose_flag
    parser = get_optparser(__file__)

    parser.add_option('-s', '--subject',
                      dest='subject',
                      help='Subject name (required)',
                      default=None)
    parser.add_option('--src', dest='fname',
                      help='Output file name. Use a name <dir>/<name>-src.fif',
                      metavar='FILE', default=None)
    parser.add_option('--morph',
                      dest='subject_to',
                      help='morph the source space to this subject',
                      default=None)
    parser.add_option('--surf',
                      dest='surface',
                      help='The surface to use. (default to white)',
                      default='white',
                      type='string')
    parser.add_option('--spacing',
                      dest='spacing',
                      help='Specifies the approximate grid spacing of the '
                           'source space in mm. (default to 7mm)',
                      default=None,
                      type='int')
    parser.add_option('--ico',
                      dest='ico',
                      help='use the recursively subdivided icosahedron '
                           'to create the source space.',
                      default=None,
                      type='int')
    parser.add_option('--oct',
                      dest='oct',
                      help='use the recursively subdivided octahedron '
                           'to create the source space.',
                      default=None,
                      type='int')
    parser.add_option('-d', '--subjects-dir',
                      dest='subjects_dir',
                      help='Subjects directory',
                      default=None)
    parser.add_option('-n', '--n-jobs',
                      dest='n_jobs',
                      help='The number of jobs to run in parallel '
                            '(default 1). Requires the joblib package. '
                            'Will use at most 2 jobs'
                            ' (one for each hemisphere).',
                      default=1,
                      type='int')
    parser.add_option('--add-dist',
                      dest='add_dist',
                      help='Add distances. Can be "True", "False", or "patch" '
                      'to only compute cortical patch statistics (like the '
                      '--cps option in MNE-C; requires SciPy >= 1.3)',
                      default='True')
    parser.add_option('-o', '--overwrite',
                      dest='overwrite',
                      help='to write over existing files',
                      default=None, action="store_true")
    _add_verbose_flag(parser)

    options, args = parser.parse_args()

    if options.subject is None:
        parser.print_help()
        sys.exit(1)

    subject = options.subject
    subject_to = options.subject_to
    fname = options.fname
    subjects_dir = options.subjects_dir
    spacing = options.spacing
    ico = options.ico
    oct = options.oct
    surface = options.surface
    n_jobs = options.n_jobs
    add_dist = options.add_dist
    _check_option('add_dist', add_dist, ('True', 'False', 'patch'))
    add_dist = {'True': True, 'False': False, 'patch': 'patch'}[add_dist]
    verbose = True if options.verbose is not None else False
    overwrite = True if options.overwrite is not None else False

    # Parse source spacing option
    spacing_options = [ico, oct, spacing]
    n_options = len([x for x in spacing_options if x is not None])
    if n_options > 1:
        raise ValueError('Only one spacing option can be set at the same time')
    elif n_options == 0:
        # Default to oct6
        use_spacing = 'oct6'
    elif n_options == 1:
        if ico is not None:
            use_spacing = "ico" + str(ico)
        elif oct is not None:
            use_spacing = "oct" + str(oct)
        elif spacing is not None:
            use_spacing = spacing
    # Generate filename
    if fname is None:
        if subject_to is None:
            fname = subject + '-' + str(use_spacing) + '-src.fif'
        else:
            fname = (subject_to + '-' + subject + '-' +
                     str(use_spacing) + '-src.fif')
    else:
        if not (fname.endswith('_src.fif') or fname.endswith('-src.fif')):
            fname = fname + "-src.fif"
    # Create source space
    src = mne.setup_source_space(subject=subject, spacing=use_spacing,
                                 surface=surface, subjects_dir=subjects_dir,
                                 n_jobs=n_jobs, add_dist=add_dist,
                                 verbose=verbose)
    # Morph source space if --morph is set
    if subject_to is not None:
        src = mne.morph_source_spaces(src, subject_to=subject_to,
                                      subjects_dir=subjects_dir,
                                      surf=surface, verbose=verbose)

    # Save source space to file
    src.save(fname=fname, overwrite=overwrite)


mne.utils.run_command_if_main()
