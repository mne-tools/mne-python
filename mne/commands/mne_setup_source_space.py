#!/usr/bin/env python
'''Set up bilateral hemisphere surface-based source space with subsampling
 using :func:`mne.setup_source_space`

Examples
--------
.. code-block:: console

    $ mne setup_source_space


 .. note : Both --ico and --spacing options can't be present at the same time.

'''

import sys

import mne
from mne.utils import _check_option


def run():
    '''Run command.'''
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option('-s', '--subject',
                      dest='subject',
                      help='Subject name (required)',
                      default=None)
    parser.add_option('--morph',
                      dest='subject_to',
                      help='Name of a subject in SUBJECTS_DIR. If this option '
                           'is present, the source space will be first '
                           'constructed for the subject defined by the '
                           ' –subject option or the SUBJECT environment '
                           'variable and then morphed to this subject. '
                           ' This option is useful if you want to create a '
                           'source spaces for several subjects and want to '
                           'directly compare the data across subjects at the '
                           'source space vertices without any morphing '
                           'procedure afterwards. The drawback of this '
                           'approach is that the spacing between source '
                           'locations in the "morph" subject is not going to '
                           'be as uniform as it would be without morphing.',
                      default=None)
    parser.add_option('--surf',
                      dest='surface',
                      help='The surface to use.',
                      default='white',
                      type='string')
    parser.add_option('--spacing',
                      dest='spacing',
                      help='Specifies the approximate grid spacing of the '
                           'source space in mm.',
                      default=None,
                      type='int')
    parser.add_option('--ico',
                      dest='ico',
                      help='Instead of using the traditional method for '
                           'cortical surface decimation it is possible to '
                           'create the source space using the topology of a '
                           'recursively subdivided icosahedron ( <number> > 0)'
                           ' or an octahedron ( <number> < 0). This method '
                           'uses the cortical surface inflated to a sphere as '
                           'a tool to find the appropriate vertices for the '
                           'source space.',
                      default=None,
                      type='int')
    parser.add_option('--src', dest='fname',
                      help='Output file name. Use a name <dir>/<name>-src.fif',
                      metavar='FILE', default=None)
    parser.add_option('-d', '--subjects-dir',
                      dest='subjects_dir',
                      help='Subjects directory',
                      default=None)
    parser.add_option('-a', '--add-dist',
                      dest='add_dist',
                      help='Add distance and patch information '
                           'to the source space.',
                      default=True)
    parser.add_option('-n', '--n-jobs',
                      dest='n_jobs',
                      help='The number of jobs to run in parallel '
                            '(default 1). Requires the joblib package. '
                            'Will use at most 2 jobs'
                            '(one for each hemisphere).',
                      default=1,
                      type='int')
    parser.add_option('--verbose',
                      dest='verbose',
                      help='Turn on verbose mode.',
                      default=None)
    parser.add_option('-o', '--overwrite',
                      dest='overwrite',
                      help='If True, the destination file (if it exists)'
                            ' will be overwritten. If False (default),'
                            ' an error will be raised if the file exists.',
                      default=False)

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
    surface = options.surface
    add_dist = options.add_dist
    n_jobs = options.n_jobs
    verbose = options.verbose
    overwrite = options.overwrite

    # Parse source spacing option
    if ico is not None and spacing is not None:
        raise ValueError('Both --ico and --spacing options can not be present '
                         'at the same time')
    elif ico is not None and spacing is None:
        if ico < 0:
            use_spacing = "ico" + str(ico)
        elif ico < 0:
            use_spacing = "oct" + str(-ico)
        elif ico == 0:
            raise ValueError('Ico can not be 0')
    elif ico is None and spacing is not None:
        use_spacing = spacing
    else:
        raise ValueError('Can not set source position, please check that at '
                         'least of --spacing or --ico parameter is set')

    if not (fname.endswith('_src.fif') or fname.endswith('-src.fif')):
        fname = fname + "-src.fif"

    # Create source space
    src = mne.setup_source_space(subject=subject, spacing=use_spacing,
                                 surface=surface, subjects_dir=subjects_dir,
                                 add_dist=add_dist, n_jobs=n_jobs,
                                 verbose=verbose)
    # Morph source space if --morph is set
    if subject_to is not None:
        src_morph = mne.morph_source_spaces(src, subject_to=subject_to,
                                            subjects_dir=subjects_dir)
    else:
        src_morph = src
    # Save source space to file
    src_morph.save(fname=fname, overwrite=overwrite)

mne.utils.run_command_if_main()
