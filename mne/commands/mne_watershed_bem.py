"""

    Create BEM surfaces using the watershed algorithm included with
        FreeSurfer

"""

from __future__ import print_function

from mne.bem import make_watershed_bem


def run():
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-s", "--subject", dest="subject",
                      help="Subject name", default=None)
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="Subjects directory", default=None)
    parser.add_option("-o", "--overwrite", dest="overwrite",
                      help="Write over existing files")
    parser.add_option("-v", "--volume", dest="volume",
                      help="Defaults to T1", default='T1')
    parser.add_option("-a", "--atlas", dest="atlas",
                      help="Specify the --atlas option for mri_watershed",
                      default=False)
    parser.add_option("-g", "--gcaatlas", dest="gcaatlas",
                      help="Use the subcortical atlas", default=False)
    parser.add_option("-p", "--preflood", dest="preflood",
                      help="Change the preflood height", default=None)
    parser.add_option("--verbose", dest="verbose",
                      help="If not None, override default verbose level",
                      default=None)

    options, args = parser.parse_args()

    subject = options.subject
    subjects_dir = options.subjects_dir
    overwrite = options.overwrite
    volume = options.volume
    atlas = options.atlas
    gcaatlas = options.gcaatlas
    preflood = options.preflood
    verbose = options.verbose

    make_watershed_bem(subject=subject, subjects_dir=subjects_dir,
                       overwrite=overwrite, volume=volume, atlas=atlas,
                       gcaatlas=gcaatlas, preflood=preflood, verbose=verbose)

is_main = (__name__ == '__main__')
if is_main:
    run()
