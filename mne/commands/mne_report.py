#!/usr/bin/env python
"""Create mne report for a folder

Example usage

mne report -p MNE-sample-data/ -i \
MNE-sample-data/MEG/sample/sample_audvis-ave.fif -d MNE-sample-data/subjects/ \
-s sample

"""

from mne.report import Reporter


if __name__ == '__main__':

    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__)

    parser.add_option("-p", "--path", dest="path",
                      help="Path to folder who MNE-Report must be created")
    parser.add_option("-i", "--info", dest="info_fname",
                      help="File from which info dictionary is to be read",
                      metavar="FILE")
    parser.add_option("-d", "--subjects-dir", dest="subjects_dir",
                      help="The subjects directory")
    parser.add_option("-s", "--subject", dest="subject",
                      help="The subject name")

    options, args = parser.parse_args()
    path = options.path
    info_fname = options.info_fname
    subjects_dir = options.subjects_dir
    subject = options.subject

    rep = Reporter(path, info_fname, subjects_dir, subject)
    rep.init_render()
    rep.render_folder()
    rep.finish_render()
