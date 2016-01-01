#!/usr/bin/env python

import codecs
from mne.externals.tempita import Template


def run():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--dir", dest="dir", help="Directory to parse")

    with codecs.open('conf_template.py', 'r', encoding='utf8') as f:
        conf_template = Template(f.read())

    options, args = parser.parse_args()
    directory = options.dir

    if directory is None:
        examples_dirs = "['../examples', '../tutorials']"
        gallery_dirs = "['auto_examples', 'auto_tutorials']"
    else:
        examples_dirs = "['../examples/%s']" % directory
        gallery_dirs = "['auto_examples/%s']" % directory

    output = conf_template.substitute(examples_dirs=examples_dirs,
                                      gallery_dirs=gallery_dirs)

    with codecs.open('conf.py', 'w', encoding='utf8') as f:
        f.write(output)


is_main = (__name__ == '__main__')
if is_main:
    run()
