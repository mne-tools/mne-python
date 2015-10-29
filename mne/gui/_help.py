# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)
import json
import os


def format_tooltip(text):
    "Helper function to format tooltip help (insert line breaks)"
    lines = []
    words = text.split(' ')
    while words:
        line = []
        while words:
            if len(line) + sum(len(w) for w in line) + len(words[0]) > 50:
                break
            line.append(words.pop(0))
        lines.append(' '.join(line))
    return os.linesep.join(lines)


def read_tooltips(gui_name):
    "Read and format tooltips, return a dict"
    dirname = os.path.dirname(__file__)
    help_path = os.path.join(dirname, 'help', gui_name + '.json')
    with open(help_path) as fid:
        tt_raw = json.load(fid)
    return {key: format_tooltip(text) for key, text in tt_raw.iteritems()}
