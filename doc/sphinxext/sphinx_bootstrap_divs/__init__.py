"""A .. collapse:: directive for sphinx-bootstrap-theme."""

import os.path as op
from docutils import nodes
from docutils.parsers.rst.directives import flag, class_option
from docutils.parsers.rst.roles import set_classes
from docutils.statemachine import StringList

from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.fileutil import copy_asset

this_dir = op.dirname(__file__)

__version__ = '0.1.0.dev0'


###############################################################################
# Super classes

class DivNode(nodes.Body, nodes.Element):
    """Generic DivNode class."""

    def __init__(self, **options):
        diff = set(options.keys()).symmetric_difference(set(self.OPTION_KEYS))
        assert len(diff) == 0, (diff, self.__class__.__name__)
        self.options = options
        super().__init__()

    def visit_node(self, node):
        """Visit the node."""
        atts = {}
        if node.BASECLASS:
            atts['class'] = node.BASECLASS
            if node.options.get('class'):
                atts['class'] += \
                    ' {}-{}'.format(node.BASECLASS, node.options['class'])
        self.body.append(self.starttag(node, node.ELEMENT, **atts))

    def depart_node(self, node):
        """Depart the node."""
        self.body.append('</{}>'.format(node.ELEMENT))


def _assemble(node, directive):
    title_text = directive.arguments[0]
    directive.add_name(node)
    header = node.HEADER_PRETITLE.format(**node.options).split('\n')
    directive.state.nested_parse(
        StringList(header), directive.content_offset, node)

    textnodes, messages = directive.state.inline_text(
        title_text, directive.lineno)
    node += textnodes
    node += messages

    header = node.HEADER_POSTTITLE.format(**node.options).split('\n')
    directive.state.nested_parse(
        StringList(header), directive.content_offset, node)

    directive.state.nested_parse(
        directive.content, directive.content_offset, node)

    footer = node.FOOTER.format(**node.options).split('\n')
    directive.state.nested_parse(
        StringList(footer), directive.content_offset, node)


###############################################################################
# .. collapse::

class CollapseNode(DivNode):
    """Class for .. collapse:: directive."""

    OPTION_KEYS = ('title', 'id_', 'extra', 'class')
    ELEMENT = 'div'
    BASECLASS = 'card'
    HEADER_PRETITLE = """.. raw:: html

    <h5 class="card-header">
    <a data-toggle="collapse" href="#collapse_{id_}" role="button" aria-expanded="false" aria-controls="collapse_{id_}">"""
    HEADER_POSTTITLE = """.. raw:: html

    </a></h5>
    <div id="collapse_{id_}" class="collapse{extra}">
    <div class="card card-body">"""
    FOOTER = """.. raw:: html

    </div></div>"""
    KNOWN_CLASSES = (
        'default', 'primary', 'success', 'info', 'warning', 'danger')

    @staticmethod
    def _check_class(class_):
        if class_ not in CollapseNode.KNOWN_CLASSES:
            raise ValueError(':class: option %r must be one of %s'
                             % (class_, CollapseNode.KNOWN_CLASSES))
        return class_


class CollapseDirective(SphinxDirective):
    """Collapse directive."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'open': flag,
                   'class': CollapseNode._check_class}
    has_content = True

    def run(self):
        """Parse."""
        self.assert_has_content()
        title_text = _(self.arguments[0])
        extra = _(' show' if 'open' in self.options else '')
        class_ = {'class': self.options.get('class', 'default')}
        id_ = nodes.make_id(title_text)
        node = CollapseNode(title=title_text, id_=id_, extra=extra, **class_)
        _assemble(node, self)
        return [node]


###############################################################################
# .. details::

class DetailsNode(DivNode):
    """Class for .. details:: directive."""

    ELEMENT = 'details'
    BASECLASS = ''
    OPTION_KEYS = ('title', 'class')
    HEADER_PRETITLE = """.. raw:: html

    <summary>"""
    HEADER_POSTTITLE = """.. raw:: html

    </summary>"""
    FOOTER = """"""


class DetailsDirective(SphinxDirective):
    """Details directive."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'class': class_option}
    has_content = True

    def run(self):
        """Parse."""
        set_classes(self.options)
        self.assert_has_content()
        title_text = _(self.arguments[0])
        class_ = {'class': self.options.get('class', '')}
        node = DetailsNode(title=title_text, **class_)
        _assemble(node, self)
        return [node]


###############################################################################
# Generic setup

def setup(app):
    """Set up for Sphinx app."""
    directives = dict(
        collapse=CollapseDirective,
        details=DetailsDirective,
    )
    for key, value in directives.items():
        app.add_directive(key, value)
    try:
        app.add_css_file('bootstrap_divs.css')
    except AttributeError:
        app.add_stylesheet('bootstrap_divs.css')
    try:
        app.add_js_file('bootstrap_divs.js')
    except AttributeError:
        app.add_javascript('bootstrap_divs.js')
    app.connect('build-finished', copy_asset_files)
    for node in (CollapseNode, DetailsNode):
        app.add_node(node,
                     html=(node.visit_node, node.depart_node),
                     latex=(node.visit_node, node.depart_node),
                     text=(node.visit_node, node.depart_node))
    return dict(version='0.1', parallel_read_safe=True,
                parallel_write_safe=True)


def copy_asset_files(app, exc):
    """Copy static assets."""
    asset_files = ['bootstrap_divs.css', 'bootstrap_divs.js']
    if exc is None:  # build succeeded
        for path in asset_files:
            copy_asset(op.join(this_dir, path),
                       op.join(app.outdir, '_static'))
