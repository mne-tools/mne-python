"""A .. collapse:: directive for sphinx-bootstrap-theme."""

import os.path as op
from docutils import nodes
from docutils.parsers.rst.directives import flag

from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.fileutil import copy_asset

this_dir = op.dirname(__file__)

__version__ = '0.1.0.dev0'


###############################################################################
# Super classes

class DivNode(nodes.General, nodes.Element):
    """Generic DivNode class."""

    def __init__(self, **options):
        diff = set(options.keys()).symmetric_difference(set(self.OPTION_KEYS))
        assert len(diff) == 0, (diff, self.__class__.__name__)
        self.options = options
        super().__init__()

    @staticmethod
    def visit_node(self, node):
        """Visit the node."""
        self.body.append(node.HEADER.format(**node.options))

    @staticmethod
    def depart_node(self, node):
        """Depart the node."""
        self.body.append(node.FOOTER)


###############################################################################
# .. collapse::

class CollapseNode(DivNode):
    """Class for .. collapse:: directive."""

    OPTION_KEYS = ('title', 'id_', 'extra', 'class_')
    HEADER = """
    <div class="panel panel-{class_}">
    <div class="panel-heading"><h4 class="panel-title">
    <a data-toggle="collapse" href="#collapse_{id_}">{title}</a>
    </h4></div>
    <div id="collapse_{id_}" class="panel-collapse collapse{extra}">
    <div class="panel-body">
    """
    FOOTER = "</div></div></div>"
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
        env = self.state.document.settings.env
        self.assert_has_content()
        extra = _(' in' if 'open' in self.options else '')
        title = _(self.arguments[0])
        class_ = self.options.get('class', 'default')
        id_ = env.new_serialno('Collapse')
        collapse_node = CollapseNode(title=title, id_=id_, extra=extra,
                                     class_=class_)
        self.add_name(collapse_node)
        self.state.nested_parse(
            self.content, self.content_offset, collapse_node)
        return [collapse_node]


###############################################################################
# .. details::

class DetailsNode(DivNode):
    """Class for .. details:: directive."""

    OPTION_KEYS = ('title', 'class_')
    HEADER = """
    <details class="{class_}"><summary>{title}</summary>"""
    FOOTER = "</details>"


class DetailsDirective(SphinxDirective):
    """Details directive."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'class': str}
    has_content = True

    def run(self):
        """Parse."""
        self.assert_has_content()
        title = _(self.arguments[0])
        class_ = self.options.get('class', '')
        details_node = DetailsNode(title=title, class_=class_)
        self.add_name(details_node)
        self.state.nested_parse(
            self.content, self.content_offset, details_node)
        return [details_node]


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
