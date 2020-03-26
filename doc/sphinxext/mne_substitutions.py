from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT


class MNESubstitution(Directive):  # noqa: D101

    has_content = True

    def run(self, **kwargs):  # noqa: D102
        if len(self.content) != 1:
            raise ValueError('MNE directive should have a single argument, '
                             'got: %s' % (self.content,))
        if self.content[0] == 'data channels list':
            rst = ('- ' + '\n- '.join(
                '``%r``' % x for x in _DATA_CH_TYPES_ORDER_DEFAULT))
        else:
            raise ValueError('MNE directive unknown: %r' % (self.content[0],))
        node = nodes.compound(rst)  # General(Body), Element
        content = StringList(
            rst.split('\n'), parent=self.content.parent,
            parent_offset=self.content.parent_offset)
        self.state.nested_parse(content, self.content_offset, node)
        return [node]


def setup(app):  # noqa: D103
    app.add_directive('mne', MNESubstitution)
    return {'version': '0.1',
            'parallel_read_safe': True,
            'parallel_write_safe': True}
