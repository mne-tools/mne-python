from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from mne.defaults import DEFAULTS
from mne.io.pick import (_PICK_TYPES_DATA_DICT, _DATA_CH_TYPES_SPLIT,
                         _DATA_CH_TYPES_ORDER_DEFAULT)


class MNESubstitution(Directive):  # noqa: D101

    has_content = False
    required_arguments = 1
    final_argument_whitespace = True

    def run(self, **kwargs):  # noqa: D102
        env = self.state.document.settings.env
        if self.arguments[0] == 'data channels list':
            keys = list()
            for key in _DATA_CH_TYPES_ORDER_DEFAULT:
                if key in _DATA_CH_TYPES_SPLIT:
                    keys.append(key)
                elif key not in ('meg', 'fnirs') and \
                        _PICK_TYPES_DATA_DICT.get(key, False):
                    keys.append(key)
            rst = '- ' + '\n- '.join(
                '``%r``: **%s** (scaled by %g to plot in *%s*)'
                % (key, DEFAULTS['titles'][key], DEFAULTS['scalings'][key],
                   DEFAULTS['units'][key])
                for key in keys)
        else:
            raise self.error(
                'MNE directive unknown in %s: %r'
                % (env.doc2path(env.docname, base=None),
                   self.arguments[0],))
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
