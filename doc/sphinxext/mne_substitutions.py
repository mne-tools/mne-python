# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from mne._fiff.pick import (
    _DATA_CH_TYPES_ORDER_DEFAULT,
    _DATA_CH_TYPES_SPLIT,
    _EYETRACK_CH_TYPES_SPLIT,
    _PICK_TYPES_DATA_DICT,
)
from mne.defaults import DEFAULTS


class MNESubstitution(Directive):  # noqa: D101
    has_content = False
    required_arguments = 1
    final_argument_whitespace = True

    def run(self, **kwargs):  # noqa: D102
        env = self.state.document.settings.env
        if self.arguments[0] == "data channels list":
            keys = list()
            for key in _DATA_CH_TYPES_ORDER_DEFAULT:
                if key in _DATA_CH_TYPES_SPLIT:
                    keys.append(key)
                elif key not in ("meg", "fnirs") and _PICK_TYPES_DATA_DICT.get(
                    key, False
                ):
                    keys.append(key)
            rst = "- " + "\n- ".join(
                f"``{repr(key)}``: **{DEFAULTS['titles'][key]}** "
                f"(scaled by {DEFAULTS['scalings'][key]} to "
                f"plot in *{DEFAULTS['units'][key]}*)"
                for key in keys
            )
        elif self.arguments[0] == "non-data channels list":
            keys = list()
            for key in _DATA_CH_TYPES_ORDER_DEFAULT:
                if (
                    _PICK_TYPES_DATA_DICT.get(key, False)
                    or key in _EYETRACK_CH_TYPES_SPLIT or
                    key == 'ref_meg'
                ):
                    keys.append(key)
            rst = "- " + "\n- ".join(
                f"``{repr(key)}``: **{DEFAULTS['titles'][key]}** "
                f"(scaled by {DEFAULTS['scalings'][key]} to "
                f"plot in *{DEFAULTS['units'][key]}*)"
                for key in keys
            )
        else:
            raise self.error(
                "MNE directive unknown in %s: %r"  # noqa: UP031
                % (
                    env.doc2path(env.docname, base=None),
                    self.arguments[0],
                )
            )
        node = nodes.compound(rst)  # General(Body), Element
        content = StringList(
            rst.split("\n"),
            parent=self.content.parent,
            parent_offset=self.content.parent_offset,
        )
        self.state.nested_parse(content, self.content_offset, node)
        return [node]


def setup(app):  # noqa: D103
    app.add_directive("mne", MNESubstitution)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
