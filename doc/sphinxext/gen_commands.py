# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import glob
from importlib import import_module
from pathlib import Path

from mne.utils import ArgvSetter, _replace_md5


def setup(app):
    app.connect("builder-inited", generate_commands_rst)


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


# Header markings go:
# 1. =/= : Page title
# 2. =   : Command name
# 3. -/- : Command description
# 4. -   : Command sections (Examples, Notes)

header = """\
:orphan:

.. _python_commands:

===============================
Command line tools using Python
===============================

"""

command_rst = """

.. _{0}:

{0}
{1}

.. rst-class:: callout

{2}

"""


def generate_commands_rst(app=None):
    try:
        from sphinx.util.display import status_iterator
    except Exception:
        from sphinx.util import status_iterator
    root = Path(__file__).parents[2]
    out_dir = root / "doc" / "generated"
    out_dir.mkdir(exist_ok=True)
    out_fname = out_dir / "commands.rst.new"

    command_path = root / "mne" / "commands"
    fnames = sorted(
        Path(fname).name for fname in glob.glob(str(command_path / "mne_*.py"))
    )
    assert len(fnames)
    iterator = status_iterator(
        fnames, "generating MNE command help ... ", length=len(fnames)
    )
    with open(out_fname, "w", encoding="utf8") as f:
        f.write(header)
        for fname in iterator:
            cmd_name = fname[:-3]
            module = import_module("." + cmd_name, "mne.commands")
            with ArgvSetter(("mne", cmd_name, "--help")) as out:
                try:
                    module.run()
                except SystemExit:  # this is how these terminate
                    pass
            output = out.stdout.getvalue().splitlines()

            # Swap usage and title lines
            output[0], output[2] = output[2], output[0]

            # Add header marking
            for idx in (1, 0):
                output.insert(idx, "-" * len(output[0]))

            # Add code styling for the "Usage: " line
            for li, line in enumerate(output):
                if line.startswith("Usage: mne "):
                    output[li] = f"Usage: ``{line[7:]}``"
                    break

            # Turn "Options:" into field list
            if "Options:" in output:
                ii = output.index("Options:")
                output[ii] = "Options"
                output.insert(ii + 1, "-------")
                output.insert(ii + 2, "")
                output.insert(ii + 3, ".. rst-class:: field-list cmd-list")
                output.insert(ii + 4, "")
            output = "\n".join(output)
            cmd_name_space = cmd_name.replace("mne_", "mne ")
            f.write(
                command_rst.format(cmd_name_space, "=" * len(cmd_name_space), output)
            )
    _replace_md5(str(out_fname))


# This is useful for testing/iterating to see what the result looks like
if __name__ == "__main__":
    generate_commands_rst()
