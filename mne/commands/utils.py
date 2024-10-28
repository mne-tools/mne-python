"""Some utility functions for commands (e.g., for cmdline handling)."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import glob
import importlib
import os
import os.path as op
import sys
from optparse import OptionParser

import mne


def _add_verbose_flag(parser):
    parser.add_option(
        "--verbose",
        dest="verbose",
        help="Enable verbose mode (printing of log messages).",
        default=None,
        action="store_true",
    )


def load_module(name, path):
    """Load module from .py/.pyc file.

    Parameters
    ----------
    name : str
        Name of the module.
    path : str
        Path to .py/.pyc file.

    Returns
    -------
    mod : module
        Imported module.

    """
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_optparser(cmdpath, usage=None, prog_prefix="mne", version=None):
    """Create OptionParser with cmd specific settings (e.g., prog value)."""
    # Fetch description
    mod = load_module("__temp", cmdpath)
    if mod.__doc__:
        doc, description, epilog = mod.__doc__, None, None

        doc_lines = doc.split("\n")
        description = doc_lines[0]
        if len(doc_lines) > 1:
            epilog = "\n".join(doc_lines[1:])

    # Get the name of the command
    command = os.path.basename(cmdpath)
    command, _ = os.path.splitext(command)
    command = command[len(prog_prefix) + 1 :]  # +1 is for `_` character

    # Set prog
    prog = prog_prefix + f" {command}"

    # Set version
    if version is None:
        version = mne.__version__

    # monkey patch OptionParser to not wrap epilog
    OptionParser.format_epilog = lambda self, formatter: self.epilog
    parser = OptionParser(
        prog=prog, version=version, description=description, epilog=epilog, usage=usage
    )

    return parser


def main():
    """Entrypoint for mne <command> usage."""
    mne_bin_dir = op.dirname(op.dirname(__file__))
    valid_commands = sorted(glob.glob(op.join(mne_bin_dir, "commands", "mne_*.py")))
    valid_commands = [c.split(op.sep)[-1][4:-3] for c in valid_commands]

    def print_help():  # noqa
        print("Usage : mne command options\n")
        print("Accepted commands :\n")
        for c in valid_commands:
            print(f"\t- {c}")
        print("\nExample : mne browse_raw --raw sample_audvis_raw.fif")
        print("\nGetting help example : mne compute_proj_eog -h")

    if len(sys.argv) == 1 or "help" in sys.argv[1] or "-h" in sys.argv[1]:
        print_help()
    elif sys.argv[1] == "--version":
        print(f"MNE {mne.__version__}")
    elif sys.argv[1] not in valid_commands:
        print(f'Invalid command: "{sys.argv[1]}"\n')
        print_help()
    else:
        cmd = sys.argv[1]
        cmd = importlib.import_module(f".mne_{cmd}", "mne.commands")
        sys.argv = sys.argv[1:]
        cmd.run()
