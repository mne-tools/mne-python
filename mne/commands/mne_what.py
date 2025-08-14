r"""Check type of FIF file.

Examples
--------
.. code-block:: console

    $ mne what sample_audvis_raw.fif
    raw
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import mne


def run():
    """Run command."""
    from mne.commands.utils import get_optparser

    parser = get_optparser(__file__, usage="usage: %prog fname [fname2 ...]")
    options, args = parser.parse_args()
    for arg in args:
        print(mne.what(arg))


mne.utils.run_command_if_main()
