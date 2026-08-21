"""Check that changelog entries link to names defined in names.inc.

Changelog fragments in doc/changes/dev/ credit people with reStructuredText
link references, either the plain form used for existing contributors or the
:newcontrib: role used for first-time ones. Both only resolve if
doc/changes/names.inc defines a matching link target; a missing target
otherwise fails the (much slower) documentation build, so check it here.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import re
import sys
from collections import defaultdict
from pathlib import Path

changes_dir = Path(__file__).parents[1] / "doc" / "changes"
names_inc = changes_dir / "names.inc"

# ".. _Jane Doe: https://github.com/janedoe"
anchor_re = re.compile(r"^\.\. _(.+?):\s+(\S+)\s*$", re.MULTILINE)
# "`Jane Doe`_" but not the inline-link form "`text <https://...>`_"
reference_re = re.compile(r"`([^`<>]+?)`_")
# ":newcontrib:`Jane Doe`"
newcontrib_re = re.compile(r":newcontrib:`([^`]+)`")
# stand-in names used to show the changelog syntax, not real credits
PLACEHOLDERS = {"firstname lastname", "new contributor"}


def _reference_key(name):
    """Normalize a name the way docutils normalizes reST reference names."""
    return re.sub(r"\s+", " ", name).strip().lower()


def main():
    """Check changelog fragments against names.inc."""
    problems = []

    # Link targets that repeat a name must agree on the URL, otherwise docutils
    # emits a warning and the documentation build (run with -W) fails
    urls = defaultdict(set)
    display = dict()
    for name, url in anchor_re.findall(names_inc.read_text("utf-8")):
        urls[_reference_key(name)].add(url)
        display.setdefault(_reference_key(name), name)
    for key, these_urls in sorted(urls.items()):
        if len(these_urls) > 1:
            problems.append(
                f"{names_inc.name} defines {display[key]!r} more than once with "
                f"different links ({', '.join(sorted(these_urls))}). Keep one "
                "link target per person, or point every spelling of the name at "
                "the same link."
            )

    # Every name credited in a changelog needs a link target. Released
    # changelogs are checked too: renaming or removing a target breaks them
    # just as easily, and only the (much slower) documentation build would say
    # so otherwise.
    # Changelogs also link to projects and tools, whose targets live in
    # doc/links.inc or in the file itself rather than in names.inc
    known = set(urls)
    links_inc = changes_dir.parent / "links.inc"
    if links_inc.is_file():
        known |= {
            _reference_key(name)
            for name, _ in anchor_re.findall(links_inc.read_text("utf-8"))
        }
    missing = defaultdict(list)
    for fname in sorted(changes_dir.rglob("*.rst")):
        text = fname.read_text("utf-8")
        here = known | {_reference_key(name) for name, _ in anchor_re.findall(text)}
        for match in list(reference_re.finditer(text)) + list(
            newcontrib_re.finditer(text)
        ):
            name = match.group(1)
            if _reference_key(name) not in here | PLACEHOLDERS:
                missing[name].append(str(fname.relative_to(changes_dir.parent.parent)))
    for name, fnames in sorted(missing.items()):
        where = ", ".join(sorted(set(fnames)))
        problems.append(
            f"{where} credits {name!r}, but doc/changes/names.inc has no link "
            f"target for that name. Add one (the file is sorted alphabetically, "
            f"ignoring case):\n\n    .. _{name}: https://github.com/<username>"
        )

    if problems:
        print(
            f"Found {len(problems)} changelog name problem(s):\n\n"
            + "\n\n".join(problems),
            file=sys.stderr,
        )
        return 1
    print("All changelog entries link to names defined in names.inc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
