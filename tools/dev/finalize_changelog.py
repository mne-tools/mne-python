"""Turn the towncrier fragments into the release changelog.

Run from the project root once the contributor credit action's PR is merged (it
is what gives new contributors the ``:newcontrib:`` entries this relies on):

    python tools/dev/finalize_changelog.py 1.13.0

This writes doc/changes/vX.Y.rst with the authors list appended and points the
what's new toctree at it. Anything it flags is a .mailmap gap: fix it, then
``git checkout doc/changes`` and run again. See the release checklist in the
wiki for the steps around it.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parents[2]
changes_dir = repo_root / "doc" / "changes"
sys.path.insert(0, str(repo_root / "doc" / "sphinxext"))
from credit_tools import BOTS  # noqa: E402, the credit page ignores the same ones


def _git(*args):
    """Run a git command in the repo and return its output."""
    return subprocess.check_output(("git",) + args, cwd=repo_root, text=True).strip()


def _mailmap_suggestion(email, linked):
    """Find the credited name for a commit address, if the credit data has one."""
    for fname in (repo_root / "doc" / "sphinxext" / "prs").glob("*.json"):
        for author in json.loads(fname.read_text("utf-8"))["authors"]:
            if author.get("e") == email and (author.get("n") or "").lower() in linked:
                return author["n"]
    return None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("version", help="version being released, e.g. 1.13.0")
parser.add_argument(
    "--force", action="store_true", help="build even if names need fixing"
)
args = parser.parse_args()
assert re.fullmatch(r"\d+\.\d+\.\d+", args.version), args.version
out_fname = changes_dir / f"v{'.'.join(args.version.split('.')[:2])}.rst"

# Everyone with a commit in the release, first-timers (who have a :newcontrib:
# entry, thanks to the credit action) marked with a trailing +
previous = _git("tag", "--list", "v*", "--sort=-v:refname").splitlines()[0]
authors = dict(  # name -> commit address
    line.split("\t", maxsplit=1)[1].rstrip(">").split(" <", maxsplit=1)
    for line in _git(
        "shortlog",
        "-se",
        "--group=author",
        "--group=trailer:co-authored-by",
        f"{previous}..HEAD",
    ).splitlines()
)
authors = {
    name: email
    for name, email in authors.items()
    if not any(bot in name or bot in email for bot in BOTS)
}
fragments = "\n".join(
    path.read_text("utf-8") for path in sorted((changes_dir / "dev").glob("*.rst"))
)
newcontribs = set(re.findall(r":newcontrib:`([^`]+)`", fragments))

# The changelog links every name through names.inc, so one that is missing is
# a .mailmap gap: git knows the commit identity, the credit page the GitHub one
linked = {
    name.lower()
    for name in re.findall(
        r"^\.\. _(.+?):", (changes_dir / "names.inc").read_text("utf-8"), re.M
    )
}
problems = 0
for name in sorted(set(authors) | newcontribs):
    if name.lower() in linked:
        continue
    problems += 1
    if name in newcontribs:
        print(f"{name!r} has a :newcontrib: entry but no commit under that name")
        continue
    proper = _mailmap_suggestion(authors[name], linked)
    print(f"{name!r} has no names.inc link, add to .mailmap:")
    print(f"    {proper or 'Their Name'} <{authors[name]}>")
if problems and not args.force:
    raise SystemExit(
        f"{problems} name(s) need fixing before the changelog is worth building; "
        "fix them (nothing has been written yet) or pass --force"
    )

subprocess.run(
    ["towncrier", "build", "--yes", "--version", args.version],
    cwd=repo_root,
    check=True,
)

# towncrier prepends the release to dev.rst, so its scaffolding (everything the
# template has above the names.inc include) is left behind below the entries
dev_fname = changes_dir / "dev.rst"
text = dev_fname.read_text("utf-8")
scaffolding = (changes_dir / "dev.rst.template").read_text("utf-8")
scaffolding = scaffolding.split(".. include::")[0]
assert scaffolding in text, "dev.rst does not look like it came from the template"
lines = [f"- {name}{'+' if name in newcontribs else ''}" for name in sorted(authors)]
head, sep, tail = text.replace(scaffolding, "").rpartition(".. include::")
text = f"{head}Authors\n-------\n\n" + "\n".join(lines) + f"\n\n{sep}{tail}"
dev_fname.write_text(text, "utf-8")

_git("mv", str(dev_fname), str(out_fname))
whats_new = repo_root / "doc" / "development" / "whats_new.rst"
text = whats_new.read_text("utf-8")
text = text.replace("   ../changes/dev.rst\n", f"   ../changes/{out_fname.name}\n")
whats_new.write_text(text, "utf-8")
print(
    f"Wrote {out_fname.relative_to(repo_root)} with {len(authors)} authors "
    f"({len(newcontribs)} new) since {previous}"
)
