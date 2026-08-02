"""Create code credit RST file.

Run ./tools/dev/update_credit_json.py first to get the latest PR JSON files.

Contributor names are resolved from the PR JSON files in this order:

1. ``.mailmap`` (authoritative): the author email is looked up and the mailmap
   name wins. Entries whose name intentionally fails our heuristics (single
   word, abbreviation, ALL-CAPS) carry a trailing ``# credit: name-ok``
   comment (valid mailmap syntax, and robust to the line-sorting pre-commit
   hook).
2. GitHub-derived names stored in the JSON files, when they look like real
   two-part names.

Anything else is reported as a ready-to-paste ``.mailmap`` suggestion, or
appended to ``.mailmap`` automatically when running with ``--fix-mailmap``
(what the monthly credit GitHub Action does).
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import argparse
import dataclasses
import fnmatch
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import sphinx.util.logging

import mne
from mne.utils import logger, verbose

sphinx_logger = sphinx.util.logging.getLogger("mne")

repo_root = Path(__file__).parents[2]
doc_root = repo_root / "doc"
data_dir = doc_root / "sphinxext"

PR_URL = "https://github.com/mne-tools/mne-python/pull/{pr}"
NAME_OK_MARKER = "# credit: name-ok"
# Substrings (in name or email) identifying non-human authors we never credit
BOTS = (
    "[bot]",
    "mne-bot@users.noreply",
    "pre-commit-ci@users.noreply",
    "copilot@github.com",
    "noreply@anthropic.com",
    "Lumberbot",
    "Deleted user",
)
# Surnames where we know we have more than one distinct contributor; a surname
# appearing more often than this makes the monthly report warn about possible
# duplicates (e.g., the same person under two name variants).
NAME_COUNTS = dict(
    Bailey=2,
    Das=2,
    Drew=2,
    Jin=2,
    Li=2,
    Peterson=2,
    Wong=2,
    Yadav=2,
    Zhang=3,
)


def _is_bot(name, email):
    return any(bot in (name or "") or bot in (email or "") for bot in BOTS)


def _good_name(name, *, name_ok=frozenset()):
    """Heuristically decide if a name looks like a real, full human name."""
    if name is None or not name.strip() or name != name.strip():
        return False
    if name in name_ok:
        return True
    if " " not in name:  # at least two parts
        return False
    first, last = name.split()[0], name.split()[-1]
    if "." in first or "." in last:  # abbreviations like "T. Wang"
        return False
    if first == first.upper() or last == last.upper():  # KING instead of King
        return False
    return True


def _normalize_name(name):
    """Fix simple fixable problems (whitespace, ALL-CAPS first/last name)."""
    parts = name.strip().split()
    if len(parts) >= 2:
        for idx in (0, -1):  # e.g., KING instead of King
            if parts[idx] == parts[idx].upper() and len(parts[idx]) > 1:
                parts[idx] = parts[idx].capitalize()
    return " ".join(parts)


@dataclasses.dataclass
class _Mailmap:
    """Parsed .mailmap: email→name mapping plus name-ok annotations."""

    names: dict  # email (canonical or alias) -> canonical name
    name_ok: set  # names exempt from _good_name heuristics
    problems: list  # malformed/inconsistent lines (fatal)


def _load_mailmap():
    names, name_ok, problems = dict(), set(), list()
    for number, line in enumerate(
        (repo_root / ".mailmap").read_text("utf-8").splitlines(), 1
    ):
        entry, _, comment = line.strip().partition("#")
        entry = entry.strip()
        if not entry:
            continue
        match = re.match("^([^<]+) <([^<>]+)>", entry)
        if match is None:
            problems.append(f".mailmap:{number}: cannot parse {entry!r}")
            continue
        name = match.group(1).strip()
        if comment.strip().startswith(NAME_OK_MARKER.lstrip("# ")):
            name_ok.add(name)
        elif not _good_name(name):
            problems.append(
                f".mailmap:{number}: name {name!r} looks incomplete; fix it or "
                f"append a trailing {NAME_OK_MARKER!r} comment to the line"
            )
        for email in re.findall("<([^<>]+)>", entry):
            if names.setdefault(email, name) != name:
                problems.append(
                    f".mailmap:{number}: {email} maps to both "
                    f"{names[email]!r} and {name!r}"
                )
    return _Mailmap(names=names, name_ok=name_ok, problems=problems)


@dataclasses.dataclass
class _Unresolved:
    """An author we could not resolve to a good name."""

    pr: str
    name: str | None
    email: str | None
    login: str | None

    @property
    def best_name(self):
        return _normalize_name(self.name or "") or self.login or "FIXME"

    @property
    def mailmap_entry(self):
        """Ready-to-paste .mailmap line, annotated since the name fails heuristics.

        Per our policy the GitHub profile name is acceptable as-is; the
        annotation just records where it came from.
        """
        assert self.email is not None
        return (
            f"{self.best_name} <{self.email}> {NAME_OK_MARKER} "
            f"(auto-added, see {PR_URL.format(pr=self.pr)})"
        )


def _resolve_name(author, pr, mailmap, fallback_names, unresolved):
    """Resolve one PR JSON author dict to a display name (or None to skip)."""
    name, email, login = author.get("n"), author.get("e"), author.get("l")
    if _is_bot(name, email):
        return None
    if email in mailmap.names:
        return mailmap.names[email]
    # GitHub-derived fallback; first good name seen for an email wins so that
    # later profile-name tweaks don't split one person into two entries
    if email in fallback_names:
        return fallback_names[email]
    candidate = _normalize_name(name) if name else None
    if candidate is not None and _good_name(candidate):
        if email is not None:
            fallback_names[email] = candidate
        return candidate
    key = email or f"{name} #{pr}"
    if key not in unresolved:
        unresolved[key] = _Unresolved(pr=pr, name=name, email=email, login=login)
    return None


def _load_pr_ignores():
    """PR numbers whose (huge, automated) diffs we don't credit."""
    ignores = [
        int(ignore.split("#", maxsplit=1)[1].strip().split()[0][:-1])
        for ignore in (repo_root / ".git-blame-ignore-revs")
        .read_text("utf-8")
        .splitlines()
        if not ignore.strip().startswith("#") and ignore.strip()
    ]
    return {str(ig): [] for ig in ignores}


def _load_pr_stats(mailmap):
    """Aggregate per-file line-change stats for every author of every PR."""
    ignores = _load_pr_ignores()
    fallback_names = dict()  # email -> good GitHub-derived name
    unresolved = dict()  # email (or name#pr) -> _Unresolved
    # (name, pr) -> total change count, used for logging the biggest PRs
    commits = defaultdict(int)
    # filename -> name -> [additions, deletions]
    stats = defaultdict(lambda: defaultdict(lambda: np.zeros(2, int)))

    for fname in sorted((data_dir / "prs").glob("*.json"), key=lambda p: int(p.stem)):
        pr = fname.stem
        data = json.loads(fname.read_text("utf-8"))
        assert data != {}, fname
        names = [
            _resolve_name(author, pr, mailmap, fallback_names, unresolved)
            for author in data["authors"]
        ]
        # dedup, keeping author order (so ties in the output sort stay stable)
        names = [name for name in dict.fromkeys(names) if name is not None]
        for file, counts in data["changes"].items():
            if pr in ignores:
                ignores[pr].append(file)
                continue
            p, m = counts["a"], counts["d"]
            # treat moves and permission changes like a single-line change
            if p == m == 0:
                p = 1
            for name in names:
                commits[(name, pr)] += p + m
                stats[file][name] += [p, m]
    return stats, commits, ignores, unresolved


def _check_duplicate_names(names):
    """Warn about surnames shared by more people than we know are distinct."""
    last_map = defaultdict(set)
    for name in names:
        last_map[name.split()[-1]].add(name)
    return [
        f"surname {last!r} is shared by {sorted(these)}; if these are the same "
        "person, merge them in .mailmap, otherwise bump NAME_COUNTS in "
        "credit_tools.py"
        for last, these in last_map.items()
        if len(these) > NAME_COUNTS.get(last, 1)
    ]


def _append_mailmap_entries(entries):
    """Insert entries into .mailmap, sorted like the file-contents-sorter hook."""
    path = repo_root / ".mailmap"
    lines = path.read_text("utf-8").splitlines() + list(entries)
    lines.sort(key=str.lower)
    path.write_text("\n".join(lines) + "\n", "utf-8")


def _report_problems(mailmap, unresolved):
    """Turn all collected problems into a single actionable error message."""
    parts = []
    if mailmap.problems:
        parts.append("Problems in .mailmap:\n" + "\n".join(mailmap.problems))
    fixable = [un for un in unresolved.values() if un.email is not None]
    if fixable:
        parts.append(
            "Unresolved contributor names. Add the entries below to .mailmap\n"
            "(fixing the names if you can figure them out from the PRs), or run\n"
            "`python doc/sphinxext/credit_tools.py --fix-mailmap` to do it for "
            "you:\n\n" + "\n".join(un.mailmap_entry for un in fixable)
        )
    return "\n\n".join(parts)


@verbose
def generate_credit_rst(
    app=None, *, fix_mailmap=False, report_file=None, verbose=False
):
    """Get the credit RST."""
    sphinx_logger.info("Creating code credit RST inclusion file")
    mailmap = _load_mailmap()
    stats, commits, ignores, unresolved = _load_pr_stats(mailmap)
    added = [un for un in unresolved.values() if un.email is not None]
    if fix_mailmap and added and not mailmap.problems:
        _append_mailmap_entries(un.mailmap_entry for un in added)
        sphinx_logger.info(f"Added {len(added)} entries to .mailmap")
        mailmap = _load_mailmap()  # second pass with the appended entries
        stats, commits, ignores, unresolved = _load_pr_stats(mailmap)
    else:
        added = []
    problems = _report_problems(mailmap, unresolved)
    if problems:
        raise RuntimeError(problems)
    # Entries without an email cannot be resolved through .mailmap; skip their
    # credit with a warning (they come from legacy JSONs predating the login
    # backfill, or from PRs whose author deleted their GitHub account)
    skipped = [un for un in unresolved.values() if un.email is None]
    if skipped:
        sphinx_logger.warning(
            f"Skipped credit for {len(skipped)} author entr(ies) with no usable "
            "name and no email, e.g. "
            f"{skipped[0].name!r} in prs/{skipped[0].pr}.json; running "
            "tools/dev/backfill_credit_json.py should fix most of them"
        )

    all_names = {name for these in stats.values() for name in these}
    duplicate_warnings = _check_duplicate_names(all_names)
    for warning in duplicate_warnings:
        sphinx_logger.warning(f"Possible duplicate contributor: {warning}")
    if report_file is not None:
        _write_report(report_file, added, skipped, duplicate_warnings)

    logger.info("Biggest included commits/PRs:")
    biggest = sorted(commits, key=lambda key: commits[key], reverse=True)
    for name, pr in biggest[:10]:
        logger.info(f"{pr.ljust(5)} @ {commits[(name, pr)]:5d} by {name}")

    logger.info("\nIgnored commits:")
    for pr, files in ignores.items():  # should have found one of each
        logger.info(f"ignored {len(files):3d} files for {pr}")
        assert len(files) >= 1, (pr, files)

    mod_stats, link_overrides, mod_file_map = _aggregate_module_stats(stats)
    _write_credit_rst(mod_stats, link_overrides, mod_file_map)


def _github_website(login):
    """Best-effort fetch of a GitHub profile's website URL (for the PR body)."""
    if login is None:
        return None
    try:
        import github  # not a doc dependency, only needed in --report mode

        token = os.environ.get("GITHUB_TOKEN")
        auth = github.Auth.Token(token) if token else None
        with github.Github(auth=auth) as gh:
            website = (gh.get_user(login).blog or "").strip()
    except Exception:
        return None
    if website and not website.startswith("http"):
        website = f"https://{website}"
    return website or None


def _write_report(report_file, added, skipped, duplicate_warnings):
    """Write a Markdown summary for the credit GitHub Action's PR body."""
    lines = ["## Contributor name resolution", ""]
    if added:
        lines += [
            f"{len(added)} new contributor(s) were added to `.mailmap` with their "
            "GitHub profile name taken as-is. To improve a name, check the links "
            "below and edit `.mailmap`:",
            "",
        ]
        for un in added:
            links = [f"[#{un.pr}]({PR_URL.format(pr=un.pr)})"]
            if un.login is not None:
                links.append(f"[profile](https://github.com/{un.login})")
            website = _github_website(un.login)
            if website is not None:
                links.append(f"[website]({website})")
            lines.append(f"- `{un.mailmap_entry}` — {', '.join(links)}")
    else:
        lines += ["All contributor names resolved cleanly."]
    if skipped:
        lines += [
            "",
            f"Credit was skipped for {len(skipped)} author entr(ies) with no "
            "usable name and no email (e.g., "
            f"`{skipped[0].name}` in `prs/{skipped[0].pr}.json`). Running "
            "`tools/dev/backfill_credit_json.py` on this branch should fix "
            "most of them.",
        ]
    if duplicate_warnings:
        lines += ["", "Possible duplicate contributors:", ""]
        lines += [f"- {warning}" for warning in duplicate_warnings]
    Path(report_file).write_text("\n".join(lines) + "\n", "utf-8")


# Changed files matching these globs get no credit (moves, autogenerated, etc.)
_NULL_GLOBS = """
    *.qrc *.png *.svg *.ico *.elc *.sfp *.lout *.lay *.csd *.txt
    mne/_version.py mne/externals/* */__init__.py* */resources.py paper.bib
    mne/html/*.css mne/html/*.js mne/io/bti/tests/data/* */SHA1SUMS *__init__py
    AUTHORS.rst CITATION.cff CONTRIBUTING.rst codemeta.json mne/tests/*.* jr-tools
    */whats_new.rst */latest.inc */dev.rst */changelog.rst */manual/* doc/*.json
    logo/LICENSE doc/credit.rst
"""
# Aliases for old/moved file locations and misc remaps, applied after the scan
# of the current mne/ layout (assigning to the same glob overrides the scan).
# The "doc" entry must precede "maintenance" so doc/*.yml etc. count as doc.
_ALIAS_GLOBS = {
    "mne.preprocessing": "mne/artifacts/*.py mne/csp.py",
    "mne.io": "mne/pick.py mne/constants.py mne/info.py mne/fiff/*.* mne/_fiff/*.* "
    "mne/raw.py mne/testing.py mne/_hdf5.py mne/compensator.py",
    "mne.transforms": "mne/transforms/*.py mne/_freesurfer.py",
    "mne.inverse_sparse": "mne/mixed_norm/*.py mne/sparse_learning/*.py",
    "mne.commands": "mne/__main__.py bin/*",
    "mne.surface": "mne/morph_map.py",
    "mne.epochs": "mne/baseline.py",
    "mne.utils": "mne/parallel.py mne/rank.py mne/misc.py mne/data/*.* "
    "mne/defaults.py mne/fixes.py mne/icons/*.* mne/icons.*",
    "mne.filter": "mne/_ola.py mne/cuda.py",
    "mne.channels": "mne/*digitization/*.py mne/layouts/*.py mne/montages/*.py "
    "mne/selection.py",
    "mne.bem": "mne/bem_surfaces.py",
    "mne.coreg": "mne/coreg/*.py",
    "mne.minimum_norm": "mne/inverse.py",
    "mne.source_estimate": "mne/stc.py",
    "mne.viz": "mne/surfer.py",
    "mne.time_frequency": "mne/tfr.py",
    "mne.report": "mne/html_templates/*.*",
    "mne-connectivity (moved)": "mne/connectivity/*.py",
    "mne-realtime (moved)": "mne/realtime/*.py",
    "doc": "doc/* doc/*.py doc/*.rst",
    "examples": "examples/*.py examples/*.rst",
    "tutorials": "tutorials/*.py tutorials/*.rst",
    "maintenance": ".circleci/* tools/* *.yml *.md setup.* MANIFEST.in Makefile "
    "README.rst flow_diagram.py *.toml debian/* logo/*.py *.git* "
    ".pre-commit-config.yaml .mailmap .coveragerc make/*",
}
_LINK_OVERRIDES = {  # website links that aren't just module paths in this repo
    "mne-connectivity (moved)": "mne-tools/mne-connectivity",
    "mne-realtime (moved)": "mne-tools/mne-realtime",
    "maintenance": "mne-tools/mne-python",
}


def _build_globs():
    """Map changed-filename globs to module names shown on the website.

    The first fnmatch wins, so order is: uncredited ("null") patterns, then the
    scan of the current mne/ layout, then _ALIAS_GLOBS.
    """
    globs = {key: "null" for key in _NULL_GLOBS.split()}
    # These must beat the mne/ scan patterns below
    globs["mne/io/edf/_open.py"] = globs["mne/_edf/open.py"] = "mne.io"
    root_path = Path(mne.__file__).parent
    mod_file_map = dict()
    for file in root_path.iterdir():
        rel = file.relative_to(root_path).with_suffix("")
        mod = f"mne.{rel}"
        if file.is_dir():
            globs[f"mne/{rel}/*.*"] = mod
            globs[f"mne/{rel}.*"] = mod
        elif file.is_file() and file.suffix == ".py":
            key = f"mne/{rel}.py"
            if file.stem == "conftest":
                globs[key] = "maintenance"
                globs["conftest.py"] = "maintenance"
            else:
                globs[key] = mod
                mod_file_map[mod] = key
    for mod, patterns in _ALIAS_GLOBS.items():
        for pattern in patterns.split():
            globs[pattern] = mod
    return globs, _LINK_OVERRIDES, mod_file_map


def _aggregate_module_stats(stats):
    """Roll per-file stats up to per-module stats using the glob mapping."""
    globs, link_overrides, mod_file_map = _build_globs()
    mod_stats = defaultdict(lambda: defaultdict(lambda: np.zeros(2, int)))
    other_files = set()
    private_mods = set()
    total_lines = np.zeros(2, int)
    for fname, counts in stats.items():
        for pattern, mod in globs.items():
            if fnmatch.fnmatch(fname, pattern):
                break
        else:
            other_files.add(fname)
            mod = "other"
        # A private (_-prefixed) submodule needs a remap to a public module
        if mod.startswith("mne.") and mod.split(".")[-1].startswith("_"):
            private_mods.add(f"{mod} (from {fname})")
            continue
        # sanity check a bit
        if mod != "null" and (".png" in fname or "/manual/" in fname):
            raise RuntimeError(f"Unexpected {mod} {fname}")
        for name, pm in counts.items():
            mod_stats[mod][name] += pm
            mod_stats["mne"][name] += pm
            total_lines += pm
    mod_stats.pop("null")  # stuff we shouldn't give credit for
    problems = []
    if private_mods:
        problems.append(
            "Private submodule(s) found in credit page; update _build_globs() in "
            "credit_tools.py to remap them to a public module:\n"
            + "\n".join(sorted(private_mods))
        )
    if other_files:
        problems.append(
            f"{len(other_files)} file(s) not matched by any glob; update "
            "_build_globs() in credit_tools.py:\n" + "\n".join(sorted(other_files))
        )
    if problems:
        raise RuntimeError("\n\n".join(problems))
    mod_stats = {
        mod: mod_stats[mod]
        for mod in sorted(
            mod_stats,
            key=lambda x: (
                not x.startswith("mne"),
                x == "maintenance",
                x.replace("-", "."),
            ),
        )
    }  # sort modules alphabetically
    logger.info(f"\nTotal line change count: {list(map(int, total_lines))}")
    return mod_stats, link_overrides, mod_file_map


def _abbreviate_count(count):
    """Format a count as a max-3-char abbreviation like 123, 1.2k, 123k, 12m."""
    # Round to two digits, e.g. 12340 -> 12000, 12560 -> 13000
    rounded = int(float(f"{count:.2g}"))
    assert rounded > 0, f"Got zero lines changed ({count})"
    for prefix in ("", "k", "m", "g"):
        if rounded >= 1000:
            rounded = rounded / 1000
        else:
            if rounded >= 10 or prefix == "":  # keep single digit as 1 not 1.0
                out = f"{int(round(rounded))}"
            else:
                out = f"{rounded:.1f}"
            return out + prefix
    raise RuntimeError(f"Too many digits in {count}")


def _write_credit_rst(mod_stats, link_overrides, mod_file_map):
    # sphinx-design badges that we use for contributors
    BADGE_KINDS = ["bdg-info-line", "bdg"]
    content = f"""\
.. THIS FILE IS AUTO-GENERATED BY {Path(__file__).stem} AND WILL BE OVERWRITTEN

.. raw:: html

   <style>
   /* Make it occupy more page width */
   .bd-main .bd-content .bd-article-container {{
       max-width: 90vw;
   }}
   /* Limit max card height */
   div.sd-card-body {{
     max-height: 15em;
   }}
   </style>

.. _code_credit:

Code credit
===========

Below are lists of code contributors to MNE-Python. The numbers in parentheses are the
number of lines changed in our code history.

- :{BADGE_KINDS[0]}:`This badge` is used for the top 10% of contributors.
- :{BADGE_KINDS[1]}:`This badge` is used for the remaining 90% of contributors.

Entire codebase
---------------

"""
    for mi, (mod, counts) in enumerate(mod_stats.items()):
        if mi == 0:
            assert mod == "mne", mod
            indent = " " * 3
        elif mi == 1:
            indent = " " * 6
            content += """

By submodule
------------

Contributors often have domain-specific expertise, so we've broken down the
contributions by submodule as well below.

.. grid:: 1 2 3 3
   :gutter: 1

"""
        these_stats = {name: pm.sum() for name, pm in counts.items()}
        these_stats = dict(
            sorted(these_stats.items(), key=lambda kv: kv[1], reverse=True)
        )
        if mi == 0:
            link = "https://github.com/mne-tools/mne-python/graphs/contributors"
        elif mod in link_overrides:
            link = f"https://github.com/{link_overrides[mod]}"
        else:
            kind = "blame" if mod in mod_file_map else "tree"
            link_mod = mod_file_map.get(mod, mod.replace(".", "/"))
            link = f"https://github.com/mne-tools/mne-python/{kind}/main/{link_mod}"
        assert "moved" not in link, (mod, link)
        # Use badges because they flow nicely, inside a grid to make it more compact
        stat_lines = []
        for ki, (name, count) in enumerate(these_stats.items()):
            # if there are 10 this is 100, if there are 100 this is 100
            idx = 0 if ki < (len(these_stats) - 1) // 10 + 1 else 1
            stat_lines.append(
                f":{BADGE_KINDS[idx]}:`{name} ({_abbreviate_count(count)})`"
            )
        stat_lines = f"\n{indent}".join(stat_lines)
        directive = ".. card::" if mi == 0 else "   .. grid-item-card::"
        content += f"""

{directive} {mod}
{indent}:class-card: overflow-auto
{indent}:link: {link}

{indent}{stat_lines}

"""
    (doc_root / "credits" / "code_credit.inc").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix-mailmap",
        action="store_true",
        help="append .mailmap entries for unresolved contributors instead of failing",
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="FILE",
        help="write a Markdown summary suitable for a PR body",
    )
    args = parser.parse_args()
    generate_credit_rst(
        fix_mailmap=args.fix_mailmap, report_file=args.report, verbose=True
    )
