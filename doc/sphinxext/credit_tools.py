"""Create code credit RST file.

Run ./tools/dev/update_credit_json.py first to get the latest PR JSON files.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import glob
import json
import pathlib
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

# TODO: For contributor names there are three sources of potential truth:
#
# 1. names.inc
# 2. GitHub profile names (that we pull dynamically here)
# 3. commit history / .mailmap.
#
# All three names can mismatch. Currently we try to defer to names.inc since this
# is assumed to have been chosen the most consciously/intentionally by contributors.
# Though it is possible that people can change their preferred names as well, so
# preferring GitHub profile info (when complete!) is probably preferable.

# Allowed singletons
single_names = "btkcodedev buildqa sviter Akshay".split()
# Surnames where we have more than one distinct contributor:
name_counts = dict(
    Bailey=2,
    Das=2,
    Drew=2,
    Li=2,
    Peterson=2,
    Wong=2,
    Zhang=2,
)
# Exceptions, e.g., abbrevitaions in first/last name or all-caps
exceptions = [
    "T. Wang",
    "Ziyi ZENG",
]
# Manual renames
manual_renames = {
    "alexandra": "Alexandra Corneyllie",  # 7600
    "alexandra.corneyllie": "Alexandra Corneyllie",  # 7600
    "akshay0724": "Akshay",  # 4046, TODO: Check singleton
    "AnneSo": "Anne-Sophie Dubarry",  # 4910
    "Basile": "Basile Pinsard",  # 1791
    "ChristinaZhao": "Christina Zhao",  # 9075
    "Drew, J.": "Jordan Drew",  # 10861
    "enzo": "Enzo Altamiranda",  # 11351
    "Frostime": "Yiping Zuo",  # 11773
    "Gennadiy": "Gennadiy Belonosov",  # 11720
    "Genuster": "Gennadiy Belonosov",  # 12936
    "GreasyCat": "Rongfei Jin",  # 13113
    "Hamid": "Hamid Maymandi",  # 10849
    "jwelzel": "Julius Welzel",  # 11118
    "Katia": "Katia Al-Amir",  # 13225
    "Martin": "Martin Billinger",  # 8099, TODO: Check
    "Mats": "Mats van Es",  # 11068
    "Michael": "Michael Krause",  # 3304
    "Naveen": "Naveen Srinivasan",  # 10787
    "NoahMarkowitz": "Noah Markowitz",  # 12669
    "PAB": "Pierre-Antoine Bannier",  # 9430
    "Rob Luke": "Robert Luke",
    "Sena": "Sena Er",  # 11029
    "TzionaN": "Tziona NessAiver",  # 10953
    "Valerii": "Valerii Chirkov",  # 9043
    "Wei": "Wei Xu",  # 13218
    "Zhenya": "Evgenii Kalenkovich",  # 6310, TODO: Check
}


def _good_name(name):
    if name is None:
        return False
    assert isinstance(name, str), type(name)
    if not name.strip():
        return False
    if " " not in name and name not in single_names:  # at least two parts
        return False
    if name not in exceptions and "." in name.split()[0] or "." in name.split()[-1]:
        return False
    if " " in name and name not in exceptions:
        first = name.split()[0]
        last = name.split()[-1]
        if first == first.upper() or last == last.upper():  # e.g., KING instead of King
            return False
    return True


@verbose
def generate_credit_rst(app=None, *, verbose=False):
    """Get the credit RST."""
    sphinx_logger.info("Creating code credit RST inclusion file")
    ignores = [
        int(ignore.split("#", maxsplit=1)[1].strip().split()[0][:-1])
        for ignore in (repo_root / ".git-blame-ignore-revs")
        .read_text("utf-8")
        .splitlines()
        if not ignore.strip().startswith("#") and ignore.strip()
    ]
    ignores = {str(ig): [] for ig in ignores}

    # Use mailmap to help translate emails to names
    mailmap = dict()
    # mapping from email to name
    name_map: dict[str, str] = dict()
    for line in (repo_root / ".mailmap").read_text("utf-8").splitlines():
        name = re.match("^([^<]+) <([^<>]+)>", line.strip()).group(1)
        assert _good_name(name), repr(name)
        emails = list(re.findall("<([^<>]+)>", line.strip()))
        assert len(emails) > 0
        new = emails[0]
        if new in name_map:
            assert name_map[new] == name
        else:
            name_map[new] = name
        if len(emails) == 1:
            continue
        for old in emails[1:]:
            if old in mailmap:
                assert new == mailmap[old]  # can be different names
            else:
                mailmap[old] = new
            if old in name_map:
                assert name_map[old] == name
            else:
                name_map[old] = name

    unknown_emails: set[str] = set()

    # dict with (name, commit) keys, values are int change counts
    # ("commits" is really "PRs" for Python mode)
    commits: dict[tuple[str], int] = defaultdict(lambda: 0)

    # dict with filename keys, values are dicts with name keys and +/- ndarrays
    stats: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: defaultdict(
            lambda: np.zeros(2, int),
        ),
    )

    bad_commits = set()
    expected_bad_names = dict()

    for fname in sorted(glob.glob(str(data_dir / "prs" / "*.json"))):
        commit = Path(fname).stem  # PR number is in the filename
        data = json.loads(Path(fname).read_text("utf-8"))
        del fname
        assert data != {}
        authors = data["authors"]
        for author in authors:
            if (
                author["e"] is not None
                and author["e"] not in name_map
                and _good_name(author["n"])
            ):
                name_map[author["e"]] = author["n"]
        for file, counts in data["changes"].items():
            if commit in ignores:
                ignores[commit].append([file, commit])
                continue
            p, m = counts["a"], counts["d"]
            used_authors = set()
            for author in authors:
                if author["e"] is not None:
                    if author["e"] not in name_map:
                        unknown_emails.add(
                            f"{author['e'].ljust(29)} "
                            "https://github.com/mne-tools/mne-python/pull/"
                            f"{commit}/files"
                        )
                        continue
                    name = name_map[author["e"]]
                else:
                    name = author["n"]
                    if name in manual_renames:
                        assert _good_name(manual_renames[name]), (
                            f"Bad manual rename: {name}"
                        )
                        name = manual_renames[name]
                    if " " in name:
                        first, last = name.rsplit(" ", maxsplit=1)
                        if last == last.upper() and len(last) > 1:
                            last = last.capitalize()
                        if first == first.upper() and len(first) > 1:
                            first = first.capitalize()
                        name = f"{first} {last}"
                        assert not first.upper() == first, f"Bad {name=} from {commit}"
                    assert _good_name(name), f"Bad {name=} from {commit}"
                    if "King" in name:
                        assert name == "Jean-Rémi King", name

                if name is None:
                    bad_commits.add(commit)
                    continue
                if name in used_authors:
                    continue
                if not _good_name(name) and name not in expected_bad_names:
                    expected_bad_names[name] = f"{name} from #{commit}"
                    if author["e"]:
                        expected_bad_names[name] += f" email {author['e']}"
                assert name.strip(), repr(name)
                used_authors.add(name)
                # treat moves and permission changes like a single-line change
                if p == m == 0:
                    p = 1
                commits[(name, commit)] += p + m
                stats[file][name] += [p, m]
    if bad_commits:
        raise RuntimeError(
            "Run:\nrm "
            + " ".join(f"{bad}.json" for bad in sorted(bad_commits, key=int))
        )

    # Check for duplicate names based on last name, and also singleton names.
    last_map = defaultdict(lambda: set())
    bad_names = set()
    for these_stats in stats.values():
        for name in these_stats:
            assert name == name.strip(), f"Un-stripped name: {repr(name)}"
            last = name.split()[-1]
            first = name.split()[0]
            last_map[last].add(name)
            name_where = expected_bad_names.get(name, name)
            if last == name and name not in single_names:
                bad_names.add(f"Singleton:    {name_where}")
            if "." in last or "." in first and name not in exceptions:
                bad_names.add(f"Abbreviation: {name_where}")
    bad_names = sorted(bad_names)
    for last, names in last_map.items():
        if len(names) > name_counts.get(last, 1):
            bad_names.append(f"Duplicates:    {sorted(names)}")
    if bad_names:
        what = (
            "Unexpected possible duplicates or bad names found, "
            f"consider modifying {'/'.join(Path(__file__).parts[-3:])}:\n"
        )
        raise RuntimeError(what + "\n".join(bad_names))

    unknown_emails = set(
        email
        for email in unknown_emails
        if "autofix-ci[bot]" not in email
        and "pre-commit-ci[bot]" not in email
        and "dependabot[bot]" not in email
        and "github-actions[bot]" not in email
    )
    what = "Unknown emails, consider adding to .mailmap:\n"
    assert len(unknown_emails) == 0, what + "\n".join(sorted(unknown_emails))

    logger.info("Biggest included commits/PRs:")
    commits = dict(
        (k, commits[k])
        for k in sorted(commits, key=lambda k_: commits[k_], reverse=True)
    )
    for ni, name in enumerate(commits, 1):
        if ni > 10:
            break
        logger.info(f"{str(name[1]).ljust(5)} @ {commits[name]:5d} by {name[0]}")

    logger.info("\nIgnored commits:")
    # Report the ignores
    for commit in ignores:  # should have found one of each
        logger.info(f"ignored {len(ignores[commit]):3d} files for {commit}")
        assert len(ignores[commit]) >= 1, (ignores[commit], commit)
    globs = dict()

    # This is the mapping from changed filename globs to module names on the website.
    # We need to include aliases for old stuff. Anything we want to exclude we put in
    # "null" with a higher priority (i.e., in dict first):
    link_overrides = dict()  # overrides for links
    for key in """
        *.qrc *.png *.svg *.ico *.elc *.sfp *.lout *.lay *.csd *.txt
        mne/_version.py mne/externals/* */__init__.py* */resources.py paper.bib
        mne/html/*.css mne/html/*.js mne/io/bti/tests/data/* */SHA1SUMS *__init__py
        AUTHORS.rst CITATION.cff CONTRIBUTING.rst codemeta.json mne/tests/*.* jr-tools
        */whats_new.rst */latest.inc */devel.rst */changelog.rst */manual/* doc/*.json
        logo/LICENSE doc/credit.rst
    """.strip().split():
        globs[key] = "null"
    # Now onto the actual module organization
    root_path = pathlib.Path(mne.__file__).parent
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
    globs["mne/artifacts/*.py"] = "mne.preprocessing"
    for key in """
        pick.py constants.py info.py fiff/*.* _fiff/*.* raw.py testing.py _hdf5.py
        compensator.py
    """.strip().split():
        globs[f"mne/{key}"] = "mne.io"
    for key in ("mne/transforms/*.py", "mne/_freesurfer.py"):
        globs[key] = "mne.transforms"
    globs["mne/mixed_norm/*.py"] = "mne.inverse_sparse"
    globs["mne/__main__.py"] = "mne.commands"
    globs["bin/*"] = "mne.commands"
    globs["mne/morph_map.py"] = "mne.surface"
    globs["mne/baseline.py"] = "mne.epochs"
    for key in """
        parallel.py rank.py misc.py data/*.* defaults.py fixes.py icons/*.* icons.*
    """.strip().split():
        globs[f"mne/{key}"] = "mne.utils"
    for key in ("mne/_ola.py", "mne/cuda.py"):
        globs[key] = "mne.filter"
    for key in """
        *digitization/*.py layouts/*.py montages/*.py selection.py
    """.strip().split():
        globs[f"mne/{key}"] = "mne.channels"
    globs["mne/sparse_learning/*.py"] = "mne.inverse_sparse"
    globs["mne/csp.py"] = "mne.preprocessing"
    globs["mne/bem_surfaces.py"] = "mne.bem"
    globs["mne/coreg/*.py"] = "mne.coreg"
    globs["mne/inverse.py"] = "mne.minimum_norm"
    globs["mne/stc.py"] = "mne.source_estimate"
    globs["mne/surfer.py"] = "mne.viz"
    globs["mne/tfr.py"] = "mne.time_frequency"
    globs["mne/connectivity/*.py"] = "mne-connectivity (moved)"
    link_overrides["mne-connectivity (moved)"] = "mne-tools/mne-connectivity"
    globs["mne/realtime/*.py"] = "mne-realtime (moved)"
    link_overrides["mne-realtime (moved)"] = "mne-tools/mne-realtime"
    globs["mne/html_templates/*.*"] = "mne.report"
    globs[".circleci/*"] = "maintenance"
    link_overrides["maintenance"] = "mne-tools/mne-python"
    globs["tools/*"] = "maintenance"
    globs["doc/*"] = "doc"
    for key in ("*.py", "*.rst"):
        for mod in ("examples", "tutorials", "doc"):
            globs[f"{mod}/{key}"] = mod
    for key in """
        *.yml *.md setup.* MANIFEST.in Makefile README.rst flow_diagram.py *.toml
        debian/* logo/*.py *.git* .pre-commit-config.yaml .mailmap .coveragerc make/*
    """.strip().split():
        globs[key] = "maintenance"

    mod_stats = defaultdict(lambda: defaultdict(lambda: np.zeros(2, int)))
    other_files = set()
    total_lines = np.zeros(2, int)
    for fname, counts in stats.items():
        for pattern, mod in globs.items():
            if glob.fnmatch.fnmatch(fname, pattern):
                break
        else:
            other_files.add(fname)
            mod = "other"
        for e, pm in counts.items():
            if mod == "mne._fiff":
                raise RuntimeError
            # sanity check a bit
            if mod != "null" and (".png" in fname or "/manual/" in fname):
                raise RuntimeError(f"Unexpected {mod} {fname}")
            mod_stats[mod][e] += pm
            mod_stats["mne"][e] += pm
            total_lines += pm
    mod_stats.pop("null")  # stuff we shouldn't give credit for
    mod_stats = dict(
        (k, mod_stats[k])
        for k in sorted(
            mod_stats,
            key=lambda x: (
                not x.startswith("mne"),
                x == "maintenance",
                x.replace("-", "."),
            ),
        )
    )  # sort modules alphabetically
    other_files = sorted(other_files)
    if len(other_files):
        raise RuntimeError(
            f"{len(other_files)} misc file(s) found:\n" + "\n".join(other_files)
        )
    logger.info(f"\nTotal line change count: {list(map(int, total_lines))}")

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
        # if there are 10 this is 100, if there are 100 this is 100
        these_stats = dict((k, v.sum()) for k, v in counts.items())
        these_stats = dict(
            (k, these_stats[k])
            for k in sorted(these_stats, key=lambda x: these_stats[x], reverse=True)
        )
        if mod in link_overrides:
            link = f"https://github.com/{link_overrides[mod]}"
        else:
            kind = "blame" if mod in mod_file_map else "tree"
            link_mod = mod_file_map.get(mod, mod.replace(".", "/"))
            link = f"https://github.com/mne-tools/mne-python/{kind}/main/{link_mod}"
        assert "moved" not in link, (mod, link)
        # Use badges because they flow nicely, inside a grid to make it more compact
        stat_lines = []
        for ki, (k, v) in enumerate(these_stats.items()):
            # Round to two digits, e.g. 12340 -> 12000, 12560 -> 13000
            v_round = int(float(f"{v:.2g}"))
            assert v_round > 0, f"Got zero lines changed for {k} in {mod}: {v_round}"
            # And then write as a max-3-char human-readable abbreviation like
            # 123, 1.2k, 123k, 12m, etc.
            for prefix in ("", "k", "m", "g"):
                if v_round >= 1000:
                    v_round = v_round / 1000
                else:
                    if v_round >= 10 or prefix == "":  # keep single digit as 1 not 1.0
                        v_round = f"{int(round(v_round))}"
                    else:
                        v_round = f"{v_round:.1f}"
                    v_round += prefix
                    break
            else:
                raise RuntimeError(f"Too many digits in {v}")
            idx = 0 if ki < (len(these_stats) - 1) // 10 + 1 else 1
            if any(b in k for b in ("[bot]", "Lumberbot", "Deleted user")):
                continue
            assert _good_name(k)
            stat_lines.append(f":{BADGE_KINDS[idx]}:`{k} ({v_round})`")
        stat_lines = f"\n{indent}".join(stat_lines)
        if mi == 0:
            content += f"""

.. card:: {mod}
   :class-card: overflow-auto
   :link: https://github.com/mne-tools/mne-python/graphs/contributors

{indent}{stat_lines}

"""
        else:
            content += f"""

   .. grid-item-card:: {mod}
      :class-card: overflow-auto
      :link: {link}

{indent}{stat_lines}

"""
    (doc_root / "code_credit.inc").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    generate_credit_rst(verbose=True)
