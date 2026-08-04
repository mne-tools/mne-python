"""Collect credit information for recently merged PRs.

A full-history rebuild (removing the cutoff below) takes a long time (hours!)
due to GitHub rate limits, even with a personal GITHUB_TOKEN.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime
import json
import os
import re
from pathlib import Path

from github import Auth, Github
from tqdm import tqdm

auth = Auth.Token(os.environ["GITHUB_TOKEN"])
g = Github(auth=auth, per_page=100)
out_path = Path(__file__).parents[2] / "doc" / "sphinxext" / "prs"
out_path.mkdir(exist_ok=True)

# JSON formatting
json_kwargs = dict(indent=2, ensure_ascii=False, sort_keys=False)
# If the above arguments are changed, existing JSON should also be reformatted with
# something like:
# for fname in sorted(glob.glob("doc/sphinxext/prs/*.json")):
#     fname = Path(fname).resolve(strict=True)
#     fname.write_text(json.dumps(json.loads(fname.read_text("utf-8")), **json_kwargs), "utf-8")  # noqa: E501

repo = g.get_repo("mne-tools/mne-python")
co_re = re.compile("Co-authored-by: ([^<>]+) <([^()>]+)>")
# Iterate closed PRs by last-updated time: merging always bumps updated_at, so
# every PR merged within the lookback window is seen regardless of how old or
# low-numbered it is (the monthly action runs far more often than this window).
# For a full-history rebuild, remove the cutoff `break` below.
cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=90)
pulls_iter = repo.get_pulls(state="closed", sort="updated", direction="desc")
iter_ = tqdm(pulls_iter, unit="pr", desc="Traversing")
n_added = 0
for pull in iter_:
    fname_out = out_path / f"{pull.number}.json"
    if pull.updated_at < cutoff:
        iter_.close()
        print(
            f"After checking {iter_.n + 1} and adding {n_added} PR(s), reached "
            f"PRs last updated before {cutoff.date()}, stopping"
        )
        break
    if fname_out.is_file():
        continue

    # PR diff credit
    if not pull.merged:
        continue
    out = dict()
    # One option is to do a git diff between pull.base and pull.head,
    # but let's see if we can stay pythonic
    out["merge_commit_sha"] = pull.merge_commit_sha
    # Prefer the GitHub username information because it should be most up to date.
    # The user's public email is rarely set, so fall back to the canonical GitHub
    # noreply address, which is what .mailmap entries can then be keyed on.
    name, email, login = pull.user.name, pull.user.email, pull.user.login
    if email is None:
        email = f"{pull.user.id}+{login}@users.noreply.github.com"
    if name is None:
        # no GitHub profile name, pull it from the first commit
        name = pull.get_commits()[0].commit.author.name
    out["authors"] = [dict(n=name, e=email, l=login)]
    # For PR 54 for example this is empty for some reason!
    if out["merge_commit_sha"]:
        try:
            merge_commit = repo.get_commit(out["merge_commit_sha"])
        except Exception:
            pass  # this happens on a lot of old PRs for some reason
        else:
            msg = merge_commit.commit.message.replace("\r", "")
            for n, e in co_re.findall(msg):
                # sometimes commit messages like for 9754 contain all
                # commit messages and include some repeated co-authorship messages
                if n not in {a["n"] for a in out["authors"]}:
                    out["authors"].append(dict(n=n, e=e))
    out["changes"] = dict()
    for file in pull.get_files():
        out["changes"][file.filename] = {
            k[0]: getattr(file, k) for k in ("additions", "deletions")
        }
    n_added += 1
    fname_out.write_text(json.dumps(out, **json_kwargs), encoding="utf-8")

    # TODO: Should add:
    # pull.get_comments()
    # pull.get_review_comments()

g.close()
