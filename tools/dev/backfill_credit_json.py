"""Backfill GitHub logins and noreply emails in doc/sphinxext/prs/*.json.

Historically ./tools/dev/update_credit_json.py only stored the PR author's GitHub
profile name and public email, and the latter is usually unset. This one-time script
adds the author's login ("l") and, where the email is missing, the canonical
``{id}+{login}@users.noreply.github.com`` address ("e") so that every author can be
resolved through .mailmap. It is idempotent and safe to re-run; it only rewrites
files that are missing this information. It makes one API request per PR, so
expect it to take a while (but well under the hourly rate limit).
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import os
from pathlib import Path

from github import Auth, Github
from tqdm import tqdm

prs_path = Path(__file__).parents[2] / "doc" / "sphinxext" / "prs"
json_kwargs = dict(indent=2, ensure_ascii=False, sort_keys=False)

need = []  # PR JSON files whose first author has no email
for fname in sorted(prs_path.glob("*.json"), key=lambda p: int(p.stem)):
    author = json.loads(fname.read_text("utf-8"))["authors"][0]
    if author.get("e") is None:
        need.append(fname)
print(f"{len(need)} of {len(list(prs_path.glob('*.json')))} PR files need backfill")

g = Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"]))
repo = g.get_repo("mne-tools/mne-python")
ghosts, n_updated = [], 0
for fname in tqdm(need, unit="pr", desc="Backfilling"):
    user = repo.get_pull(int(fname.stem)).user
    if user is None or user.login == "ghost":
        ghosts.append(fname.stem)  # deleted account, keep name-based fallback
        continue
    data = json.loads(fname.read_text("utf-8"))
    entry = data["authors"][0]
    if entry.get("e") is None:
        entry["e"] = f"{user.id}+{user.login}@users.noreply.github.com"
    entry["l"] = user.login
    fname.write_text(json.dumps(data, **json_kwargs), encoding="utf-8")
    n_updated += 1
g.close()

print(f"Updated {n_updated} files")
if ghosts:
    print(f"Skipped {len(ghosts)} PRs with deleted authors:")
    print(" ".join(ghosts))
