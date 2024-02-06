#!/usr/bin/env python3

# Adapted from action-towncrier-changelog
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from github import Github
from tomllib import loads

event_name = os.getenv('GITHUB_EVENT_NAME', 'pull_request')
if not event_name.startswith('pull_request'):
    print(f'No-op for {event_name}')
    sys.exit(0)
if 'GITHUB_EVENT_PATH' in os.environ:
    with open(os.environ['GITHUB_EVENT_PATH'], encoding='utf-8') as fin:
        event = json.load(fin)
    pr_num = event['number']
    basereponame = event['pull_request']['base']['repo']['full_name']
    real = True
else:  # local testing
    pr_num = 12318  # added some towncrier files
    basereponame = "mne-tools/mne-python"
    real = False

g = Github(os.environ.get('GITHUB_TOKEN'))
baserepo = g.get_repo(basereponame)

# Grab config from upstream's default branch
toml_cfg = loads(Path("pyproject.toml").read_text("utf-8"))

config = toml_cfg["tool"]["towncrier"]
pr = baserepo.get_pull(pr_num)
modified_files = [f.filename for f in pr.get_files()]

# Get types from config
types = [ent["directory"] for ent in toml_cfg["tool"]["towncrier"]["type"]]
type_pipe = "|".join(types)

# Get files that potentially match the types
directory = toml_cfg["tool"]["towncrier"]["directory"]
assert directory.endswith("/"), directory

file_re = re.compile(rf"^{directory}({type_pipe})\.rst$")
found_stubs = [
    f for f in modified_files if file_re.match(f)
]
for stub in found_stubs:
    fro = stub
    to = file_re.sub(rf"{directory}{pr_num}.\1.rst", fro)
    print(f"Renaming {fro} to {to}")
    if real:
        subprocess.check_call(["mv", fro, to])
