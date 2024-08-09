"""Check the steering committee membership.

Install pygithub and add GITHUB_TOKEN as env var using a personal access token
with read:org, read:user, and read:project
https://docs.github.com/en/enterprise-server@3.6/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
https://pygithub.readthedocs.io/
"""  # noqa: E501

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import pprint
from collections import Counter
from datetime import datetime, timedelta, timezone

from github import Auth, Github
from github.Commit import Commit
from tqdm import tqdm

daysback = 365

# Authenticate
auth = Auth.Token(os.environ["GITHUB_TOKEN"])
g = Github(auth=auth, per_page=100)
print(f"Authenticated as {g.get_user().login}")
org = g.get_organization("mne-tools")
teams = org.get_teams()
team_names = [team.name for team in teams]
team = teams[team_names.index("MNE-Python Steering Committee")]
members = list(team.get_members())
when = (datetime.now().astimezone(timezone.utc) - timedelta(days=daysback)).replace(
    tzinfo=None
)
events = {
    user.login: Counter(Total=0, Commit=0, IssueComment=0, PullRequestComment=0)
    for user in members
}
kinds = ("commits", "pulls_comments", "issues_comments")
ljust = max(len(k) for k in kinds)
for repo in org.get_repos():
    print(f"{repo.name}:")
    for kind in kinds:
        kind_events = getattr(repo, f"get_{kind}")(since=when)
        desc = f" {kind}".ljust(ljust + 2)
        total = kind_events.totalCount
        if total == 0:
            continue
        for event in tqdm(kind_events, total=total, desc=desc):
            if isinstance(event, Commit):
                if event.author is None:  # happens on mne-testing-data
                    continue
                key = event.author.login
            else:
                key = event.user.login
            try:
                count = events[key]
            except KeyError:
                continue
            count["Total"] += 1
            count[type(event).__name__] += 1
events = {k: dict(v) for k, v in events.items()}
ljust = max(len(k) for k in events)
pp = pprint.PrettyPrinter(width=120, compact=True)
pp.pprint({k.ljust(ljust): v for k, v in events.items()})
