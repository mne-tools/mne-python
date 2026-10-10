"""Refresh the citation counts in doc/documentation/cited.rst from OpenAlex.

Run by the monthly contributor credit action. Google Scholar, which this page
used to quote, hands scripted requests wildly varying undercounts, so we use
the OpenAlex API instead. The works to look up are read back out of the page,
so it stays the only place listing what we count.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import json
import re
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

cited_fname = Path(__file__).parents[2] / "doc" / "documentation" / "cited.rst"
# OpenAlex asks who is calling and gives the faster "polite pool" in return
headers = {"User-Agent": "mne-python (https://github.com/mne-tools/mne-python)"}
entry_re = re.compile(r"^- `(.+?) \(([\d,]+)\) <\S+?(W\d+)>`_$", re.M)

text = cited_fname.read_text("utf-8")
entries = entry_re.findall(text)
assert entries, f"No citation entries found in {cited_fname}"
for name, count, work in entries:
    url = f"https://api.openalex.org/works/{work}"
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as fid:
            new_count = f"{json.load(fid)['cited_by_count']:,}"
    except Exception as exc:  # a flaky lookup should not fail the credit action
        print(f"Leaving {cited_fname.name} alone, {name} lookup failed: {exc}")
        break
    print(f"{name}: {count} -> {new_count}")
    text = text.replace(f"`{name} ({count})", f"`{name} ({new_count})")
else:
    today = date.today()
    text = re.sub(r"(?<=as of )[^:]+", f"{today.day} {today:%B %Y}", text)
    cited_fname.write_text(text, "utf-8")
