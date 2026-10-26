"""Generate the sponsor-review issue body."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import sys
from pathlib import Path

import yaml


def main(output_file):
    """Generate the sponsor-review issue body in output_file."""
    sponsors = yaml.safe_load(Path("doc/_static/sponsors.yml").read_text())
    current = [name for name in sponsors["current"] if not name.endswith("_dk")]
    categories = {
        "Sponsors": sponsors["sponsors"],
        "Supporting institutions": sponsors["partner_institutions"],
    }
    base_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
    source_url = f"{base_url}/blob/{os.environ['GITHUB_SHA']}/doc/_static/sponsors.yml"
    run_url = f"{base_url}/actions/runs/{os.environ['GITHUB_RUN_ID']}"

    with output_file.open("w") as file:
        message = (
            "Please check that the following current sponsors and supporting "
            "institutions remain accurate. They are rendered on the "
            "[homepage](https://mne.tools/dev/) and "
            "[credits page](https://mne.tools/dev/credits/sponsors.html)."
            "\n\n"
            "Update the website listings manually if needed. The checklist is generated"
            f" from [`doc/_static/sponsors.yml`]({source_url})."
        )
        print(message, file=file)

        for heading, entries in categories.items():
            print(file=file)
            print(f"## {heading}", file=file)
            print(file=file)
            for name in current:
                if name not in entries:
                    continue
                entry = entries[name]
                title = entry["title"]
                if "url" in entry:
                    title = f"[{title}]({entry['url']})"
                print(f"- [ ] {title}", file=file)

        print(file=file)
        print(f"Created by [this GitHub Actions run]({run_url}).", file=file)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
