# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import mne

TARGET_DIRS = [".circleci", ".github/workflows"]
TARGET_FILES = ["azure-pipelines.yml"]


def setup(app):
    app.connect("builder-inited", make_ci_overview)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def setup_module():
    # HACK: Stop nosetests running setup() above
    pass


def make_ci_overview(app=None):
    """Make a CI overview table for the documentation."""
    if app is None:
        return

    # Get files to check for overview information
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for target_dir in TARGET_DIRS:
        files = os.listdir(os.path.join(base_dir, target_dir))
        for file in files:
            if file.endswith((".yml", ".yaml")):
                TARGET_FILES.append(os.path.join(target_dir, file))

    # Extract overview information from each file
    columns = ["name", "description", "runs-when"]
    expected_header = f"# | {' | '.join(columns)} |"
    n_cols = len(columns)
    data = []
    for file in TARGET_FILES:
        with open(os.path.join(base_dir, file), encoding="utf-8") as f:
            lines = f.readlines()
        has_table = True
        while has_table:
            for line in lines:
                if line.startswith(expected_header):
                    # Extract the table rows from the file
                    for row in lines[lines.index(line) + 1 :]:
                        if row.startswith("# | "):
                            this_row = row.removeprefix("# | ").removesuffix(" |\n")
                            this_row = this_row.split(" | ")
                            if len(this_row) != n_cols:
                                raise ValueError(
                                    f"Unexpected number of columns (got "
                                    f"{len(this_row)}, expected {n_cols}) in row: {row}"
                                )
                            this_row = dict(zip(columns, this_row))
                            this_row["file"] = file
                            data.append(this_row)
                        else:
                            has_table = False
                            break
                else:
                    has_table = False

    # Convert file names to links
    if "dev" in mne.__version__:
        kind = "main"
    else:
        kind = "maint/" + ".".join(mne.__version__.split(".")[:2])
    base_url = f"http://github.com/mne-tools/mne-python/blob/{kind}/"
    for item in data:
        item["url"] = f"{base_url}{item['file']}"

    # Format overview information as a list
    overview = [
        """\
.. ci-overview:

Overview of CI jobs
===================
"""
    ]
    for item in data:
        overview.append(f"""\
| **Name**: {item["name"]}
|   **Description**: {item["description"]}
|   **When it runs**: {item["runs-when"]}
|   **File**: `{item["file"]} <{item["url"]}>`_
""")
    # Combine into a single string
    overview = "\n".join(overview)

    # Write to file
    out_file = os.path.join(app.srcdir, "development", "ci_overview.rst")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(overview)
