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


def _get_category(file):
    """Get the category of a CI job based on its file path."""
    if file.startswith(".circleci"):
        return "CircleCI"
    if file.startswith(".github/workflows"):
        return "GitHub Actions"
    if file == "azure-pipelines.yml":
        return "Azure Pipelines"
    raise ValueError(f"Unknown CI job file: {file}")


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
                            this_row["category"] = _get_category(file)
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

    # Write the overview information to a file
    with open(
        os.path.join(app.srcdir, "development", "ci_overview.rst"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            """\
.. ci-overview:

Overview of CI jobs
===================

"""
        )

        f.write(
            """\
.. container::

  .. raw:: html

    <style scoped>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet" />
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet" />
    </style>
    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script type="text/javascript" class="init">
    $(document).ready( function () {
        $('table.sg-datatable').DataTable({
            order: [[4, 'asc']], pageLength: 25
        });
    } );
    </script>

  .. list-table::
   :header-rows: 1
   :class: table table-striped sg-datatable

   * - Name
     - Description
     - When it runs
     - File
     - Category
"""  # noqa: E501
        )
        for item in data:
            f.write(
                f"""\
   * - {item["name"]}
     - {item["description"]}
     - {item["runs-when"]}
     - `{item["file"]} <{item["url"]}>`_
     - {item["category"]}
"""
            )
