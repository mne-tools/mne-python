# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

import mne

TARGET_DIRS = [".circleci", ".github/workflows"]
TARGET_FILES = ["azure-pipelines.yml"]


def setup(app):
    app.add_directive("ci-overview", CIOverviewDirective)
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


class CIOverviewDirective(SphinxDirective):
    """Insert a sortable table of the CI jobs."""

    def run(self):
        """Run the directive."""
        rst, files = _get_ci_overview_rst()
        # Re-read the page in incremental builds when any of the CI files change
        for file in files:
            self.env.note_dependency(file)
        node = nodes.Element()
        content = StringList(rst.splitlines(), source=self.get_source_info()[0])
        self.state.nested_parse(content, self.content_offset, node)
        return node.children


def _get_ci_overview_rst():
    """Get the rst for the CI overview table and the files it was built from."""
    # Get files to check for overview information
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    target_files = list(TARGET_FILES)
    for target_dir in TARGET_DIRS:
        files = sorted(os.listdir(os.path.join(base_dir, target_dir)))
        for file in files:
            if file.endswith((".yml", ".yaml")):
                target_files.append(os.path.join(target_dir, file))

    # Extract overview information from each file
    columns = ["name", "description", "runs-when"]
    expected_header = f"# | {' | '.join(columns)} |"
    n_cols = len(columns)
    data = []
    for file in target_files:
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

    # Build the table
    rst = """\
.. container::

  .. raw:: html

    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet" />
    <script src="https://code.jquery.com/jquery-3.7.0.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script type="text/javascript" class="init">
    $(document).ready( function () {
        $('table.ci-overview-table').DataTable({
            order: [[4, 'asc']], pageLength: 25
        });
    } );
    </script>

  .. list-table::
   :header-rows: 1
   :class: table ci-overview-table

   * - Name
     - Description
     - When it runs
     - File
     - Category
"""  # noqa: E501
    for item in data:
        rst += f"""\
   * - {item["name"]}
     - {item["description"]}
     - {item["runs-when"]}
     - `{item["file"]} <{item["url"]}>`_
     - {item["category"]}
"""
    return rst, [os.path.join(base_dir, file) for file in target_files]
