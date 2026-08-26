"""Create a list of related software.

To add a package to the list:

1. Add it to the MNE-installers if possible, and it will automatically appear.
2. If it's on PyPI and not in the MNE-installers, add it to the PYPI_PACKAGES set.
3. If it's not on PyPI, add it to the MANUAL_PACKAGES dictionary.

If PyPI or manual, also add package name to `related_software.txt` or
`related_software_nodeps.txt` so that it's installed at doc-build time (for package
metadata querying).
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import functools
import importlib.metadata
import os
import pathlib
import urllib.error
import urllib.request

import joblib
from docutils import nodes
from docutils.parsers.rst import Directive
from mne_doc_utils import sphinx_logger
from sphinx.errors import ExtensionError
from sphinx.util.display import status_iterator

# 1. If a package is in MNE-Installers (preferred method), no need to add it here.
#    But still add it to doc/sphinxext/related_software.txt!

# 2. If it's available on PyPI, add it to this set:
PYPI_PACKAGES = {
    "cross-domain-saliency-maps",
    "meggie",
    "niseq",
    "sesameeg",
    "zuna",
}

# 3. If it's not available on PyPI, add it to this dict:
MANUAL_PACKAGES = {
    # TODO: These packages are not pip-installable as of 2025/11/19, so we have to
    # manually populate them -- should open issues on their package repos.
    "best-python": {
        "Home-page": "https://github.com/multifunkim/best-python",
        "Summary": "The goal of this project is to provide a way to use the best-brainstorm Matlab solvers in Python, compatible with MNE-Python.",  # noqa: E501
    },
    "mne-hcp": {
        "Home-page": "https://github.com/mne-tools/mne-hcp",
        "Summary": "We provide Python tools for seamless integration of MEG data from the Human Connectome Project into the Python ecosystem",  # noqa: E501
    },
    "posthoc": {
        "Home-page": "https://users.aalto.fi/~vanvlm1/posthoc/python",
        "Summary": "post-hoc modification of linear models",
    },
    # https://github.com/freesurfer/surfa/pull/66
    "surfa": {
        "Home-page": "https://github.com/freesurfer/surfa",
        "Summary": "Utilities for medical image and surface processing.",
    },
    # not on PyPI
    "conpy": {
        "Home-page": "https://github.com/aaltoimaginglanguage/conpy",
        "Summary": "Functions and classes for performing connectivity analysis on MEG data.",  # noqa: E501
    },
}


# 4. Add a category to each package.
# A package can belong to more than one category.

CATEGORY_ORDER = [
    "Data I/O and interoperability",
    "Data organization and workflows",
    "Preprocessing and artifact correction",
    "Oscillations and time-frequency analysis",
    "Connectivity and interactions",
    "Source localization and neuroimaging",
    "Statistics, machine learning and decoding",
    "Microstates and neural states",
    "fNIRS, sleep and physiological signals",
    "Visualization, GUIs and real-time analysis",
    "Other",
]

PACKAGE_CATEGORIES = {
    # Data I/O and interoperability
    # -------------------------------------------------------------------------
    "antio": ["Data I/O and interoperability",],
    "curryreader": ["Data I/O and interoperability",],
    "edfio": ["Data I/O and interoperability",],
    "eeg_positions": [
        "Data I/O and interoperability",
    ],
    "eeglabio": [
        "Data I/O and interoperability",
    ],
    "mffpy": [
        "Data I/O and interoperability",
    ],
    "neo": [
        "Data I/O and interoperability",
    ],
    "pybv": [
        "Data I/O and interoperability",
    ],
    "pybvrf": [
        "Data I/O and interoperability",
    ],
    "snirf": [
        "Data I/O and interoperability",
        "fNIRS, sleep and physiological signals",
    ],
    "wfdb": [
        "Data I/O and interoperability",
        "fNIRS, sleep and physiological signals",
    ],

    # -------------------------------------------------------------------------
    # Data organization and workflows
    # -------------------------------------------------------------------------
    "hedtools": [
        "Data organization and workflows",
    ],
    "mne-bids": [
        "Data organization and workflows",
    ],
    "mne-bids-pipeline": [
        "Data organization and workflows",
    ],
    "mne-hcp": [
        "Data organization and workflows",
        "Source localization and neuroimaging",
    ],
    "openneuro-py": [
        "Data organization and workflows",
    ],

    # -------------------------------------------------------------------------
    # Preprocessing and artifact correction
    # -------------------------------------------------------------------------
    "autoreject": [
        "Preprocessing and artifact correction",
    ],
    "meegkit": [
        "Preprocessing and artifact correction",
    ],
    "mne-denoise": [
        "Preprocessing and artifact correction",
    ],
    "mne-faster": [
        "Preprocessing and artifact correction",
    ],
    "mne-icalabel": [
        "Preprocessing and artifact correction",
    ],
    "pyprep": [
        "Preprocessing and artifact correction",
    ],
    "python-picard": [
        "Preprocessing and artifact correction",
    ],

    # -------------------------------------------------------------------------
    # Oscillations and time-frequency analysis
    # -------------------------------------------------------------------------
    "bycycle": [
        "Oscillations and time-frequency analysis",
    ],
    "emd": [
        "Oscillations and time-frequency analysis",
    ],
    "fooof": [
        "Oscillations and time-frequency analysis",
    ],
    "neurodsp": [
        "Oscillations and time-frequency analysis",
    ],
    "nitime": [
        "Oscillations and time-frequency analysis",
    ],
    "pactools": [
        "Oscillations and time-frequency analysis",
        "Connectivity and interactions",
    ],
    "pybispectra": [
        "Oscillations and time-frequency analysis",
        "Connectivity and interactions",
    ],
    "tensorpac": [
        "Oscillations and time-frequency analysis",
        "Connectivity and interactions",
    ],

    # -------------------------------------------------------------------------
    # Connectivity and interactions
    # -------------------------------------------------------------------------
    "conpy": [
        "Connectivity and interactions",
    ],
    "mne-connectivity": [
        "Connectivity and interactions",
    ],

    # -------------------------------------------------------------------------
    # Source localization and neuroimaging
    # -------------------------------------------------------------------------
    "best-python": [
        "Source localization and neuroimaging",
    ],
    "dcm2niix": [
        "Source localization and neuroimaging",
    ],
    "dipy": [
        "Source localization and neuroimaging",
    ],
    "openmeeg": [
        "Source localization and neuroimaging",
    ],
    "sesameeg": [
        "Source localization and neuroimaging",
    ],
    "surfa": [
        "Source localization and neuroimaging",
    ],
    "nilearn": [
        "Source localization and neuroimaging",
        "Statistics, machine learning and decoding",
    ],

    # -------------------------------------------------------------------------
    # Statistics, machine learning and decoding
    # -------------------------------------------------------------------------
    "alphacsc": [
        "Statistics, machine learning and decoding",
    ],
    "cross-domain-saliency-maps": [
        "Statistics, machine learning and decoding",
    ],
    "eelbrain": [
        "Statistics, machine learning and decoding",
    ],
    "mne-ari": [
        "Statistics, machine learning and decoding",
    ],
    "mne-features": [
        "Statistics, machine learning and decoding",
    ],
    "mne-rsa": [
        "Statistics, machine learning and decoding",
    ],
    "niseq": [
        "Statistics, machine learning and decoding",
    ],
    "posthoc": [
        "Statistics, machine learning and decoding",
    ],
    "pyriemann": [
        "Statistics, machine learning and decoding",
    ],
    "rsatoolbox": [
        "Statistics, machine learning and decoding",
    ],

    # -------------------------------------------------------------------------
    # Microstates and neural states
    # -------------------------------------------------------------------------
    "mne-microstates": [
        "Microstates and neural states",
    ],
    "pycrostates": [
        "Microstates and neural states",
    ],

    # -------------------------------------------------------------------------
    # fNIRS, sleep and physiological signals
    # -------------------------------------------------------------------------
    "mne-nirs": [
        "fNIRS, sleep and physiological signals",
    ],
    "neurokit2": [
        "fNIRS, sleep and physiological signals",
        "Preprocessing and artifact correction",
    ],
    "sleepecg": [
        "fNIRS, sleep and physiological signals",
    ],
    "yasa": [
        "fNIRS, sleep and physiological signals",
    ],

    # -------------------------------------------------------------------------
    # Visualization, GUIs and real-time analysis
    # -------------------------------------------------------------------------
    "fsleyes": [
        "Visualization, GUIs and real-time analysis",
    ],
    "meggie": [
        "Visualization, GUIs and real-time analysis",
    ],
    "mne-gui-addons": [
        "Visualization, GUIs and real-time analysis",
    ],
    "mne-kit-gui": [
        "Visualization, GUIs and real-time analysis",
    ],
    "mne-lsl": [
        "Visualization, GUIs and real-time analysis",
    ],
    "mne-qt-browser": [
        "Visualization, GUIs and real-time analysis",
    ],
    "mne-videobrowser": [
        "Visualization, GUIs and real-time analysis",
    ],
    "mnelab": [
        "Visualization, GUIs and real-time analysis",
    ],
}


REQUIRE_INSTALLED = os.getenv("MNE_REQUIRE_RELATED_SOFTWARE_INSTALLED", "false").lower()
REQUIRE_INSTALLED = REQUIRE_INSTALLED in ("true", "1")
REQUIRE_METADATA = REQUIRE_INSTALLED

# These packages pip-install with a different name than the package name
RENAMES = {
    "python-neo": "neo",
    "matplotlib-base": "matplotlib",
}

_memory = joblib.Memory(location=pathlib.Path(__file__).parent / ".joblib", verbose=0)


@_memory.cache(cache_validation_callback=joblib.expires_after(days=7))
def _get_installer_packages():
    """Get the MNE-Python installer package list YAML."""
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/mne-tools/mne-installers/main/recipes/mne-python/construct.yaml"
    ) as url:
        data = url.read().decode("utf-8")
    # Parse data for list of names of packages
    lines = [line.strip() for line in data.splitlines()]
    start_idx = lines.index("# <<< BEGIN RELATED SOFTWARE LIST >>>") + 1
    stop_idx = lines.index("# <<< END RELATED SOFTWARE LIST >>>")
    packages = [
        # Lines look like
        # - mne-ari =0.0.0
        # or similar.
        line.split()[1]
        for line in lines[start_idx:stop_idx]
        if not line.startswith("#")
    ]
    return packages


@functools.lru_cache
def _get_packages() -> dict[str, str]:
    try:
        packages = _get_installer_packages()
    except urllib.error.URLError as exc:  # e.g., bad internet connection
        if not REQUIRE_METADATA:
            sphinx_logger.warning(f"Could not fetch package list, got: {exc}")
            return dict()
        raise
    # There can be duplicates in manual and installer packages because some of the
    # PyPI entries for installer packages are incorrect or unusable (see above), so
    # we don't enforce that. But PyPI and manual should be disjoint:
    dups = set(MANUAL_PACKAGES) & set(PYPI_PACKAGES)
    assert not dups, f"Duplicates in MANUAL_PACKAGES and PYPI_PACKAGES: {sorted(dups)}"
    # And the installer and PyPI-only should be disjoint:
    dups = set(PYPI_PACKAGES) & set(packages)
    assert not dups, (
        f"Duplicates in PYPI_PACKAGES and installer packages: {sorted(dups)}"
    )
    for name in PYPI_PACKAGES | set(MANUAL_PACKAGES):
        if name not in packages:
            packages.append(name)
    # Simple alphabetical order
    packages = sorted(packages, key=lambda x: x.lower())
    packages = [RENAMES.get(package, package) for package in packages]
    out = dict()
    reasons = []
    for package in status_iterator(
        packages, f"Adding {len(packages)} related software packages: "
    ):
        out[package] = dict()
        try:
            if package in MANUAL_PACKAGES:
                md = MANUAL_PACKAGES[package]
            else:
                md = importlib.metadata.metadata(package)
        except importlib.metadata.PackageNotFoundError:
            reasons.append(f"{package}: not found, needs to be installed")
            continue  # raise a complete error later
        else:
            # Every project should really have this
            do_continue = False
            for key in ("Summary",):
                if key not in md:
                    reasons.extend(f"{package}: missing {repr(key)}")
                    do_continue = True
            if do_continue:
                continue
            # It is annoying to find the home page
            url = None
            if "Home-page" in md:
                url = md["Home-page"]
            else:
                for prefix in ("homepage", "documentation", "user documentation"):
                    for key, val in md.items():
                        if key == "Project-URL" and val.lower().startswith(
                            f"{prefix}, "
                        ):
                            url = val.split(", ", 1)[1]
                            break
                    if url is not None:
                        break
                else:
                    reasons.append(
                        f"{package}: could not find Home-page in {sorted(md)}"
                    )
                    continue
            out[package]["url"] = url
            out[package]["description"] = md["Summary"].replace("\n", "")
    if not REQUIRE_INSTALLED:
        reasons = [
            reason
            for reason in reasons
            if "not found, needs to be installed" not in reason
        ]
    reason_str = "\n".join(reasons)
    if reason_str and REQUIRE_METADATA:
        raise ExtensionError(
            f"Could not find suitable metadata for related software:\n{reason_str}"
        )

    return out

def _get_categorized_packages(packages):
    """Group packages by category while preserving package alphabetical order."""
    categorized = {category: [] for category in CATEGORY_ORDER}

    for package in sorted(packages, key=lambda x: x.lower()):
        categories = PACKAGE_CATEGORIES.get(package)

        if categories is None:
            categories = ["Other"]

        for category in categories:
            if category not in CATEGORY_ORDER:
                raise ExtensionError(
                    f"{package}: unknown related software category "
                    f"{repr(category)}"
                )

            categorized[category].append(package)

    # Don't render empty categories.
    return {
        category: package_list
        for category, package_list in categorized.items()
        if package_list
    }

def _validate_package_categories(packages):
    """Validate that package categorization is complete and consistent."""
    unknown_packages = set(PACKAGE_CATEGORIES) - set(packages)

    if unknown_packages:
        raise ExtensionError(
            "Related software categories contain packages that are not "
            "in the related software list:\n"
            + "\n".join(
                f"- {package}"
                for package in sorted(unknown_packages)
            )
        )
    
class RelatedSoftwareDirective(Directive):
    """Create a directive that inserts a bullet list of related software."""

    def run(self):
        """Run the directive."""
        my_list = nodes.bullet_list(bullet="*")
        for package, data in _get_packages().items():
            item = nodes.list_item()
            if "description" not in data:
                para = nodes.paragraph(text=f"{package}")
            else:
                para = nodes.paragraph(text=f": {data['description']}")
                refnode = nodes.reference(
                    "url",
                    package,
                    internal=False,
                    refuri=data["url"],
                )
                para.insert(0, refnode)
            item += para
            my_list.append(item)
        return [my_list]

def run(self):
        """Run the directive."""
        packages = _get_packages()
        _validate_package_categories(packages)
        categorized_packages = _get_categorized_packages(packages)
        content = []

        for category, package_list in categorized_packages.items():
            # Category heading
            section_title = nodes.title(text=category)
            content.append(section_title)

            # Package list
            my_list = nodes.bullet_list(bullet="*")

            for package in package_list:
                data = packages[package]
                item = nodes.list_item()

                if "description" not in data:
                    para = nodes.paragraph(text=package)
                else:
                    para = nodes.paragraph(
                        text=f": {data['description']}"
                    )

                    refnode = nodes.reference(
                        "url",
                        package,
                        internal=False,
                        refuri=data["url"],
                    )

                    para.insert(0, refnode)

                item += para
                my_list.append(item)

            content.append(my_list)

        return content



def setup(app):  # noqa: D103
    app.add_directive("related-software", RelatedSoftwareDirective)
    # Run it as soon as this is added as a Sphinx extension so that any errors
    # / new packages are reported early. The next call in run() will be cached.
    _get_packages()
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


if __name__ == "__main__":  # pragma: no cover
    # running `python doc/sphinxext/related_software.py` for testing
    # require metadata for any installed packages (for debugging)
    REQUIRE_METADATA = True
    items = list(RelatedSoftwareDirective.run(None)[0].children)
    print(f"Got {len(items)} related software packages:")
    for item in items:
        print(f"- {item.astext()}")
