"""Create a list of related software.

To add a package to the list:

1. Add it to the MNE-installers if possible, and it will automatically appear.
2. If it's on PyPI and not in the MNE-installers, add it to the PYPI_PACKAGES set.
3. If it's not on PyPI, add it to the MANUAL_PACKAGES dictionary.
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

# If a package is in MNE-Installers, it will be automatically added!

# If it's available on PyPI, add it to this set:
PYPI_PACKAGES = {
    "meggie",
    "niseq",
    "sesameeg",
    "invertmeeg",
}

# If it's not available on PyPI, add it to this dict:
MANUAL_PACKAGES = {
    # TODO: These packages are not pip-installable as of 2024/07/17, so we have to
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
    # This package does not provide wheels, so don't force CircleCI to build it.
    # If it eventually provides binary wheels we could add it to
    # `tools/circleci_dependencies.sh` and remove from here.
    "eelbrain": {
        "Home-page": "https://eelbrain.readthedocs.io/en/stable/",
        "Summary": "Open-source Python toolkit for MEG and EEG data analysis.",
    },
    # mne-kit-gui requires mayavi (ugh)
    "mne-kit-gui": {
        "Home-page": "https://github.com/mne-tools/mne-kit-gui",
        "Summary": "A module for KIT MEG coregistration.",
    },
    # fsleyes requires wxpython, which needs to build
    "fsleyes": {
        "Home-page": "https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes",
        "Summary": "FSLeyes is the FSL image viewer.",
    },
    # dcm2niix must be built from source
    "dcm2niix": {
        "Home-page": "https://github.com/rordenlab/dcm2niix",
        "Summary": "DICOM to NIfTI converter",
    },
    # TODO: these do not set a valid homepage or documentation page on PyPI
    "eeg_positions": {
        "Home-page": "https://eeg-positions.readthedocs.io",
        "Summary": "Compute and plot standard EEG electrode positions.",
    },
    "mne-faster": {
        "Home-page": "https://github.com/wmvanvliet/mne-faster",
        "Summary": "MNE-FASTER: automatic bad channel/epoch/component detection.",  # noqa: E501
    },
    "mne-features": {
        "Home-page": "https://mne.tools/mne-features",
        "Summary": "MNE-Features software for extracting features from multivariate time series",  # noqa: E501
    },
    "mne-rsa": {
        "Home-page": "https://users.aalto.fi/~vanvlm1/mne-rsa",
        "Summary": "Code for performing Representational Similarity Analysis on MNE-Python data structures.",  # noqa: E501
    },
    "mffpy": {
        "Home-page": "https://github.com/BEL-Public/mffpy",
        "Summary": "Reader and Writer for Philips' MFF file format.",
    },
    "emd": {
        "Home-page": "https://emd.readthedocs.io/en/stable",
        "Summary": "Empirical Mode Decomposition in Python.",
    },
    # Needs a release with homepage set properly
    "meegkit": {
        "Home-page": "https://nbara.github.io/python-meegkit",
        "Summary": "Denoising tools for M/EEG processing.",
    },
    # not on PyPI
    "conpy": {
        "Home-page": "https://github.com/aaltoimaginglanguage/conpy",
        "Summary": "Functions and classes for performing connectivity analysis on MEG data.",  # noqa: E501
    },
}

REQUIRE_METADATA = os.getenv("MNE_REQUIRE_RELATED_SOFTWARE_INSTALLED", "false").lower()
REQUIRE_METADATA = REQUIRE_METADATA in ("true", "1")

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
                for prefix in ("homepage", "documentation"):
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
    reason_str = "\n".join(reasons)
    if reason_str and REQUIRE_METADATA:
        raise ExtensionError(
            f"Could not find suitable metadata for related software:\n{reason_str}"
        )

    return out


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


def setup(app):
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
    items = list(RelatedSoftwareDirective.run(None)[0].children)
    print(f"Got {len(items)} related software packages:")
    for item in items:
        print(f"- {item.astext()}")
