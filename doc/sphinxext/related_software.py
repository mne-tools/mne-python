import functools
import importlib.metadata
import os
import pathlib
import re
import urllib.request

import joblib
import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.errors import ExtensionError
from sphinx.util.display import status_iterator

REQUIRE_METADATA = os.getenv("MNE_REQUIRE_RELATED_SOFTWARE_INSTALLED", "false").lower()
REQUIRE_METADATA = REQUIRE_METADATA in ("true", "1")

# These packages pip-install with a different name than the package name
RENAMES = {
    "python-neo": "neo",
    "matplotlib-base": "matplotlib",
}

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
    # If it eventually provides binary wheels we could add it to doc_related in
    # pyproject.toml and remove from here.
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
    # TODO: mnelab forces PySide6, it can be added to doc_related when we use PySide6
    # for doc building. Also its package does not set the Home-page property.
    "mnelab": {
        "Home-page": "https://github.com/cbrnr/mnelab",
        "Summary": "A graphical user interface for MNE",
    },
    # TODO: these do not set a valid homepage or documentation page
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
}
# TODO: Missing and we should eventually add them to mne-installers,
# but at least alphacsc and eelbrain haven't been rebuilt for Python 3.12.
ALLOW_MISSING_FROM_INSTALLERS = set(
    """
alphaCSC best-python conpy eelbrain meggie niseq sesameeg
""".strip().split()
)

# mne-realtime was removed from this list because it's deprecated in favor of mne-lsl
OLD_LIST = set(
    """
autoreject alphaCSC best-python conpy eelbrain meggie mnelab mne-ari mne-bids
mne-connectivity mne-hcp mne-icalabel mne-microstates mne-nirs mne-rsa
niseq openmeeg pactools posthoc pyprep python-picard sesameeg""".strip().split()
)

_memory = joblib.Memory(location=pathlib.Path(__file__).parent / ".joblib", verbose=0)
_ignore_re = re.compile(
    "^("
    # root stuff
    "python|mne|mne-installer-menus|pip|conda|mamba|"
    # tools that are not domain specific enough
    "git|gh|make|vulture|hatch.*|mypy|twine|.*_profiler|polars|openblas|libblas|"
    "spyder.*|pingouin|jupyter.*|dcm2niix|mayavi|traits.*|pyface|pyside.*|qt6-.+|"
    "pqdm|ipy.*|plotly|trame.*|questionary|pyobjc.+|seaborn|pyvista.*|vtk|qtpy|"
    "xlrd|openpyxl|pyxdf|cython|numba|qdarkstyle|darkdetect|scipy|numpy|pandas|"
    "imageio.*|matplotlib.*|tornado.*|termcolor.*|"
    # doc building and development
    "intersphinx-registry|sphinxcontrib-youtube|sphinx-copybutton|py-spy|ruff|uv|"
    ".*sphinx.*|selenium|.*graphviz|numpydoc|towncrier|check-manifest|codespell|"
    "pre-commit|pytest.*|setuptools.*|defusedxml|"
    ")$"
)


@_memory.cache(cache_validation_callback=joblib.expires_after(days=7))
def _get_installer_yaml():
    """Get the MNE-Python installer package list YAML."""
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/mne-tools/mne-installers/main/recipes/mne-python/construct.yaml"
    ) as url:
        return yaml.safe_load(url.read().decode("utf-8"))


# We need a final allowed packages to prevent cruft from leaking in, e.g., if we add
# some new dependency not at all neuroscience related to the installer we should
# modify our ignore list above, or add it here.
ALLOWED_PACKAGES = set(
    """
autoreject best-python bycycle dipy eeglabio eelbrain emd fooof fsleyes mffpy
mne-ari mne-bids mne-bids-pipeline mne-connectivity mne-faster mne-features
mne-gui-addons mne-hcp mne-icalabel mne-kit-gui mne-lsl mne-microstates mne-nirs
mne-qt-browser mne-rsa mnelab neurodsp neurokit2 nitime openmeeg openneuro-py pactools
posthoc pybv pycrostates pyprep pyriemann python-picard neo sleepecg snirf tensorpac
yasa
""".strip().split()
)


@functools.lru_cache
def _get_packages():
    packages = _get_installer_yaml()["specs"]
    packages = [package.split()[0] for package in packages]
    packages = [package for package in packages if not _ignore_re.match(package)]
    for name in MANUAL_PACKAGES:
        if name not in packages:
            packages.append(name)
    # Simple alphabetical order
    packages = sorted(packages, key=lambda x: x.lower())
    missing = sorted(OLD_LIST - set(packages) - ALLOW_MISSING_FROM_INSTALLERS)
    if missing:
        raise ExtensionError(f"Missing packages from old list:\n{' '.join(missing)}")
    packages = [RENAMES.get(package, package) for package in packages]
    new = set(packages) - set(ALLOWED_PACKAGES)
    if new:
        raise ExtensionError(
            f"New packages from mne-installers not found in ALLOWED_PACKAGES:\n\n"
            f"{'\n'.join(sorted(new))}\n\n"
            "Please add them to ALLOWED_PACKAGES or _ignore_re"
            "in doc/related_software.py"
        )
    out = dict()
    for package in status_iterator(packages, "Adding related software: "):
        out[package] = dict()
        try:
            if package in MANUAL_PACKAGES:
                md = MANUAL_PACKAGES[package]
            else:
                md = importlib.metadata.metadata(package)
        except importlib.metadata.PackageNotFoundError:
            if REQUIRE_METADATA:
                raise
        else:
            # Every project should really have this
            for key in ("Summary",):
                if key not in md:
                    raise ExtensionError(f"Missing {repr(key)} for {package}")
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
                    raise RuntimeError(
                        f"Could not find Home-page for {package} in:\n"
                        f"{sorted(set(md))}\nwith Summary:\n{md['Summary']}"
                    )
            out[package]["url"] = url
            out[package]["description"] = md["Summary"].replace("\n", "")
    return out


class RelatedSoftwareDirective(Directive):
    """Create a directive that inserts a bullet list of related software."""

    def run(self):
        """Run the directive."""
        # TODO: Consider a definition list maybe?
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
    _get_packages()
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


if __name__ == "__main__":  # pragma: no cover
    for item in RelatedSoftwareDirective.run(None)[0].children:
        print(item.astext())
