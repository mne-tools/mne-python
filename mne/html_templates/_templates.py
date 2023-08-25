import functools


@functools.lru_cache(maxsize=2)
def _get_html_templates_env(kind):
    # For _html_repr_() and mne.Report
    assert kind in ("repr", "report"), kind
    import jinja2

    autoescape = jinja2.select_autoescape(default=True, default_for_string=True)
    templates_env = jinja2.Environment(
        loader=jinja2.PackageLoader(
            package_name="mne.html_templates", package_path=kind
        ),
        autoescape=autoescape,
    )
    if kind == "report":
        templates_env.filters["zip"] = zip
    return templates_env


def _get_html_template(kind, name):
    return _get_html_templates_env(kind).get_template(name)
