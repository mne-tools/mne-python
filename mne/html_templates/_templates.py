import jinja2

repr_env = jinja2.Environment(
    loader=jinja2.PackageLoader('mne.html_templates'),
    autoescape=jinja2.select_autoescape(
        default=True,
        default_for_string=True
    )
)
