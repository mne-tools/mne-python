import jinja2

autoescape = jinja2.select_autoescape(
    default=True,
    default_for_string=True
)

# For _html_repr_()
repr_templates_env = jinja2.Environment(
    loader=jinja2.PackageLoader(
        package_name='mne.html_templates',
        package_path='repr'
    ),
    autoescape=autoescape
)

# For mne.Report
report_templates_env = jinja2.Environment(
    loader=jinja2.PackageLoader(
        package_name='mne.html_templates',
        package_path='report'
    ),
    autoescape=autoescape
)
report_templates_env.filters['zip'] = zip
