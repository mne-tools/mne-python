"""
Generate a bootstrap-icons CSS file with embedded font.

- Install rcssmin (for CSS minification)
- Download and extract bootstrap-icons
- Copy to this directory:
  - bootstrap-icons.css
  - bootstrap-icons.woff2
- Run this script. It will generate bootstrap-icons.mne.css and
  bootstrap-icons.mne.min.css
"""

# Author: Richard Höchenberger <richard.hoechenberger@gmail.com>
#
# License: BSD-3-Clause


from pathlib import Path
import base64
import rcssmin


base_dir = Path('.')
css_path_in = base_dir / 'bootstrap-icons.css'
css_path_out = base_dir / 'bootstrap-icons.mne.css'
css_minified_path_out = base_dir / 'bootstrap-icons.mne.min.css'
font_path = base_dir / 'bootstrap-icons.woff2'


def main():
    """Start the CSS modification."""
    css_in = css_path_in.read_text(encoding='utf-8')
    font_binary = font_path.read_bytes()
    font_b64 = base64.b64encode(font_binary).decode('utf-8')

    css_out = []
    for css in css_in.split('\n'):
        if 'src: url(' in css:
            css = (f'  src: url(data:font/woff2;charset=utf-8;'
                   f'base64,{font_b64}) format("woff2");')
        elif 'url(' in css:
            continue

        css_out.append(css)

    css_out = '\n'.join(css_out)
    css_minified_out = rcssmin.cssmin(style=css_out)

    css_path_out.write_text(
        data=css_out,
        encoding='utf-8'
    )
    css_minified_path_out.write_text(
        data=css_minified_out,
        encoding='utf-8'
    )


if __name__ == '__main__':
    main()
