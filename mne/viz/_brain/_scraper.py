# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import os.path as op

from ._brain import Brain


class _BrainScraper:
    """Scrape Brain objects."""

    def __repr__(self):
        return "<BrainScraper>"

    def __call__(self, block, block_vars, gallery_conf):
        rst = ""
        for brain in list(block_vars["example_globals"].values()):
            # Only need to process if it's a brain with a time_viewer
            # with traces on and shown in the same window, otherwise
            # PyVista and matplotlib scrapers can just do the work
            if (not isinstance(brain, Brain)) or brain._closed:
                continue
            from matplotlib import animation
            from matplotlib import pyplot as plt
            from sphinx_gallery.scrapers import matplotlib_scraper

            img = brain.screenshot(time_viewer=True)
            dpi = 100.0
            figsize = (img.shape[1] / dpi, img.shape[0] / dpi)
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = plt.Axes(fig, [0, 0, 1, 1])
            fig.add_axes(ax)
            img = ax.imshow(img)
            movie_key = "# brain.save_movie"
            if movie_key in block[1]:
                kwargs = dict()
                # Parse our parameters
                lines = block[1].splitlines()
                for li, line in enumerate(block[1].splitlines()):
                    if line.startswith(movie_key):
                        line = line[len(movie_key) :].replace("..., ", "")
                        for ni in range(1, 5):  # should be enough
                            if len(lines) > li + ni and lines[li + ni].startswith(
                                "#  "
                            ):
                                line = line + lines[li + ni][1:].strip()
                            else:
                                break
                        assert line.startswith("(") and line.endswith(")")
                        kwargs.update(eval(f"dict{line}"))  # nosec B307
                for key, default in [
                    ("time_dilation", 4),
                    ("framerate", 24),
                    ("tmin", None),
                    ("tmax", None),
                    ("interpolation", None),
                    ("time_viewer", False),
                ]:
                    if key not in kwargs:
                        kwargs[key] = default
                kwargs.pop("filename", None)  # always omit this one
                if brain.time_viewer:
                    assert kwargs["time_viewer"], "Must use time_viewer=True"
                frames = brain._make_movie_frames(callback=None, **kwargs)

                # Turn them into an animation
                def func(frame):
                    img.set_data(frame)
                    return [img]

                anim = animation.FuncAnimation(
                    fig,
                    func=func,
                    frames=frames,
                    blit=True,
                    interval=1000.0 / kwargs["framerate"],
                )

                # Out to sphinx-gallery:
                #
                # 1. A static image but hide it (useful for carousel)
                if animation.FFMpegWriter.isAvailable():
                    writer = "ffmpeg"
                elif animation.ImageMagickWriter.isAvailable():
                    writer = "imagemagick"
                else:
                    writer = None
                static_fname = next(block_vars["image_path_iterator"])
                static_fname = static_fname[:-4] + ".gif"
                anim.save(static_fname, writer=writer, dpi=dpi)
                rel_fname = op.relpath(static_fname, gallery_conf["src_dir"])
                rel_fname = rel_fname.replace(os.sep, "/").lstrip("/")
                rst += f"\n.. image:: /{rel_fname}\n    :class: hidden\n"

                # 2. An animation that will be embedded and visible
                block_vars["example_globals"]["_brain_anim_"] = anim

            brain.close()
            rst += matplotlib_scraper(block, block_vars, gallery_conf)
        return rst
