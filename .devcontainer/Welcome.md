# 👋 Welcome to the MNE-Python Dev Container!

It's so great to see you! 🤩

This appears to be the first time you're starting up the container,
or you've restarted it after uninstalling MNE-Python.

In any case, **we're currently running the MNE-Python installation
procedure.** You can view progress by opening the terminal window
with the the spinning icon (or exclamation mark, in some cases!) in the bottom-right of
your screen!

Once installation is finished, that terminal window will close and your browser will
open to connect you to a VNC desktop. This is where interactive plots will appear.

Enjoy, have a great day, and: **Happy hacking!** 🚀🚀🚀

### Some technical background

The Dev Container is based on Debian 12 ("bookworm") GNU/Linux.

Python is installed in a `conda` environment (named `base`), together with a few
dependencies that are currently not available from PyPI for all platforms:

- `h5py`
- `psutil`
- `pyside6`
- `vtk`

Everything else is pulled and installed from PyPI through `uv`. Specifically, the command
that is run to install MNE-Python is:

```shell
pipx run uv pip install -e ".[full-pyside6,dev,test_extra]
```
It is totally acceptable and safe to install or update dependencies via `pip` if you
wish.

All `git` pre-commit hooks are automatically installed.

The noVNC server (for connecting to the VNC desktop via a browser) is exposed on TCP
port 6080.

The host's `mne_data` directory is mounted at `~/mne_data` inside the container.
