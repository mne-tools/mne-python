
# Light-weight Desktop (desktop-lite)

Adds a lightweight Fluxbox based desktop to the container that can be accessed using a VNC viewer or the web. GUI-based commands executed from the built-in VS code terminal will open on the desktop automatically.

## Example Usage

```json
"features": {
    "ghcr.io/devcontainers/features/desktop-lite:1": {}
}
```

## Options

| Options Id | Description | Type | Default Value |
|-----|-----|-----|-----|
| version | Currently Unused! | string | latest |
| noVncVersion | The noVNC version to use | string | 1.2.0 |
| password | Enter a password for desktop connections. If `noPassword`, connections from the local host can be established without entering a password | string | vscode |
| webPort | Enter a port for the VNC web client (noVNC) | string | 6080 |
| vncPort | Enter a port for the desktop VNC server (TigerVNC) | string | 5901 |

## Connecting to the desktop

This feature provides two ways of connecting to the desktop environment it adds. The first is to connect using a web browser. To do so:

1. Forward the noVNC port (`6080` by default) to your local machine using either the `forwardPorts` property in `devcontainer.json` or the user interface in your tool (e.g., you can press <kbd>F1</kbd> or <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> and select **Ports: Focus on Ports View** in VS Code to bring it into focus).
1. Open the ports view in your tool, select the noVNC port, and click the Globe icon.
1. In the browser that appears, click the **Connect** button and enter the desktop password (`vscode` by default).

To set up the `6080` port from your `devcontainer.json` file, include the following:
```json
  "forwardPorts": [6080],
  "portsAttributes": {
    "6080": {
      "label": "desktop"
    }
  }
```

You can also connect to the desktop using a [VNC viewer](https://www.realvnc.com/en/connect/download/viewer/). To do so:

1. Connect to the environment from a desktop tool that supports the dev container spec (e.g., VS Code client).
1. Forward the VNC server port (`5901` by default) to your local machine using either the `forwardPorts` property in `devcontainer.json` or the user interface in your tool (e.g., you can press <kbd>F1</kbd> or <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> and select **Ports: Focus on Ports View** in VS Code to bring it into focus).
1. Start your VNC Viewer and connect to localhost:5901. Note that you may need to bump up the color depth to 24 bits to see full color.
1. Enter the desktop password (`vscode` by default).

## Customizing Fluxbox

The window manager installed is [Fluxbox](http://fluxbox.org/). **Right-click** to see the application menu. In addition, any UI-based commands you execute inside the dev container will automatically appear on the desktop.

You can customize the desktop using Fluxbox configuration files. The configuration files are located in the `.fluxbox` folder of the home directory of the user you using to connect to the dev container (`$HOME/.fluxbox`).

If you add custom content to your base image or a Dockerfile in this location, the Feature will automatically use it rather than its default configuration.

See the [Fluxbox menu documentation](http://www.fluxbox.org/help/man-fluxbox-menu.php) for format details. More information on additional customization can be found in Fluxbox's [help](http://www.fluxbox.org/help/) and [general](http://fluxbox.sourceforge.net/docbook/en/html/book1.html) documentation.

## Resolving crashes

If you run into applications crashing, you may need to increase the size of the shared memory space allocated to your container. For example, this will bump it up to 1 GB in `devcontainer.json`:

```json
"runArgs": ["--shm-size=1g"]
```

Or using Docker Compose:

```yaml
services:
  your-service-here:
    # ...
    shm_size: '1gb'
    # ...
```

## Installing a browser

If you need a browser, you can install **Firefox ESR** by adding the following to `.devcontainer/Dockerfile`:

```Dockerfile
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y firefox-esr
```

If you want the full version of **Google Chrome** in the desktop:

1. Add the following to `.devcontainer/Dockerfile`

    ```Dockerfile
    RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
        && curl -sSL https://dl.google.com/linux/direct/google-chrome-stable_current_$(dpkg --print-architecture).deb -o /tmp/chrome.deb \
        && apt-get -y install /tmp/chrome.deb
    ```

2. Chrome sandbox support requires you set up and run as a non-root user. The [`common-utils`](https://github.com/devcontainers/features/tree/main/src/common-utils) script can do this for you, or you [set one up yourself](https://aka.ms/vscode-remote/containers/non-root). Alternatively, you can start Chrome using `google-chrome --no-sandbox`

That's it!


## OS Support

This Feature should work on recent versions of Debian/Ubuntu-based distributions with the `apt` package manager installed.

`bash` is required to execute the `install.sh` script.


---

_Note: This file was auto-generated from the [devcontainer-feature.json](https://github.com/devcontainers/features/blob/main/src/desktop-lite/devcontainer-feature.json).  Add additional notes to a `NOTES.md`._
