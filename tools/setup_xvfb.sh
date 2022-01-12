#!/bin/bash -ef

# this is only relevant for GitHub Actions, but it avoids
# https://github.com/actions/virtual-environments/issues/323
# via
# https://github.community/t/ubuntu-latest-apt-repository-list-issues/17182/10#M4501
sudo rm -f /etc/apt/sources.list.d/dotnetdev.list /etc/apt/sources.list.d/*microsoft*.list

sudo apt update
sudo apt install -yqq libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libopengl0
/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset
