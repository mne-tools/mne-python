#!/usr/bin/env bash

set -eu

echo -e "🌏 Opening VNC desktop in web browser…\n"
xdg-open 'http://localhost:6080?autoconnect=true'
echo -e "Welcome to the MNE-Python Dev Container!\nCreate a plot in VS Code and it will show up here." | xmessage -file -
