#!/usr/bin/env bash

set -eu

echo "🌏 Opening VNC desktop in web browser…"
xdg-open 'http://localhost:6080?autoconnect=true'
echo -e "Welcome to the MNE-Python Dev Container!\nCreate a plot in VS Code and it will show up here." | xmessage -file -
