#!/usr/bin/env bash

set -eu

# Exit immediately if not running inside a Dev Container
if [ -z "${RUNNING_IN_DEV_CONTAINER+x}" ]; then
  echo -e "👋 Not running in dev container, not opening web browser.\n"
  exit
fi

echo -e "🌏 Opening VNC desktop in web browser…\n"
xdg-open 'http://localhost:6080?autoconnect=true'
echo -e "Welcome to the MNE-Python Dev Container!\nCreate a plot in VS Code and it will show up here." | xmessage -center -timeout 60 -title "Welcome to MNE-Python!" -file -
