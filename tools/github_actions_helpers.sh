#!/bin/bash

# For non-pixi 'kinds', we assume a pre-activated environment
run_python() {
  if [[ "${MNE_CI_KIND}" == "pixi" ]]; then
    pixi run python "$@"
  else
    python "$@"
  fi
}

run_pytest() {
  if [[ "${MNE_CI_KIND}" == "pixi" ]]; then
    pixi run pytest "$@"
  else
    pytest "$@"
  fi
}