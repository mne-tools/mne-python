#!/bin/bash

set -ex

# Set git default branch name
git config --global init.defaultBranch main

# Work around "dubious ownership in repository" error
git config --global --add safe.directory /workspaces/*

# Use VS Code as default git editor, diff, and merge tool
git config --global core.editor 'code --wait --reuse-window'
git config --global --replace-all difftool.default-difftool.cmd 'code --wait --diff $LOCAL $REMOTE'
git config --global --replace-all diff.tool default-difftool
git config --global --replace-all mergetool.code.cmd 'code --wait --merge $REMOTE $LOCAL $BASE $MERGED'
git config --global --replace-all merge.tool code

# Show indicator for "dirty" git repositories in the shell prompt.
# This can be slow on large repositories and should be disabled in that case.
git config --global --replace-all devcontainers-theme.show-dirty 1

# Make "git blame" ignore certain commits
git config --local blame.ignoreRevsFile .git-blame-ignore-revs
