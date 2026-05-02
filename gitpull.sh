#!/usr/bin/env bash
# Force local repo to exactly match the remote main branch (via gh-proxy.org).
# Discards all local changes.
set -e
URL="https://gh-proxy.org/https://github.com/jasonwei1002/TALE.git"
git fetch "$URL" main
git reset --hard FETCH_HEAD
