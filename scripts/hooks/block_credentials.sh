#!/usr/bin/env bash
git diff --cached --name-only | grep -qE "(\.pem|\.key|id_rsa|id_ed25519|\.netrc|\.git-credentials)$" && echo "ERROR - Credential file detected. Never commit SSH keys or tokens!" && exit 1 || exit 0
