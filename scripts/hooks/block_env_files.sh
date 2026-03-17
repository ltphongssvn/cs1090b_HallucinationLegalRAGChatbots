#!/usr/bin/env bash
git diff --cached --name-only | grep -qE "^\.env(\..+)?$" && echo "ERROR - .env file detected. Never commit secrets!" && exit 1 || exit 0
