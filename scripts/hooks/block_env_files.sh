#!/usr/bin/env bash
blocked=$(git diff --cached --name-only | grep -E "^\.env(\..+)?$" | grep -vE "^\.env\.(template|example)$")
if [ -n "$blocked" ]; then
    echo "ERROR - .env file detected. Never commit secrets!"
    echo "Blocked files: $blocked"
    exit 1
fi
exit 0
