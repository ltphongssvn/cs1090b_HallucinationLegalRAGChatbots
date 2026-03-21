#!/usr/bin/env bash
git diff --cached --name-only | grep -qE "\.(h5|pkl|pth|onnx|parquet|sqlite3|bin|pt|safetensors)$" && echo "ERROR - Large model/binary file. Use DVC or HF Hub. Remove: git rm --cached <file>" && exit 1 || exit 0
