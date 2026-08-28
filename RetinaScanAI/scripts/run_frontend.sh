#!/usr/bin/env bash
# Serves the static frontend on http://localhost:5500
cd "$(dirname "$0")/../frontend"
python -m http.server 5500
