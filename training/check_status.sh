#!/bin/bash
# Quick wrapper that runs the Python status script
cd "$(dirname "$0")"
python3 check_status.py "$@"
