#!/usr/bin/env bash
set -e
export PROJ_DB_PATH=$(find /nix/store -name proj.db 2>/dev/null | head -1)
cd /mnt/d/github/tyler
cargo check 2>&1
echo "=== CHECK OK ==="
cargo test 2>&1
echo "=== TEST OK ==="
