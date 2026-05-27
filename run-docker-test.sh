#!/bin/bash
set -e
cd /mnt/d/github/tyler
docker build --progress=plain -f docker/tyler-test.dockerfile . 2>&1 | tee /mnt/d/github/tyler/test-build.log
echo "EXIT=$?"
