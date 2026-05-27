#!/bin/bash
cd /mnt/d/github/tyler
docker build --progress=plain -f docker/tyler-test.dockerfile . > /mnt/d/github/tyler/test-build.log 2>&1
echo "EXIT=$?" >> /mnt/d/github/tyler/test-build.log
