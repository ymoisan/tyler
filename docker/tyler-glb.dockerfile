# Builder Dockerfile for the legacy fork binary `tyler-glb`
# (v0.3.14-based, branch `3dtiles` of github.com/ymoisan/tyler).
#
# Self-contained: clones the fork inside the build.
#
# Uses Debian (glibc) because the fork's vendored proj-sys 0.23.1 always
# runs bindgen, which requires dynamic libclang — incompatible with Alpine
# musl static rust. The resulting `tyler-glb` is glibc-dynamic (not static).
# For a fully static binary, use tyler-041 (proj-sys 0.27 with
# PROJ_SYS_SKIP_BINDGEN).
#
# Patches vendored PROJ 9.1.0 for GCC 13+ compatibility (missing <cstdint>).
#
# Usage:
#   docker build --no-cache --output type=local,dest=. -f docker/tyler-glb.dockerfile .

FROM rust:1.88-bookworm AS builder

COPY certs/ /usr/local/share/ca-certificates/
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    cmake \
    clang \
    libclang-dev \
    g++ \
    pkg-config \
    libsqlite3-dev \
    sqlite3 \
    proj-data \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PROJ_DB_PATH=/usr/share/proj/proj.db \
    CMAKE_POLICY_VERSION_MINIMUM=3.5

WORKDIR /usr/src
ARG FORK_REPO=https://github.com/ymoisan/tyler.git
ARG FORK_BRANCH=3dtiles
ARG PROJ_SUBMODULE_REPO=https://github.com/balazsdukai/proj.git
ARG PROJ_SUBMODULE_BRANCH=main
RUN git clone --depth 1 --branch ${FORK_BRANCH} ${FORK_REPO} tyler && \
    rm -rf tyler/proj && \
    git clone --depth 1 --branch ${PROJ_SUBMODULE_BRANCH} \
        ${PROJ_SUBMODULE_REPO} tyler/proj

WORKDIR /usr/src/tyler

# Patch vendored PROJ 9.1.0 for GCC 13+: add missing <cstdint>.
RUN cd proj/proj-sys/PROJSRC && \
    tar xzf proj-9.1.0.tar.gz && \
    find proj-9.1.0/src \( -name '*.hpp' -o -name '*.cpp' \) \
        -exec sed -i '1i #include <cstdint>' {} + && \
    tar czf proj-9.1.0.tar.gz proj-9.1.0 && \
    rm -rf proj-9.1.0

RUN cargo build --release && \
    cp target/release/tyler-glb /usr/local/bin/tyler-glb

RUN file /usr/local/bin/tyler-glb && ldd /usr/local/bin/tyler-glb 2>&1 || true

FROM scratch
COPY --from=builder /usr/local/bin/tyler-glb /tyler-glb
