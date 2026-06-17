# Builder-only Dockerfile for producing a static tyler-041 binary.
#
# Uses Alpine (musl-native) to avoid glibc/musl ABI mismatches.
# All C/C++ code is compiled natively against musl — no cross-compilation.
# SQLite is compiled from source automatically by libsqlite3-sys (bundled
# feature). A small Debian stage provides proj.db (Alpine has no proj-data
# package).
#
# Usage:
#   docker build --no-cache --output type=local,dest=. -f docker/tyler-041.dockerfile .
#
# This writes ./tyler-041 directly to the current directory.

# ── Stage 1: grab proj.db from Debian ──
FROM debian:bookworm-slim AS projdb
RUN apt-get update && apt-get install -y --no-install-recommends proj-data \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Alpine build ──
FROM rust:1.93-alpine AS builder

# Corporate CA certificates (optional; certs/ is gitignored).
COPY certs/ /usr/local/share/ca-certificates/
RUN apk add --no-cache ca-certificates && update-ca-certificates

RUN apk add --no-cache \
    build-base \
    cmake \
    clang-dev \
    pkgconf \
    sqlite

# Skip bindgen: libclang needs dlopen, incompatible with musl static builds.
ENV PROJ_SYS_SKIP_BINDGEN=1

# Embed proj.db from the Debian stage.
COPY --from=projdb /usr/share/proj/proj.db /usr/share/proj/proj.db
ENV PROJ_DB_PATH=/usr/share/proj/proj.db

WORKDIR /usr/src/tyler
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY cityjson-convert ./cityjson-convert
COPY resources ./resources
COPY src ./src

RUN cargo build --release --target x86_64-unknown-linux-musl && \
    cp target/x86_64-unknown-linux-musl/release/tyler /usr/local/bin/tyler-041

RUN file /usr/local/bin/tyler-041 && ldd /usr/local/bin/tyler-041 2>&1 || true

# Minimal final stage — just the binary, for easy extraction.
FROM scratch
COPY --from=builder /usr/local/bin/tyler-041 /tyler-041
