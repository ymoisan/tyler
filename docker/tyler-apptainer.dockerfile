#############################
# Builder stage (Alpine + Rust)
#############################

FROM rust:1.77-alpine3.19 AS builder

# Install build and PROJ dependencies
# - build-base: gcc, g++, make, etc.
# - clang: required by bindgen for proj-sys
# - pkgconfig: for pkg-config detection of libproj
# - proj-dev: PROJ >= 9.1.0 with tiff, sqlite, curl support
RUN apk add --no-cache \
    build-base \
    clang \
    pkgconfig \
    proj-dev

WORKDIR /usr/src/tyler

# Copy manifest and sources
COPY Cargo.toml Cargo.lock ./
COPY resources ./resources
COPY src ./src
COPY proj ./proj

# Build and install Tyler binary (includes native-glb support from this fork)
RUN cargo install --path . --locked --bin tyler


#############################
# Runtime stage (minimal Alpine)
#############################

FROM alpine:3.19

# Runtime dependencies:
# - proj, proj-data: PROJ library and grid files
# - tiff: libtiff for PROJ network/grid support
# - sqlite-libs: SQLite runtime required by PROJ
# - libstdc++: C++ standard library for proj / dependencies
# - python3: for roofer2tyler.py preprocessing of Roofer CityJSONL
# - ca-certificates: TLS roots (useful if input paths or future features require HTTPS)
RUN apk add --no-cache \
    proj \
    proj-data \
    tiff \
    sqlite-libs \
    libstdc++ \
    python3 \
    ca-certificates

ENV PROJ_LIB=/usr/share/proj

# Tyler binary
COPY --from=builder /usr/local/cargo/bin/tyler /usr/local/bin/tyler

# Roofer → Tyler converter and entrypoint wrapper
COPY roofer2tyler.py /usr/local/bin/roofer2tyler.py
COPY docker/tyler-entrypoint.sh /usr/local/bin/tyler-entrypoint.sh

RUN chmod +x /usr/local/bin/tyler-entrypoint.sh

WORKDIR /data

ENTRYPOINT ["/usr/local/bin/tyler-entrypoint.sh"]
CMD ["--help"]

