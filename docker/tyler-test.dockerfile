# Test Dockerfile — runs cargo test to verify glTF 2.0 conformance fixes
FROM debian:bookworm-slim AS projdb
RUN apt-get update && apt-get install -y --no-install-recommends proj-data \
    && rm -rf /var/lib/apt/lists/*

FROM rust:1.93-alpine AS builder
COPY certs/ /usr/local/share/ca-certificates/
RUN apk add --no-cache ca-certificates && update-ca-certificates
RUN apk add --no-cache build-base cmake clang-dev pkgconf sqlite
ENV PROJ_SYS_SKIP_BINDGEN=1
COPY --from=projdb /usr/share/proj/proj.db /usr/share/proj/proj.db
ENV PROJ_DB_PATH=/usr/share/proj/proj.db
WORKDIR /usr/src/tyler
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY resources ./resources
COPY src ./src
COPY proj ./proj
RUN cargo test --target x86_64-unknown-linux-musl
