#!/usr/bin/env bash
set -euo pipefail

# Build a fully-static tyler binary for x86_64-linux (musl).
# Output: tyler-041 — distinguishes this v0.4.1-based build from the
# legacy tyler-glb fork binary.
#
# Prerequisites:
#   - rustup target add x86_64-unknown-linux-musl
#   - musl-tools (musl-gcc)
#   - cmake, make (for bundled PROJ)
#
# Usage:
#   ./build-musl.sh          # release
#   ./build-musl.sh --debug  # debug

TARGET="x86_64-unknown-linux-musl"
PROFILE="release"

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE="dev"
fi

rustup target add "$TARGET" 2>/dev/null || true

export CC_x86_64_unknown_linux_musl="musl-gcc"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="musl-gcc"

if [[ "$PROFILE" == "release" ]]; then
    cargo build --release --target "$TARGET"
    SRC="target/${TARGET}/release/tyler"
    DST="target/${TARGET}/release/tyler-041"
else
    cargo build --target "$TARGET"
    SRC="target/${TARGET}/debug/tyler"
    DST="target/${TARGET}/debug/tyler-041"
fi

cp -f "$SRC" "$DST"

if command -v file &>/dev/null; then
    echo ""
    file "$DST"
fi
if command -v ldd &>/dev/null; then
    echo ""
    ldd "$DST" 2>&1 || true
fi

echo ""
echo "Binary: $DST"
ls -lh "$DST"
