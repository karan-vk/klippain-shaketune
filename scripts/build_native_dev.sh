#!/bin/bash
# Builds the Shake&Tune Rust native extension for local development and vendors
# it into shaketune/native/lib/dev/_core.abi3.so so the Python loader can pick
# it up without going through the full cross-compilation CI matrix.
set -euo pipefail

export PATH="${HOME}/.cargo/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SHAKETUNE_RUST_TREE_HASH=dev cargo build --release --manifest-path "${REPO_ROOT}/rust/Cargo.toml"

mkdir -p "${REPO_ROOT}/shaketune/native/lib/dev"
cp "${REPO_ROOT}/rust/target/release/lib_core.so" "${REPO_ROOT}/shaketune/native/lib/dev/_core.abi3.so"

echo done
