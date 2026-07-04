# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: __init__.py
# Description: Loader for the optional compiled Rust "_core" native extension. Handles
#              architecture/interpreter-bitness candidate selection, a "dev" override slot,
#              a source-tree-hash freshness check against the checkout's "rust/" subtree, and
#              graceful degradation to the pure-Python fallback implementations when no
#              matching (or fresh enough) native binary is available.

import importlib.machinery
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Optional

from ..helpers.console_output import ConsoleOutput

# Cached loaded native module (or None if unavailable/disabled). None also means "not yet
# resolved" the very first time, but we disambiguate that with _native_resolved below since
# a successful resolution that yields "no native available" must also be cached as such.
_native_module = None
_native_resolved = False

# Human-readable explanation of the last selection outcome, surfaced via status().
_native_status = 'not yet resolved'


def _candidate_triples():
    """Return the ordered list of target triples to try, based on interpreter bitness first,
    then platform.machine(). Returns an empty list when there is no known native target."""
    machine = platform.machine()
    is_64bit = sys.maxsize > 2**32

    if is_64bit:
        if machine in ('aarch64', 'arm64'):
            return ['aarch64-unknown-linux-gnu']
        if machine in ('x86_64', 'amd64'):
            return ['x86_64-unknown-linux-gnu']
        return []
    else:
        if machine in ('armv7l', 'armv8l', 'aarch64'):
            return ['armv7-unknown-linux-gnueabihf', 'arm-unknown-linux-gnueabihf']
        if machine == 'armv6l':
            return ['arm-unknown-linux-gnueabihf']
        return []


def _lib_dir() -> Path:
    return Path(__file__).resolve().parent / 'lib'


def _repo_root() -> Path:
    # shaketune/native/__init__.py -> parents[0]=native, [1]=shaketune, [2]=repo root
    return Path(__file__).resolve().parents[2]


def _rust_tree_hash() -> Optional[str]:
    """Return the git tree hash of the "rust" subtree of this checkout, or None if this isn't
    a git checkout (or it has no "rust" path), in which case callers should skip the check."""
    try:
        from git import Repo  # lazy import: GitPython is not a hard runtime dependency

        repo = Repo(_repo_root(), search_parent_directories=True)
        return repo.head.commit.tree['rust'].hexsha
    except Exception:
        return None


def _load_module_from_path(so_path: Path):
    module_name = __name__ + '._core'
    loader = importlib.machinery.ExtensionFileLoader(module_name, str(so_path))
    spec = importlib.util.spec_from_loader(module_name, loader, origin=str(so_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


def _try_load(so_path: Path, *, skip_hash_check: bool, label: str):
    """Attempt to load and validate a single candidate .so. Returns (module, status_str) on
    success, or (None, reason_str) on failure/rejection."""
    if not so_path.exists():
        return None, f'no binary at {so_path}'

    try:
        module = _load_module_from_path(so_path)
    except (ImportError, OSError, Exception) as exc:  # noqa: BLE001 - must never raise
        return None, f'failed to load {so_path} ({exc})'

    if not skip_hash_check:
        tree_hash = getattr(module, '__source_tree_hash__', 'dev')
        if tree_hash != 'dev':
            expected = _rust_tree_hash()
            if expected is None:
                # Not a git checkout (or no "rust" path) - can't verify, accept as-is.
                pass
            elif expected != tree_hash:
                return None, 'stale binary (tree hash mismatch) -> using Python fallback'

    version = getattr(module, '__version__', 'unknown')
    return module, f'loaded {label} (native v{version})'


def _resolve_native():
    """Resolve and cache the native module + status string. Never raises."""
    global _native_module, _native_status

    lib_dir = _lib_dir()

    # (b) Dev override slot: if present, always wins and skips the hash check entirely.
    dev_so = lib_dir / 'dev' / '_core.abi3.so'
    if dev_so.exists():
        module, msg = _try_load(dev_so, skip_hash_check=True, label='dev build')
        if module is not None:
            _native_module = module
            _native_status = f'{msg} (dev build)'
            return
        # Dev slot exists but failed to load - fall through and try arch slots.

    # (c) Arch slots, tried in bitness/machine-appropriate order.
    triples = _candidate_triples()
    if not triples:
        _native_module = None
        _native_status = f'no matching binary for {platform.machine()}/{"64" if sys.maxsize > 2**32 else "32"}bit'
        return

    last_reason = None
    for triple in triples:
        so_path = lib_dir / triple / '_core.abi3.so'
        module, reason = _try_load(so_path, skip_hash_check=False, label=triple)
        if module is not None:
            _native_module = module
            _native_status = reason
            return
        last_reason = reason

    _native_module = None
    if last_reason and last_reason.startswith('stale binary'):
        _native_status = last_reason
    else:
        _native_status = last_reason or f'no matching binary for {platform.machine()}'


def get_native():
    """Return the loaded native "_core" module, or None if unavailable/disabled. The env kill
    switch SHAKETUNE_DISABLE_NATIVE=1 is checked on every call (not just the first) so tests
    can toggle native availability at runtime without restarting the process."""
    global _native_resolved

    if os.environ.get('SHAKETUNE_DISABLE_NATIVE') == '1':
        return None

    if not _native_resolved:
        try:
            _resolve_native()
        except Exception as exc:  # noqa: BLE001 - get_native() must never raise
            global _native_module, _native_status
            _native_module = None
            _native_status = f'failed to resolve native module ({exc})'
        _native_resolved = True
        if _native_module is None:
            ConsoleOutput.print(f'[Shake&Tune] Native module not available: {_native_status}')

    return _native_module


def is_native_available() -> bool:
    return get_native() is not None


def status() -> str:
    """Human-readable explanation of the current native-module selection outcome."""
    if os.environ.get('SHAKETUNE_DISABLE_NATIVE') == '1':
        return 'disabled by env'
    get_native()  # ensure resolution has happened at least once
    return _native_status
