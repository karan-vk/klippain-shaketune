# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: psd_compat.py
# Description: Digest-based allowlist guard for the native PSD reimplementation. Klipper's
#              PSD-relevant source (ShaperCalibrate._psd / calc_freq_response / CalibrationData)
#              is not version-tagged, so we can't safely assume our native reimplementation
#              matches an arbitrary Klipper/Kalico checkout's math just because an attribute of
#              the right name exists (see helpers/compat.py's KlipperCompatibility for the
#              broader hasattr()-based approach and why it can silently misroute). Instead we
#              hash the *actual source* of the relevant callables and only allow native PSD to
#              run when that exact combined digest has been recorded (by CI, against a real
#              Klipper/Kalico checkout) as verified-equivalent.

import hashlib
import inspect
import re

from ..helpers.console_output import ConsoleOutput

# Digests of the whitespace-normalized source of ShaperCalibrate._psd + calc_freq_response for
# the Klipper/Kalico versions whose PSD math has been verified equivalent to the native
# reimplementation (freq_bins exact, psd_* within rtol=1e-8 — see tests/parity/run_parity.py,
# kernel 4). An unknown digest disables native PSD and falls back to Klipper's own code, so this
# list only ever needs *adding* to; it never causes wrong results. New versions are verified and
# appended here (the parity test prints the digest of any checkout it runs against).
ALLOWLIST = {
    'a74ab85405fcdd9eb767a7d4ecd1b8b5112a572eba6fca2c0e461b8539724ed3',  # klipper3d/klipper master (2026-07)
    '5f2abd6ce69a48debb74474130be9505fcfc0f022b4a40814d6cf540c6802563',  # klipper3d/klipper v0.13.0
    'bcd814828f11fe17a0a79f069bd2c8c63036a7ed46a38a957206d20808ee9e74',  # KalicoCrew/kalico master (2026-07)
}

# Cache of digest -> usable bool, so repeated calls with the same source don't re-hash/re-log.
_digest_cache = {}

_PSD_ATTRS = ('_psd', 'calc_freq_response')
_CALIBRATION_DATA_ATTRS = ('__init__',)


def _normalize_source(src: str) -> str:
    """Whitespace-normalize source text: strip leading/trailing whitespace and collapse any
    run of whitespace to a single space, so purely cosmetic formatting differences (indentation
    style, blank lines, trailing spaces) don't change the digest."""
    return re.sub(r'\s+', ' ', src.strip())


def _iter_relevant_sources(mod_or_instance):
    """Yield normalized source strings for whichever PSD-relevant callables exist on the given
    ShaperCalibrate module/class/instance or CalibrationData class, in a fixed, stable order."""
    # ShaperCalibrate._psd / calc_freq_response (may live on the instance, its class, or be
    # looked up as plain functions/methods - inspect.getsource handles all of those uniformly).
    for attr in _PSD_ATTRS:
        func = getattr(mod_or_instance, attr, None)
        if func is not None:
            try:
                yield _normalize_source(inspect.getsource(func))
            except (OSError, TypeError):
                continue

    # CalibrationData.__init__, if a CalibrationData class/attribute is reachable from here.
    calibration_data = getattr(mod_or_instance, 'CalibrationData', None)
    if calibration_data is not None:
        for attr in _CALIBRATION_DATA_ATTRS:
            func = getattr(calibration_data, attr, None)
            if func is not None:
                try:
                    yield _normalize_source(inspect.getsource(func))
                except (OSError, TypeError):
                    continue


def compute_digest(mod_or_instance) -> str:
    """Compute the combined SHA-256 hex digest of the whitespace-normalized source of every
    PSD-relevant callable reachable from the given ShaperCalibrate module/class/instance. Used
    both by native_psd_usable() and by tests/CI to print/record digests for allowlisting."""
    sha256 = hashlib.sha256()
    for source in _iter_relevant_sources(mod_or_instance):
        sha256.update(source.encode('utf-8'))
    return sha256.hexdigest()


def record_digest(hexdigest: str) -> None:
    """Add a digest to the in-memory allowlist at runtime. Used by the CI parity test to
    self-allowlist the exact Klipper/Kalico checkout under test after verifying equivalence."""
    ALLOWLIST.add(hexdigest)


def native_psd_usable(shaper_calibrate_instance_or_module) -> bool:
    """Return True if the native PSD reimplementation is known-equivalent to the PSD-relevant
    source reachable from the given ShaperCalibrate module/class/instance, i.e. its combined
    source digest is present in ALLOWLIST. Returns False (caller should fall back to Klipper's
    own implementation) for any unknown digest, logging one note the first time it's seen."""
    digest = compute_digest(shaper_calibrate_instance_or_module)

    if digest in _digest_cache:
        return _digest_cache[digest]

    usable = digest in ALLOWLIST
    _digest_cache[digest] = usable

    if not usable:
        ConsoleOutput.print(
            f'[Shake&Tune] Native PSD not enabled: unknown Klipper PSD source digest ({digest[:12]}...), '
            'falling back to the built-in Klipper implementation.'
        )

    return usable
