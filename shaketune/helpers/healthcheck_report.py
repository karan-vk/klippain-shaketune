# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: healthcheck_report.py
# Description: Stdlib-only home for the SHAKETUNE_HEALTHCHECK baseline persistence and the
#              current-vs-baseline comparison logic (PASS/WARN verdict + actionable text).
#              Imported on both sides of the process boundary (the graph subprocess writes the
#              baseline, the Klipper parent process reads it back to print the verdict), so this
#              module must never import numpy/matplotlib and must never raise on I/O errors: a
#              healthcheck failure must never crash the calling command.

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .console_output import ConsoleOutput

BASELINE_FILENAME = 'baseline_healthcheck.json'

# All healthcheck thresholds are sourced here, once, so both the comparison logic and (if ever
# needed) the computation stay in sync.
THRESHOLD_FR_WARN_HZ = 3.0  # Absolute resonant frequency drift that warrants a WARN
THRESHOLD_ZETA_WARN_REL = 0.30  # Relative damping ratio drift (|Δζ|/ζ_baseline) that warrants a WARN
# The half-power-bandwidth zeta estimate is numerically noisy at the small values (~0.01-0.05)
# typical of 3D printers, so a relative-only bar fires spuriously; also require a meaningful ABSOLUTE
# change before warning on damping.
THRESHOLD_ZETA_WARN_ABS = 0.02
THRESHOLD_NEW_PEAK_MIN_REL_AMP = 0.2  # Minimum relative amplitude for a new peak to be worth flagging
NEW_PEAK_PROXIMITY_HZ = 3.0  # A current peak within this distance of a baseline peak isn't "new"


def baseline_path(st_config, explicit: Optional[Path] = None) -> Path:
    """Path to the persisted healthcheck baseline: an explicit override (e.g. from the CLI or a
    custom BASELINE_FILE param) if provided, otherwise the root of the results folder"""
    if explicit is not None:
        return Path(explicit)
    return st_config.get_results_folder() / BASELINE_FILENAME


def load_baseline(st_config, explicit: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read back the persisted baseline record. Returns None on any error (missing file,
    malformed JSON, permission issue, etc.) rather than raising"""
    try:
        with open(baseline_path(st_config, explicit)) as f:
            return json.loads(f.read())
    except Exception:
        return None


def save_baseline(st_config, record: Dict[str, Any], explicit: Optional[Path] = None) -> None:
    """Atomically persist a new baseline record (write to a sibling temp file, then os.replace()
    it into place) so a crash or concurrent read can never observe a half-written baseline.
    Never raises: a failure here must never fail the healthcheck run it's attached to"""
    path = baseline_path(st_config, explicit)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f'.{path.name}.tmp-{os.getpid()}'
        with open(tmp_path, 'w') as f:
            json.dump(record, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        ConsoleOutput.print(f'Warning: unable to save Shake&Tune healthcheck baseline: {e}')


def _find_new_peaks(
    current_peaks: Optional[List[Any]], baseline_peaks: Optional[List[Any]], proximity_hz: float = NEW_PEAK_PROXIMITY_HZ
) -> List[float]:
    """Return the current peak frequencies that don't have any baseline peak within
    proximity_hz -- i.e. peaks that look new since the baseline was captured"""
    if not current_peaks:
        return []
    baseline_freqs = []
    for f in baseline_peaks or []:
        try:
            baseline_freqs.append(float(f))
        except (TypeError, ValueError):
            continue

    new_peaks = []
    for f in current_peaks:
        try:
            f = float(f)
        except (TypeError, ValueError):
            continue
        if not any(abs(f - b) <= proximity_hz for b in baseline_freqs):
            new_peaks.append(f)
    return new_peaks


def _compare_axis(axis: str, current: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[List[str], bool]:
    """Compare one axis' current summary against its baseline summary, returning the lines to
    print for that axis and whether any of them is a WARN"""
    lines: List[str] = []
    warned = False

    fr_warned = False
    fr_cur, fr_base = current.get('fr'), baseline.get('fr')
    if fr_cur is not None and fr_base is not None:
        delta_fr = fr_cur - fr_base
        if abs(delta_fr) > THRESHOLD_FR_WARN_HZ:
            warned = True
            fr_warned = True
            lines.append(
                f'WARN [{axis}]: resonant frequency shifted {delta_fr:+.1f} Hz '
                f'({fr_base:.1f} Hz -> {fr_cur:.1f} Hz) -- check belt tension / mechanical wear'
            )

    # Only flag a damping change when it's both a large RELATIVE and a meaningful ABSOLUTE move, and
    # not when this axis already warned on frequency (the zeta estimate wobbles along with an fr
    # shift, so reporting it too would just be a noisy second line for the same root cause).
    zeta_cur, zeta_base = current.get('zeta'), baseline.get('zeta')
    if not fr_warned and zeta_cur is not None and zeta_base is not None and zeta_base > 0:
        delta_zeta = zeta_cur - zeta_base
        delta_zeta_rel = abs(delta_zeta) / zeta_base
        if delta_zeta_rel > THRESHOLD_ZETA_WARN_REL and abs(delta_zeta) > THRESHOLD_ZETA_WARN_ABS:
            warned = True
            direction = 'increased' if delta_zeta > 0 else 'decreased'
            lines.append(
                f'WARN [{axis}]: damping ratio {direction} by {delta_zeta_rel * 100:.0f}% '
                f'({zeta_base:.3f} -> {zeta_cur:.3f}) -- check for changes in mechanical friction/damping'
            )

    new_peaks = _find_new_peaks(current.get('peak_freqs'), baseline.get('peak_freqs'))
    if new_peaks:
        # detect_peaks() can occasionally report the same physical peak twice (adjacent indices
        # refining to the same local max); dedupe at display resolution so the message stays clean
        formatted = ', '.join(dict.fromkeys(f'{f:.1f} Hz' for f in sorted(new_peaks)))
        lines.append(f'INFO [{axis}]: new resonance peak(s) at {formatted} not present in the baseline')

    return lines, warned


def compare_to_baseline(current: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Compare a current healthcheck summary against a stored baseline, both of shape
    {'per_axis': {'X': {'fr', 'zeta', 'peak_freqs'}, 'Y': {...}}}. Returns (status, lines):
    status is 'WARN' if any axis produced a WARN line, else 'PASS'. Never raises: any malformed
    input degrades to a PASS with no lines rather than crashing the calling command"""
    lines: List[str] = []
    has_warning = False

    try:
        current_axes = (current or {}).get('per_axis') or {}
        baseline_axes = (baseline or {}).get('per_axis') or {}

        for axis in sorted(set(current_axes) & set(baseline_axes)):
            axis_lines, warned = _compare_axis(axis, current_axes.get(axis) or {}, baseline_axes.get(axis) or {})
            lines.extend(axis_lines)
            has_warning = has_warning or warned
    except Exception as e:
        ConsoleOutput.print(f'Warning: unable to compare Shake&Tune healthcheck results to baseline: {e}')
        return 'PASS', []

    return ('WARN' if has_warning else 'PASS'), lines
