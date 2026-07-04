# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: metrics_store.py
# Description: Stdlib-only child->parent metrics channel: persists each run's JSON-safe
#              summary next to its output file and appends it to a never-pruned history.jsonl
#              at the results-folder root. Imported on both sides of the process boundary
#              (the graph subprocess writes, the Klipper parent process reads), so this
#              module must never import numpy/matplotlib and must never raise: a metrics
#              failure must not fail a graph.

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .console_output import ConsoleOutput

HISTORY_FILENAME = 'history.jsonl'


def json_sanitize(obj: Any) -> Any:
    """Best-effort recursive conversion of obj into native JSON-safe types. Never raises:
    anything that can't be handled is coerced to its string representation (or None)"""
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(key): json_sanitize(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [json_sanitize(value) for value in obj]
        if hasattr(obj, 'tolist'):  # numpy arrays and scalars
            return json_sanitize(obj.tolist())
        if hasattr(obj, 'item'):  # other numpy-like scalar fallback
            return json_sanitize(obj.item())
        return str(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def history_path(st_config) -> Path:
    """Path to the never-pruned run history, at the results-folder root"""
    return st_config.get_results_folder() / HISTORY_FILENAME


def write_run_artifacts(output_target: Path, st_config, graph_type: str, summary: Dict[str, Any]) -> None:
    """Write the sibling '<output_target>.json' summary file and append one compact line to
    history.jsonl. Never raises: a failure here must never fail the graph it's attached to"""
    try:
        record = {
            'ts': datetime.now().isoformat(timespec='seconds'),
            'graph_type': graph_type,
            'summary': json_sanitize(summary),
        }

        with open(output_target.with_suffix('.json'), 'w') as f:
            json.dump(record, f, indent=2)

        history_file = history_path(st_config)
        history_file.parent.mkdir(parents=True, exist_ok=True)
        # A single f.write() below PIPE_BUF is an atomic append: no locking needed since only
        # one Shake&Tune graph subprocess ever runs at a time and the parent only reads the
        # history after wait_for_completion() has returned.
        with open(history_file, 'a') as f:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
    except Exception as e:
        ConsoleOutput.print(f'Warning: unable to write Shake&Tune metrics artifacts: {e}')
        return


def read_current_summary(output_target: Path) -> Optional[Dict[str, Any]]:
    """Read back the 'summary' dict written by write_run_artifacts() for this run. Returns
    None on any error (missing file, malformed JSON, etc.)"""
    try:
        with open(output_target.with_suffix('.json')) as f:
            record = json.load(f)
        if isinstance(record, dict):
            return record.get('summary', record)
        return None
    except Exception:
        return None


def read_history(st_config) -> List[Dict[str, Any]]:
    """Read and parse history.jsonl, skipping any malformed lines. Returns [] if the file
    doesn't exist yet or can't be read at all"""
    records: List[Dict[str, Any]] = []
    try:
        history_file = history_path(st_config)
        if not history_file.exists():
            return records
        with open(history_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return records
    return records


def find_previous(
    history: List[Dict[str, Any]], graph_type: str, axis: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the previous run's history record for graph_type (and, when given, matching
    summary['axis']). The current run's own line has already been appended to history.jsonl
    by the child by the time the parent calls this, so the true previous run is the
    second-to-last match, not the last one"""
    try:
        matches = [
            record
            for record in history
            if record.get('graph_type') == graph_type
            and (axis is None or (record.get('summary') or {}).get('axis') == axis)
        ]
        if len(matches) >= 2:
            return matches[-2]
        return None
    except Exception:
        return None


def _delta_suffix(current: Optional[float], previous: Optional[float], unit: str = '') -> str:
    if previous is None or current is None:
        return ' (first recorded run)'
    return f' (was {previous:.1f}{unit}, Δ{current - previous:+.1f})'


def _print_input_shaper(summary: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[str]:
    fr = summary.get('fr')
    if fr is None:
        return None
    axis = summary.get('axis', 'unknown')
    axis_label = axis.upper() if isinstance(axis, str) else axis
    prev_fr = previous.get('fr') if previous else None
    return f'{axis_label} resonant frequency: {fr:.1f} Hz' + _delta_suffix(fr, prev_fr, ' Hz')


def _print_belts_comparison(summary: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[str]:
    similarity = summary.get('similarity_factor')
    mhi = summary.get('mhi')
    if similarity is None and mhi is None:
        return None
    parts = []
    if similarity is not None:
        prev_similarity = previous.get('similarity_factor') if previous else None
        parts.append(f'Belts similarity: {similarity:.1f}%' + _delta_suffix(similarity, prev_similarity, '%'))
    if mhi is not None:
        parts.append(f'mechanical health: {mhi}')
    return ' | '.join(parts)


def _print_vibrations_profile(summary: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[str]:
    symmetry = summary.get('symmetry_factor')
    motor_fr = summary.get('motor_fr')
    if symmetry is None and motor_fr is None:
        return None
    parts = []
    if symmetry is not None:
        prev_symmetry = previous.get('symmetry_factor') if previous else None
        parts.append(f'Vibrations symmetry: {symmetry:.1f}%' + _delta_suffix(symmetry, prev_symmetry, '%'))
    if motor_fr is not None:
        prev_motor_fr = previous.get('motor_fr') if previous else None
        parts.append(f'motors resonant frequency: {motor_fr:.1f} Hz' + _delta_suffix(motor_fr, prev_motor_fr, ' Hz'))
    return ' | '.join(parts)


def _print_axes_map(summary: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[str]:
    angle_errors = [e for e in (summary.get('angle_errors') or []) if e is not None]
    if not angle_errors:
        return None
    max_error = max(angle_errors)
    prev_errors = [e for e in ((previous or {}).get('angle_errors') or []) if e is not None]
    prev_max_error = max(prev_errors) if prev_errors else None
    return f'Max axis angle error: {max_error:.1f}°' + _delta_suffix(max_error, prev_max_error, '°')


def _print_static_frequency(summary: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Optional[str]:
    energy = summary.get('energy')
    if energy is None:
        return None
    freq = summary.get('freq')
    prefix = f'Static frequency test energy at {freq:.1f} Hz' if freq is not None else 'Static frequency test energy'
    prev_energy = previous.get('energy') if previous else None
    return f'{prefix}: {energy:.1f}' + _delta_suffix(energy, prev_energy)


_SUMMARY_PRINTERS: Dict[str, Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Optional[str]]] = {
    'input shaper': _print_input_shaper,
    'belts comparison': _print_belts_comparison,
    'vibrations profile': _print_vibrations_profile,
    'axes map': _print_axes_map,
    'static frequency': _print_static_frequency,
}


def print_run_summary(st_config, output_target: Path, graph_type: str) -> None:
    """Parent-side: print one compact console line comparing this run's summary against the
    previous recorded run for the same graph_type (and axis, for input shaper). Never raises"""
    try:
        summary = read_current_summary(output_target)
        if summary is None:
            return

        printer = _SUMMARY_PRINTERS.get(graph_type)
        if printer is None:
            return

        axis = summary.get('axis') if isinstance(summary, dict) else None
        history = read_history(st_config)
        previous_record = find_previous(history, graph_type, axis)
        previous_summary = previous_record.get('summary') if previous_record else None

        line = printer(summary, previous_summary)
        if line:
            ConsoleOutput.print(line)
    except Exception:
        return
