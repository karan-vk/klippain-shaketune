# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: trend_computation.py
# Description: Computation implementation for the metrics-history trend graph

from typing import Any, Dict, List, Optional, Tuple

from ..base_models import GraphMetadata
from ..computation_results import TrendResult

# {label: (list_of_ts_strings, list_of_values)}
SeriesDict = Dict[str, Tuple[List[str], List[float]]]


class TrendComputation:
    """Computation for the metrics-history trend graph. Takes already-loaded history.jsonl
    records (no accelerometer measurements involved) and groups them into per-metric time
    series. Records may come from an older or heterogeneous schema (missing keys, unexpected
    types, etc.) so every access here is best-effort and this must never raise"""

    def __init__(self, records: List[Dict[str, Any]], last_n: Optional[int], st_version: str):
        self.records = records if records is not None else []
        self.last_n = last_n
        self.st_version = st_version

    def compute(self) -> TrendResult:
        records = self.records
        try:
            if self.last_n is not None and self.last_n > 0:
                records = records[-self.last_n :]
        except Exception:
            records = self.records

        series: SeriesDict = {}
        for record in records:
            self._ingest_record(series, record)

        metadata = GraphMetadata(title='SHAKE&TUNE METRICS TREND', version=self.st_version)

        return TrendResult(
            metadata=metadata,
            measurements=[],
            series=series,
            n_records=len(self.records),
        )

    def _ingest_record(self, series: SeriesDict, record: Any) -> None:
        """Extract every known metric from a single history record into `series`. Never raises:
        a malformed/unexpected record is simply skipped"""
        try:
            if not isinstance(record, dict):
                return
            summary = record.get('summary')
            if not isinstance(summary, dict):
                return
            graph_type = record.get('graph_type')
            ts = record.get('ts')

            if graph_type == 'input shaper':
                axis = summary.get('axis', '?')
                self._append(series, f'shaper fr [{axis}]', ts, summary.get('fr'))
                self._append(series, f'shaper damping [{axis}]', ts, summary.get('zeta'))

            elif graph_type == 'belts comparison':
                self._append(series, 'belt similarity %', ts, summary.get('similarity_factor'))
                tension_guidance = summary.get('tension_guidance')
                if isinstance(tension_guidance, dict):
                    self._append(series, 'belt tension Δf (Hz)', ts, tension_guidance.get('delta_f'))

            elif graph_type == 'vibrations profile':
                self._append(series, 'vibration symmetry %', ts, summary.get('symmetry_factor'))
                self._append(series, 'motor resonance (Hz)', ts, summary.get('motor_fr'))

            elif graph_type == 'healthcheck':
                per_axis = summary.get('per_axis')
                if isinstance(per_axis, dict):
                    for axis, axis_data in per_axis.items():
                        if isinstance(axis_data, dict):
                            self._append(series, f'healthcheck fr [{axis}]', ts, axis_data.get('fr'))
        except Exception:
            return

    @staticmethod
    def _append(series: SeriesDict, label: str, ts: Any, value: Any) -> None:
        """Append one (ts, value) data point to the named series, skipping silently if value
        isn't a usable number"""
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        ts_list, values_list = series.setdefault(label, ([], []))
        ts_list.append(str(ts) if ts is not None else '')
        values_list.append(value)
