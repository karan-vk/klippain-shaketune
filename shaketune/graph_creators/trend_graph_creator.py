# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: trend_graph_creator.py
# Description: Trend graph creator implementation. Unlike every other graph creator, this one
#              consumes no accelerometer measurements at all: it reads back the Shake&Tune
#              metrics history (history.jsonl) and renders how the metrics evolved over time.

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..helpers import metrics_store
from ..helpers.accelerometer import MeasurementsManager
from ..helpers.console_output import ConsoleOutput
from ..shaketune_config import ShakeTuneConfig
from .computations.trend_computation import TrendComputation
from .graph_creator import GraphCreator
from .plotters.trend_plotter import TrendPlotter


@GraphCreator.register('trend')
class TrendGraphCreator(GraphCreator):
    """Trend graph creator: renders the evolution of past runs' metrics. It never records new
    measurements and it must never itself be appended to the history it reads from"""

    writes_history = False

    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config, TrendComputation, TrendPlotter)
        self._last_n: Optional[int] = None
        self._history_file: Optional[Path] = None

    def configure(self, last_n: Optional[int] = None, history_file: Optional[Path] = None) -> None:
        """Configure the trend graph parameters"""
        self._last_n = last_n
        self._history_file = history_file

    def _create_computation(self, measurements_manager: MeasurementsManager) -> TrendComputation:
        """Create the computation instance with proper configuration. The measurements_manager
        is intentionally ignored: this graph type has no accelerometer data of its own"""
        records = self._read_history()
        return TrendComputation(records=records, last_n=self._last_n, st_version=self._version)

    def _read_history(self) -> List[Dict[str, Any]]:
        """Read the run history either from an explicitly provided history file (CLI usage) or
        from the configured results folder's history.jsonl (normal Klipper usage). Never raises:
        an unreadable/missing history simply results in an empty (placeholder) trend graph"""
        if self._history_file is not None:
            records: List[Dict[str, Any]] = []
            try:
                with open(self._history_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            continue
            except Exception as e:
                ConsoleOutput.print(f'Warning: unable to read history file {self._history_file}: {e}')
            return records

        return metrics_store.read_history(self._config)
