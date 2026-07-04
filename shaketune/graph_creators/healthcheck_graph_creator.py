# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: healthcheck_graph_creator.py
# Description: Healthcheck graph creator implementation. Unlike the other graph creators, in
#              MODE=BASELINE this one also persists a new baseline (via helpers/healthcheck_report)
#              alongside rendering the graph and writing the run's history summary.

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..helpers import healthcheck_report
from ..helpers.accelerometer import MeasurementsManager
from ..shaketune_config import ShakeTuneConfig
from .computations.healthcheck_computation import HealthCheckComputation
from .graph_creator import GraphCreator
from .plotters.healthcheck_plotter import HealthCheckPlotter


@GraphCreator.register('healthcheck')
class HealthCheckGraphCreator(GraphCreator):
    """Healthcheck graph creator using composition-based architecture"""

    def __init__(self, config: ShakeTuneConfig):
        super().__init__(config, HealthCheckComputation, HealthCheckPlotter)
        self._mode: str = 'check'
        self._baseline_file: Optional[Path] = None

    def configure(self, mode: str = 'check', baseline_file: Optional[Path] = None) -> None:
        """Configure the healthcheck parameters"""
        self._mode = mode
        self._baseline_file = baseline_file

    def _create_computation(self, measurements_manager: MeasurementsManager) -> HealthCheckComputation:
        """Create the computation instance with proper configuration"""
        baseline = healthcheck_report.load_baseline(self._config, self._baseline_file)
        return HealthCheckComputation(
            measurements=measurements_manager.get_measurements(),
            max_freq=self._config.max_freq,
            baseline=baseline,
            st_version=self._version,
        )

    def _after_save(self, result) -> None:
        """MODE=BASELINE only: persist the freshly-measured per-axis PSDs as the new baseline for
        future MODE=CHECK comparisons. Hooks into the base create_graph() pipeline (run after the
        figure and history summary are saved) rather than overriding the pipeline itself."""
        if self._mode == 'baseline':
            baseline_record = {
                'ts': datetime.now().isoformat(timespec='seconds'),
                'per_axis': result.per_axis,
            }
            healthcheck_report.save_baseline(self._config, baseline_record, self._baseline_file)
