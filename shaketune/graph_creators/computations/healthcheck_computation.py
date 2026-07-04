# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: healthcheck_computation.py
# Description: Computation implementation for the quick per-axis healthcheck sweep

from typing import Any, Dict, List, Optional

import numpy as np

from ...helpers.accelerometer import Measurement
from ...helpers.common_func import compute_mechanical_parameters, detect_peaks
from .. import get_shaper_calibrate_module, process_accelerometer_data_compat
from ..base_models import GraphMetadata
from ..computation_results import HealthCheckResult

PEAKS_DETECTION_THRESHOLD = 0.1  # Threshold to detect peaks in the PSD signal (10% of max)
GRID_POINTS = 300  # Fixed-size resampling grid: bounds the persisted baseline file size


class HealthCheckComputation:
    """Computation for the quick per-axis healthcheck sweep: resamples each axis' PSD onto a
    fixed 300-point frequency grid (so the persisted baseline JSON has a bounded, predictable
    size, mirroring belts_computation's common_freqs trick) and extracts the half-power-bandwidth
    resonant frequency/damping ratio and detected peaks. PASS/WARN text lives parent-side in
    helpers/healthcheck_report.py -- this class is numeric only"""

    def __init__(
        self,
        measurements: List[Measurement],
        max_freq: float,
        baseline: Optional[Dict[str, Any]],
        st_version: str,
    ):
        self.measurements = measurements
        self.max_freq = max_freq
        self.baseline = baseline
        self.st_version = st_version

    def compute(self) -> HealthCheckResult:
        """Perform the per-axis healthcheck computation"""
        shaper_calibrate, _ = get_shaper_calibrate_module()
        grid = np.linspace(0, self.max_freq, GRID_POINTS)

        per_axis: Dict[str, Any] = {}
        for measurement in self.measurements:
            if measurement['samples'] is None:
                continue

            axis = measurement['name'].split('_')[1].upper()
            samples = np.asarray(measurement['samples'], dtype=np.float64)

            fr_resp = process_accelerometer_data_compat(shaper_calibrate, samples)
            freqs = fr_resp.freq_bins
            psd = fr_resp.psd_sum[freqs <= self.max_freq]
            freqs = freqs[freqs <= self.max_freq]

            psd_on_grid = np.interp(grid, freqs, psd)

            fr, zeta, _, _ = compute_mechanical_parameters(psd_on_grid, grid)
            _, _, peak_freqs = detect_peaks(
                psd_on_grid,
                grid,
                PEAKS_DETECTION_THRESHOLD * psd_on_grid.max(),
                window_size=20,
                vicinity=15,
            )

            per_axis[axis] = {
                'freqs': grid.tolist(),
                'psd': psd_on_grid.tolist(),
                'fr': float(fr) if fr else None,
                'zeta': float(zeta) if zeta else None,
                'peak_freqs': [float(f) for f in peak_freqs],
            }

        metadata = GraphMetadata(title='SHAKE&TUNE HEALTHCHECK', version=self.st_version)

        return HealthCheckResult(
            metadata=metadata,
            measurements=self.measurements,
            per_axis=per_axis,
            baseline=self.baseline,
            max_freq=self.max_freq,
        )
