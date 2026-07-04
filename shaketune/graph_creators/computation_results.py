# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: computation_results.py
# Description: Specific computation result models for each graph type

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..helpers.common_func import AXIS_CONFIG
from .base_models import ComputationResult


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort cast to a plain float, returning None instead of raising"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_axis_from_name(name: str) -> str:
    """Infer the short axis identifier (e.g. 'x', 'a') from a measurement name by doing
    a longest-prefix match against AXIS_CONFIG's 'label' and 'axis' entries"""
    best_axis = 'unknown'
    best_len = 0
    lname = (name or '').lower()
    for entry in AXIS_CONFIG:
        for candidate in (entry['label'], entry['axis']):
            if candidate and lname.startswith(candidate.lower()) and len(candidate) > best_len:
                best_axis = entry['axis']
                best_len = len(candidate)
    return best_axis


@dataclass
class AxesMapResult(ComputationResult):
    """Result from axes map detection computation using velocity-based algorithm"""

    acceleration_data: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    velocity_data: List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    gravity: float
    noise_level: float
    quality_status: Dict[str, Any]
    peak_velocities_data: List[Dict[str, float]]
    direction_vectors: List[np.ndarray]
    actual_directions: List[np.ndarray]
    rotation_matrix: np.ndarray  # Orthonormalized 3x3 rotation matrix
    euler_angles: Tuple[float, float, float]  # (roll, pitch, yaw) in degrees
    angle_errors: List[float]
    confidences: List[float]
    formatted_direction_vector: str
    accel: Optional[float] = None
    extrapolated_axis: Optional[int] = None  # Index (0=X, 1=Y, 2=Z) if 2-axis machine

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'acceleration_data': self.acceleration_data,
            'velocity_data': self.velocity_data,
            'gravity': self.gravity,
            'noise_level': self.noise_level,
            'quality_status': self.quality_status,
            'peak_velocities_data': self.peak_velocities_data,
            'direction_vectors': self.direction_vectors,
            'actual_directions': self.actual_directions,
            'rotation_matrix': self.rotation_matrix,
            'euler_angles': self.euler_angles,
            'angle_errors': self.angle_errors,
            'confidences': self.confidences,
            'formatted_direction_vector': self.formatted_direction_vector,
            'measurements': self.measurements,
            'accel': self.accel,
            'extrapolated_axis': self.extrapolated_axis,
            'st_version': self.metadata.version,
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            roll, pitch, yaw = self.euler_angles if self.euler_angles is not None else (None, None, None)
            return {
                'formatted_direction_vector': self.formatted_direction_vector,
                'angle_errors': [_safe_float(a) for a in self.angle_errors] if self.angle_errors is not None else [],
                'noise_level': _safe_float(self.noise_level),
                'euler_angles': {
                    'roll': _safe_float(roll),
                    'pitch': _safe_float(pitch),
                    'yaw': _safe_float(yaw),
                },
            }
        except Exception:
            return {
                'formatted_direction_vector': None,
                'angle_errors': [],
                'noise_level': None,
                'euler_angles': None,
            }


@dataclass
class SignalData:
    """Data for a single signal in belts comparison"""

    freqs: np.ndarray
    psd: np.ndarray
    peaks: np.ndarray
    paired_peaks: Optional[List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]] = None
    unpaired_peaks: Optional[List[int]] = None


@dataclass
class BeltsResult(ComputationResult):
    """Result from belts comparison computation"""

    signal1: SignalData
    signal2: SignalData
    signal1_belt: str
    signal2_belt: str
    kinematics: Optional[str]
    test_params: Any  # testParams type
    max_freq: float
    max_scale: Optional[int]
    similarity_factor: Optional[float] = None
    mhi: Optional[str] = None
    tension_guidance: Optional[Dict[str, Any]] = None

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'signal1': self.signal1,
            'signal2': self.signal2,
            'similarity_factor': self.similarity_factor,
            'mhi': self.mhi,
            'signal1_belt': self.signal1_belt,
            'signal2_belt': self.signal2_belt,
            'kinematics': self.kinematics,
            'test_params': self.test_params,
            'st_version': self.metadata.version,
            'measurements': self.measurements,
            'max_freq': self.max_freq,
            'max_scale': self.max_scale,
            'tension_guidance': self.tension_guidance,
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            return {
                'similarity_factor': _safe_float(self.similarity_factor),
                'mhi': self.mhi,
                'belt1': _belt_signal_summary(self.signal1_belt, self.signal1),
                'belt2': _belt_signal_summary(self.signal2_belt, self.signal2),
                'paired_peak_deltas_hz': _paired_peak_deltas_hz(self.signal1.paired_peaks),
                'num_unpaired_peaks': _unpaired_peaks_count(self.signal1) + _unpaired_peaks_count(self.signal2),
                'tension_guidance': self.tension_guidance,
            }
        except Exception:
            return {
                'similarity_factor': None,
                'mhi': None,
                'belt1': None,
                'belt2': None,
                'paired_peak_deltas_hz': [],
                'num_unpaired_peaks': None,
                'tension_guidance': None,
            }


def _belt_signal_summary(name: Optional[str], signal: SignalData) -> Dict[str, Any]:
    """Build the JSON-safe {name, peak_freq, peak_amplitude} summary for one belt signal,
    peak_freq/peak_amplitude are taken from the highest-amplitude detected peak"""
    peak_freq = None
    peak_amplitude = None
    try:
        peaks = signal.peaks
        if peaks is not None and len(peaks) > 0:
            best_idx = int(np.argmax(signal.psd[peaks]))
            peak_idx = int(peaks[best_idx])
            peak_freq = _safe_float(signal.freqs[peak_idx])
            peak_amplitude = _safe_float(signal.psd[peak_idx])
    except Exception:
        peak_freq = None
        peak_amplitude = None
    return {'name': name, 'peak_freq': peak_freq, 'peak_amplitude': peak_amplitude}


PairedPeaks = Optional[List[Tuple[Tuple[int, float, float], Tuple[int, float, float]]]]


def _paired_peak_deltas_hz(paired_peaks: PairedPeaks) -> List[float]:
    """Compute |f1 - f2| for every paired peak, from ((idx, freq, psd), (idx, freq, psd)) tuples"""
    deltas = []
    try:
        for p1, p2 in paired_peaks or []:
            deltas.append(abs(float(p1[1]) - float(p2[1])))
    except Exception:
        return []
    return deltas


def _unpaired_peaks_count(signal: SignalData) -> int:
    try:
        return len(signal.unpaired_peaks) if signal.unpaired_peaks is not None else 0
    except Exception:
        return 0


@dataclass
class StaticFrequencyResult(ComputationResult):
    """Result from static frequency computation"""

    freq: Optional[float]
    duration: Optional[float]
    accel_per_hz: Optional[float]
    t: np.ndarray
    bins: np.ndarray
    pdata: np.ndarray
    max_freq: float

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'freq': self.freq,
            'duration': self.duration,
            'accel_per_hz': self.accel_per_hz,
            'st_version': self.metadata.version,
            'measurements': self.measurements,
            't': self.t,
            'bins': self.bins,
            'pdata': self.pdata,
            'max_freq': self.max_freq,
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            energy = _safe_float(np.sum(self.pdata)) if self.pdata is not None else None
        except Exception:
            energy = None
        return {
            'freq': _safe_float(self.freq),
            'duration': _safe_float(self.duration),
            'accel_per_hz': _safe_float(self.accel_per_hz),
            'energy': energy,
        }


@dataclass
class ShaperResult(ComputationResult):
    """Result from input shaper computation"""

    calibration_data: Any  # CalibrationData type
    shapers: List[Any]  # List of shaper objects
    shaper_table_data: Dict[str, Any]
    shaper_choices: List[str]
    peaks: np.ndarray
    peaks_freqs: np.ndarray
    peaks_threshold: Tuple[float, float]
    fr: float
    zeta: float
    t: np.ndarray
    bins: np.ndarray
    pdata: np.ndarray
    test_params: Any
    max_smoothing: Optional[float]
    scv: float
    max_freq: float
    max_scale: Optional[float]
    compat: bool = False
    max_smoothing_computed: Optional[float] = None
    damping_ratio_measured: bool = False
    low_vibration_shaper: Optional[Dict[str, Any]] = None
    performance_shaper: Optional[Dict[str, Any]] = None

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'measurements': self.measurements,
            'compat': self.compat,
            'max_smoothing_computed': self.max_smoothing_computed,
            'max_freq': self.max_freq,
            'calibration_data': self.calibration_data,
            'shapers': self.shapers,
            'shaper_table_data': self.shaper_table_data,
            'shaper_choices': self.shaper_choices,
            'peaks': self.peaks,
            'peaks_freqs': self.peaks_freqs,
            'peaks_threshold': self.peaks_threshold,
            'fr': self.fr,
            'zeta': self.zeta,
            't': self.t,
            'bins': self.bins,
            'pdata': self.pdata,
            'test_params': self.test_params,
            'max_smoothing': self.max_smoothing,
            'scv': self.scv,
            'st_version': self.metadata.version,
            'max_scale': self.max_scale,
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            axis = 'unknown'
            if self.measurements:
                axis = _infer_axis_from_name(self.measurements[0].get('name', ''))

            peaks_freqs = self.peaks_freqs
            peak_freqs = [_safe_float(f) for f in peaks_freqs] if peaks_freqs is not None else []

            return {
                'axis': axis,
                'fr': _safe_float(self.fr),
                'zeta': _safe_float(self.zeta),
                'scv': _safe_float(self.scv),
                'compat': bool(self.compat),
                'shaper_choices': list(self.shaper_choices) if self.shaper_choices else [],
                'num_peaks': len(self.peaks) if self.peaks is not None else 0,
                'peak_freqs': peak_freqs,
                'damping_ratio_measured': bool(self.damping_ratio_measured),
                'recommendations': {
                    'low_vibration': self.low_vibration_shaper,
                    'performance': self.performance_shaper,
                },
            }
        except Exception:
            return {
                'axis': 'unknown',
                'fr': None,
                'zeta': None,
                'scv': None,
                'compat': None,
                'shaper_choices': [],
                'num_peaks': 0,
                'peak_freqs': [],
                'damping_ratio_measured': False,
                'recommendations': {'low_vibration': None, 'performance': None},
            }


@dataclass
class VibrationsResult(ComputationResult):
    """Result from vibrations analysis computation"""

    all_speeds: np.ndarray
    all_angles: np.ndarray
    all_angles_energy: Dict[float, np.ndarray]
    good_speeds: np.ndarray
    good_angles: np.ndarray
    kinematics: str
    accel: float
    motors: Optional[List[Any]]  # Motor objects
    motors_config_differences: Optional[str]
    symmetry_factor: float
    spectrogram_data: np.ndarray
    sp_min_energy: float
    sp_max_energy: float
    sp_variance_energy: float
    vibration_metric: float
    num_peaks: int
    vibration_peaks: List[Tuple[float, float, float, float]]
    target_freqs: List[Tuple[str, List[float]]]
    main_angles: List[float]
    global_motor_profile: Optional[Tuple[str, Tuple[float, float]]]
    motor_profiles: Optional[List[Tuple[str, Tuple[float, float]]]]
    max_freq: float
    motor_fr: Optional[float]
    motor_zeta: Optional[float]
    motor_res_idx: Optional[int]
    stealthchop_findings: List[Dict[str, Any]] = field(default_factory=list)

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'measurements': self.measurements,
            'all_speeds': self.all_speeds,
            'all_angles': self.all_angles,
            'all_angles_energy': self.all_angles_energy,
            'good_speeds': self.good_speeds,
            'good_angles': self.good_angles,
            'kinematics': self.kinematics,
            'accel': self.accel,
            'motors': self.motors,
            'motors_config_differences': self.motors_config_differences,
            'symmetry_factor': self.symmetry_factor,
            'spectrogram_data': self.spectrogram_data,
            'sp_min_energy': self.sp_min_energy,
            'sp_max_energy': self.sp_max_energy,
            'sp_variance_energy': self.sp_variance_energy,
            'vibration_metric': self.vibration_metric,
            'num_peaks': self.num_peaks,
            'vibration_peaks': self.vibration_peaks,
            'target_freqs': self.target_freqs,
            'main_angles': self.main_angles,
            'global_motor_profile': self.global_motor_profile,
            'motor_profiles': self.motor_profiles,
            'max_freq': self.max_freq,
            'motor_fr': self.motor_fr,
            'motor_zeta': self.motor_zeta,
            'motor_res_idx': self.motor_res_idx,
            'stealthchop_findings': self.stealthchop_findings,
            'st_version': self.metadata.version,
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            peak_speeds = []
            for idx in self.vibration_peaks if self.vibration_peaks is not None else []:
                try:
                    peak_speeds.append(_safe_float(self.all_speeds[int(idx)]))
                except Exception:
                    continue

            good_speeds = []
            for speed_range in self.good_speeds if self.good_speeds is not None else []:
                try:
                    start_idx, end_idx = int(speed_range[0]), int(speed_range[1])
                    good_speeds.append([_safe_float(self.all_speeds[start_idx]), _safe_float(self.all_speeds[end_idx])])
                except Exception:
                    continue

            stealthchop_findings = []
            for finding in self.stealthchop_findings or []:
                try:
                    stealthchop_findings.append(
                        {
                            'motor': finding.get('motor'),
                            'threshold_mm_s': _safe_float(finding.get('threshold_mm_s')),
                            'worse_side': finding.get('worse_side'),
                        }
                    )
                except Exception:
                    continue

            return {
                'symmetry_factor': _safe_float(self.symmetry_factor),
                'motor_fr': _safe_float(self.motor_fr),
                'motor_zeta': _safe_float(self.motor_zeta),
                'num_peaks': int(self.num_peaks) if self.num_peaks is not None else 0,
                'peak_speeds': peak_speeds,
                'good_speeds': good_speeds,
                'stealthchop_findings': stealthchop_findings,
            }
        except Exception:
            return {
                'symmetry_factor': None,
                'motor_fr': None,
                'motor_zeta': None,
                'num_peaks': 0,
                'peak_speeds': [],
                'good_speeds': [],
                'stealthchop_findings': [],
            }


@dataclass
class HealthCheckResult(ComputationResult):
    """Result from the quick per-axis healthcheck sweep computation"""

    per_axis: Dict[str, Any]
    baseline: Optional[Dict[str, Any]]
    max_freq: float

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'per_axis': self.per_axis,
            'baseline': self.baseline,
            'max_freq': self.max_freq,
            'st_version': self.metadata.version,
        }

    def get_summary(self) -> Dict[str, Any]:
        try:
            return {
                'per_axis': {
                    axis: {
                        'fr': _safe_float(data.get('fr')),
                        'zeta': _safe_float(data.get('zeta')),
                        'peak_freqs': [_safe_float(f) for f in data.get('peak_freqs') or []],
                    }
                    for axis, data in (self.per_axis or {}).items()
                }
            }
        except Exception:
            return {'per_axis': {}}


@dataclass
class TrendResult(ComputationResult):
    """Result from the metrics-history trend computation. Holds no accelerometer measurements
    of its own (it's derived from history.jsonl records) and is never persisted back into the
    history: get_summary() always returns an empty dict"""

    series: Dict[str, Any]
    n_records: int

    def get_plot_data(self) -> Dict[str, Any]:
        return {
            'series': self.series,
            'n_records': self.n_records,
            'st_version': self.metadata.version,
        }

    def get_summary(self) -> Dict[str, Any]:
        return {}
