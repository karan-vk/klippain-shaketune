# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: vibrations_computation.py
# Description: Computation implementation for machine vibrations analysis

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...helpers.accelerometer import Measurement
from ...helpers.common_func import compute_mechanical_parameters, detect_peaks, identify_low_energy_zones, trapezoid
from ...helpers.console_output import ConsoleOutput
from ...helpers.motors_config_parser import Motor
from .. import get_shaper_calibrate_module, process_accelerometer_data_compat
from ..base_models import GraphMetadata
from ..computation_results import VibrationsResult

PEAKS_DETECTION_THRESHOLD = 0.05
PEAKS_RELATIVE_HEIGHT_THRESHOLD = 0.04
CURVE_SIMILARITY_SIGMOID_K = 0.5
SPEEDS_VALLEY_DETECTION_THRESHOLD = 0.7  # Lower is more sensitive
SPEEDS_AROUND_PEAK_DELETION = 3  # to delete +-3mm/s around a peak
ANGLES_VALLEY_DETECTION_THRESHOLD = 1.1  # Lower is more sensitive

# Directional speed spectrogram grid resolution. (N_ANGLES, SPEED_OVERSAMPLING) is the finer grid
# used when the compiled native kernel is available; (_REFERENCE_N_ANGLES, _REFERENCE_SPEED_OVERSAMPLING)
# is the original grid used as a fallback. `_resolve_grid()` selects between them per run: the finer
# grid is only used with native acceleration, because the pure-Python projection is
# O(n_angles * m * speed_oversampling) and the finer grid would be ~4x slower on a host without
# native (e.g. a Pi Zero whose prebuilt binary is missing/stale) -- there we keep the original grid
# so performance is unchanged.
#
# The chosen grid is passed through the FFI on every native call, so a stale native binary (built
# against an older, hardcoded grid) can never silently produce a differently-shaped result than the
# Python fallback: an old 4-arg binary raises TypeError, which the dispatch `except Exception`
# routes to the Python fallback at the same grid (and a stale binary also fails the loader's
# tree-hash check, so `_resolve_grid` then picks the reference grid anyway).
#
# All index-unit smoothing windows below (_compute_angle_powers, _compute_speed_powers, detect_peaks)
# are rescaled by the exact ratio of new/old index counts so their PHYSICAL smoothing width (in
# degrees or mm/s) is preserved. At the reference grid the ratio is exactly 1.0 (bit-identical to the
# original hardcoded windows). At the finer grid the physics is preserved to within one grid step:
# for well-separated peaks the detected-peak COUNT is unchanged; a borderline shoulder sitting right
# on the detection threshold can differ by one (the finer grid errs toward MERGING it, i.e. reporting
# fewer peaks -- it never invents new peaks), so higher resolution does not add spurious peaks.
N_ANGLES = 1440
SPEED_OVERSAMPLING = 12
_REFERENCE_N_ANGLES = 720
_REFERENCE_SPEED_OVERSAMPLING = 6

# stealthChop->spreadCycle transition is only flagged as a likely cause of extra vibrations when the
# energy difference across it is large: this is a causal claim, so we're deliberately conservative here
STEALTHCHOP_ENERGY_RATIO_THRESHOLD = 1.6  # require >=60% more mean energy on one side of the transition
STEALTHCHOP_RANGE_MARGIN_FRACTION = 0.10  # transition must sit >=10% of the tested-speed span inside the range


def _round_to_odd(x: float) -> int:
    """Round x to the NEAREST odd integer (>=1). Convolution kernels and padding widths need an odd
    size to stay centered; picking the nearest odd number (rather than always rounding up) keeps the
    physical smoothing width as close as possible to the ideal, minimizing any resolution-dependent
    drift in which borderline peaks survive detection."""
    n = max(1, int(round(x)))
    if n % 2 == 1:
        return n
    lower, upper = n - 1, n + 1
    if lower < 1:
        return upper
    return lower if abs(x - lower) <= abs(x - upper) else upper


def _angle_window_ratio(n_angles: int) -> float:
    """Ratio of the given/reference angle index counts, used to rescale angle-domain smoothing
    windows so their PHYSICAL (degrees) width stays constant regardless of the grid resolution."""
    return (n_angles - 1) / (_REFERENCE_N_ANGLES - 1)


def _speed_window_ratio(m: int, speed_oversampling: int) -> float:
    """Ratio of the given/reference speed index counts for a test with `m` measured speeds, used to
    rescale speed-domain smoothing windows so their PHYSICAL (mm/s) width stays constant regardless
    of the speed oversampling factor."""
    old = m * _REFERENCE_SPEED_OVERSAMPLING
    new = m * speed_oversampling
    return 1.0 if old <= 1 else (new - 1) / (old - 1)


class VibrationsComputation:
    """Computation for machine vibrations analysis"""

    def __init__(
        self,
        measurements: List[Measurement],
        kinematics: str,
        accel: float,
        max_freq: float,
        motors: Optional[List[Motor]],
        st_version: str,
    ):
        self.measurements = measurements
        self.kinematics = kinematics
        self.accel = accel
        self.max_freq = max_freq
        self.motors = motors
        self.st_version = st_version
        # Spectrogram grid; resolved per-run in compute() via _resolve_grid(). Defaulted here so
        # the grid-dependent helpers are safe if a method is exercised directly (e.g. in tests).
        self._n_angles = N_ANGLES
        self._speed_oversampling = SPEED_OVERSAMPLING

    @staticmethod
    def _resolve_grid() -> Tuple[int, int]:
        """Choose (n_angles, speed_oversampling) for this run. The finer grid is only used when the
        compiled native kernel is available: the pure-Python fallback of the projection is
        O(n_angles * m * speed_oversampling), so the finer grid would be ~4x slower on a host without
        native acceleration (e.g. a Pi Zero whose prebuilt binary is missing or stale). On such hosts
        the original (720, 6) grid is kept so performance is unchanged; where native is present the
        finer (1440, 12) grid costs only a few extra milliseconds."""
        try:
            from ...native import get_native

            if get_native() is not None:
                return N_ANGLES, SPEED_OVERSAMPLING
        except Exception:
            pass
        return _REFERENCE_N_ANGLES, _REFERENCE_SPEED_OVERSAMPLING

    def compute(self) -> VibrationsResult:
        """Perform vibrations analysis computation"""
        if self.kinematics in {'cartesian', 'limited_cartesian', 'corexz', 'limited_corexz'}:
            main_angles = [0, 90]
        elif self.kinematics in {'corexy', 'limited_corexy'}:
            main_angles = [45, 135]
        else:
            raise ValueError('Only Cartesian, CoreXY and CoreXZ kinematics are supported by this tool at the moment!')

        # Resolve the spectrogram grid once per run (finer only when native acceleration is available)
        self._n_angles, self._speed_oversampling = self._resolve_grid()

        psds = {}
        psds_sum = {}
        target_freqs_initialized = False
        target_freqs = None

        shaper_calibrate, _ = get_shaper_calibrate_module()

        for measurement in self.measurements:
            data = np.asarray(measurement['samples'], dtype=np.float64)
            if data is None:
                continue  # Measurement data is not in the expected format or is empty, skip it

            angle, speed = self._extract_angle_and_speed(measurement['name'])
            freq_response = process_accelerometer_data_compat(shaper_calibrate, data)
            first_freqs = freq_response.freq_bins
            psd_sum = freq_response.psd_sum

            if not target_freqs_initialized:
                target_freqs = first_freqs[first_freqs <= self.max_freq]
                target_freqs_initialized = True

            psd_sum = psd_sum[first_freqs <= self.max_freq]
            first_freqs = first_freqs[first_freqs <= self.max_freq]

            # Initialize the angle dictionary if it doesn't exist
            if angle not in psds:
                psds[angle] = {}
                psds_sum[angle] = {}

            # Store the interpolated PSD and integral values
            psds[angle][speed] = np.interp(target_freqs, first_freqs, psd_sum)
            psds_sum[angle][speed] = trapezoid(psd_sum, first_freqs)

        measured_angles = sorted(psds_sum.keys())
        measured_speeds = sorted({speed for angle_speeds in psds_sum.values() for speed in angle_speeds.keys()})

        for main_angle in main_angles:
            if main_angle not in measured_angles:
                raise ValueError('Measurements not taken at the correct angles for the specified kinematics!')

        # Precompute the variables used in plot functions
        all_angles, all_speeds, spectrogram_data = self._compute_dir_speed_spectrogram(
            measured_speeds, psds_sum, self.kinematics, main_angles
        )
        all_angles_energy = self._compute_angle_powers(spectrogram_data)
        # The speed grid is per-test (m depends on how many speeds were measured), so the
        # smoothing/detection windows below are rescaled at runtime from m rather than hardcoded,
        # pinning their PHYSICAL (mm/s) width to what they were at the reference oversampling
        # factor (6x -> smoothing_window=15, detect_peaks window/vicinity=10).
        m = len(measured_speeds)
        speed_ratio = _speed_window_ratio(m, self._speed_oversampling)
        sp_min_energy, sp_max_energy, sp_variance_energy, vibration_metric = self._compute_speed_powers(
            spectrogram_data, smoothing_window=_round_to_odd(15 * speed_ratio)
        )
        motor_profiles, global_motor_profile = self._compute_motor_profiles(
            target_freqs, psds, all_angles_energy, main_angles
        )

        symmetry_factor = self._compute_symmetry_analysis(all_angles, spectrogram_data, main_angles)
        ConsoleOutput.print(f'Machine estimated vibration symmetry: {symmetry_factor:.1f}%')

        # Analyze low variance ranges of vibration energy across all angles for each speed to identify clean speeds
        # and highlight them. Also find the peaks to identify speeds to avoid due to high resonances
        detect_peaks_window = max(1, int(round(10 * speed_ratio)))
        num_peaks, vibration_peaks, peaks_speeds = detect_peaks(
            vibration_metric,
            all_speeds,
            PEAKS_DETECTION_THRESHOLD * vibration_metric.max(),
            PEAKS_RELATIVE_HEIGHT_THRESHOLD,
            detect_peaks_window,
            detect_peaks_window,
        )
        formated_peaks_speeds = ['{:.1f}'.format(pspeed) for pspeed in peaks_speeds]
        ConsoleOutput.print(
            f'Vibrations peaks detected: {num_peaks} @ {", ".join(map(str, formated_peaks_speeds))} mm/s (avoid setting a speed near these values in your slicer print profile)'
        )

        good_speeds = identify_low_energy_zones(vibration_metric, SPEEDS_VALLEY_DETECTION_THRESHOLD)
        if good_speeds is not None:
            deletion_range = int(SPEEDS_AROUND_PEAK_DELETION / (all_speeds[1] - all_speeds[0]))
            peak_speed_indices = {pspeed: np.where(all_speeds == pspeed)[0][0] for pspeed in set(peaks_speeds)}

            # Filter and split ranges based on peak indices, avoiding overlaps
            good_speeds = self._filter_and_split_ranges(all_speeds, good_speeds, peak_speed_indices, deletion_range)

            # Add some logging about the good speeds found
            ConsoleOutput.print(f'Lowest vibrations speeds ({len(good_speeds)} ranges sorted from best to worse):')
            for idx, (start, end, _) in enumerate(good_speeds):
                ConsoleOutput.print(f'{idx + 1}: {all_speeds[start]:.1f} to {all_speeds[end]:.1f} mm/s')

        # Angle low energy valleys identification (good angles ranges) and print them to the console
        good_angles = identify_low_energy_zones(all_angles_energy, ANGLES_VALLEY_DETECTION_THRESHOLD)
        if good_angles is not None:
            ConsoleOutput.print(f'Lowest vibrations angles ({len(good_angles)} ranges sorted from best to worse):')
            for idx, (start, end, energy) in enumerate(good_angles):
                ConsoleOutput.print(
                    f'{idx + 1}: {all_angles[start]:.1f}° to {all_angles[end]:.1f}° (mean vibrations energy: {energy:.2f}% of max)'
                )

        # Motors infos and config differences check
        if self.motors is not None and len(self.motors) == 2:
            motors_config_differences = self.motors[0].compare_to(self.motors[1])
            if motors_config_differences is not None and self.kinematics in {'corexy', 'limited_corexy'}:
                ConsoleOutput.print(
                    f'Warning: motors have different TMC configurations! Differing: '
                    f'{self._format_motor_diff(motors_config_differences)}'
                )
        else:
            motors_config_differences = None

        # Check if the stealthChop->spreadCycle transition threshold falls inside the tested speed
        # range and correlates with a significant vibration energy jump across it
        stealthchop_findings = self._check_stealthchop_transitions(all_speeds, sp_max_energy)
        for finding in stealthchop_findings:
            ConsoleOutput.print(finding['message'])

        # Compute mechanical parameters and check the main resonant frequency of motors
        motor_fr, motor_zeta, motor_res_idx, lowfreq_max = compute_mechanical_parameters(
            global_motor_profile, target_freqs, 30
        )
        if lowfreq_max:
            ConsoleOutput.print(
                '[WARNING] There are a lot of low frequency vibrations that can alter the readings. This is probably due to the test being performed at too high an acceleration!'
            )
            ConsoleOutput.print(
                'Try lowering the ACCEL value and/or increasing the SIZE value before restarting the macro to ensure that only constant speeds are being recorded and that the dynamic behavior of the machine is not affecting the measurements'
            )
        if motor_zeta is not None:
            ConsoleOutput.print(
                f'Motors have a main resonant frequency at {motor_fr:.1f}Hz with an estimated damping ratio of {motor_zeta:.3f}'
            )
        else:
            ConsoleOutput.print(
                f'Motors have a main resonant frequency at {motor_fr:.1f}Hz but it was impossible to estimate a damping ratio.'
            )

        # Create metadata
        metadata = GraphMetadata(
            title='MACHINE VIBRATIONS ANALYSIS TOOL',
            version=self.st_version,
            additional_info={
                'kinematics': self.kinematics,
                'accel': self.accel,
            },
        )

        return VibrationsResult(
            metadata=metadata,
            measurements=self.measurements,
            all_speeds=all_speeds,
            all_angles=all_angles,
            all_angles_energy=all_angles_energy,
            good_speeds=good_speeds,
            good_angles=good_angles,
            kinematics=self.kinematics,
            accel=self.accel,
            motors=self.motors,
            motors_config_differences=motors_config_differences,
            symmetry_factor=symmetry_factor,
            spectrogram_data=spectrogram_data,
            sp_min_energy=sp_min_energy,
            sp_max_energy=sp_max_energy,
            sp_variance_energy=sp_variance_energy,
            vibration_metric=vibration_metric,
            motor_profiles=motor_profiles,
            global_motor_profile=global_motor_profile,
            num_peaks=num_peaks,
            vibration_peaks=vibration_peaks,
            target_freqs=target_freqs,
            main_angles=main_angles,
            max_freq=self.max_freq,
            motor_fr=motor_fr,
            motor_zeta=motor_zeta,
            motor_res_idx=motor_res_idx,
            stealthchop_findings=stealthchop_findings,
        )

    def _compute_motor_profiles(
        self,
        freqs: np.ndarray,
        psds: dict,
        all_angles_energy: dict,
        measured_angles: Optional[List[int]] = None,
        energy_amplification_factor: int = 2,
    ) -> Tuple[dict, np.ndarray]:
        """Calculate motor frequency profiles based on the measured Power Spectral Density (PSD) measurements"""
        if measured_angles is None:
            measured_angles = [0, 90]

        motor_profiles = {}
        weighted_sum_profiles = np.zeros_like(freqs)
        total_weight = 0
        conv_filter = np.ones(20) / 20

        # Creating the PSD motor profiles for each angles
        for angle in measured_angles:
            # Calculate the sum of PSDs for the current angle and then convolve
            sum_curve = np.sum(np.array([psds[angle][speed] for speed in psds[angle]]), axis=0)
            motor_profiles[angle] = np.convolve(sum_curve / len(psds[angle]), conv_filter, mode='same')

            # Calculate weights
            angle_energy = (
                all_angles_energy[angle] ** energy_amplification_factor
            )  # First weighting factor is based on the total vibrations of the machine at the specified angle
            curve_area = (
                trapezoid(motor_profiles[angle], freqs) ** energy_amplification_factor
            )  # Additional weighting factor is based on the area under the current motor profile at this specified angle
            total_angle_weight = angle_energy * curve_area

            # Update weighted sum profiles to get the global motor profile
            weighted_sum_profiles += motor_profiles[angle] * total_angle_weight
            total_weight += total_angle_weight

        # Creating a global average motor profile that is the weighted average of all the PSD motor profiles
        global_motor_profile = weighted_sum_profiles / total_weight if total_weight != 0 else weighted_sum_profiles

        return motor_profiles, global_motor_profile

    def _compute_dir_speed_spectrogram(
        self,
        measured_speeds: List[float],
        data: dict,
        kinematics: str = 'cartesian',
        measured_angles: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute directional speed spectrogram using trigonometry to project motor vibrations"""
        if measured_angles is None:
            measured_angles = [0, 90]

        nat = None
        try:
            from ...native import get_native

            nat = get_native()
        except Exception:
            nat = None
        # Only dispatch to the native kernel for the kinematics the pure-Python loop below
        # explicitly handles (an unknown value must keep its original failure mode)
        if nat is not None and kinematics in {
            'cartesian',
            'limited_cartesian',
            'corexz',
            'limited_corexz',
            'corexy',
            'limited_corexy',
        }:
            ms = np.asarray(measured_speeds, dtype=np.float64)
            a0, a1 = measured_angles[0], measured_angles[1]
            vibs_a = np.array([data[a0].get(s, 0.0) for s in measured_speeds], dtype=np.float64)
            vibs_b = np.array([data[a1].get(s, 0.0) for s in measured_speeds], dtype=np.float64)
            corexy = kinematics in {'corexy', 'limited_corexy'}
            try:
                return nat.dir_speed_spectrogram(ms, vibs_a, vibs_b, corexy, self._n_angles, self._speed_oversampling)
            except Exception:
                pass  # fall through to the pure-Python implementation below

        # We want to project the motor vibrations measured on their own axes on the [0, 360] range
        spectrum_angles = np.linspace(0, 360, self._n_angles)
        spectrum_speeds = np.linspace(
            min(measured_speeds), max(measured_speeds), len(measured_speeds) * self._speed_oversampling
        )
        spectrum_vibrations = np.zeros((len(spectrum_angles), len(spectrum_speeds)))

        def get_interpolated_vibrations(data: dict, speed: float, speeds: List[float]) -> float:
            idx = np.clip(np.searchsorted(speeds, speed, side='left'), 1, len(speeds) - 1)
            lower_speed = speeds[idx - 1]
            upper_speed = speeds[idx]
            lower_vibrations = data.get(lower_speed, 0)
            upper_vibrations = data.get(upper_speed, 0)
            return lower_vibrations + (speed - lower_speed) * (upper_vibrations - lower_vibrations) / (
                upper_speed - lower_speed
            )

        # Precompute trigonometric values and constant before the loop
        angle_radians = np.deg2rad(spectrum_angles)
        cos_vals = np.cos(angle_radians)
        sin_vals = np.sin(angle_radians)
        sqrt_2_inv = 1 / math.sqrt(2)

        # Compute the spectrum vibrations for each angle and speed combination
        for target_angle_idx, (cos_val, sin_val) in enumerate(zip(cos_vals, sin_vals)):
            for target_speed_idx, target_speed in enumerate(spectrum_speeds):
                if kinematics in {'cartesian', 'limited_cartesian', 'corexz', 'limited_corexz'}:
                    speed_1 = np.abs(target_speed * cos_val)
                    speed_2 = np.abs(target_speed * sin_val)
                elif kinematics in {'corexy', 'limited_corexy'}:
                    speed_1 = np.abs(target_speed * (cos_val + sin_val) * sqrt_2_inv)
                    speed_2 = np.abs(target_speed * (cos_val - sin_val) * sqrt_2_inv)

                vibrations_1 = get_interpolated_vibrations(data[measured_angles[0]], speed_1, measured_speeds)
                vibrations_2 = get_interpolated_vibrations(data[measured_angles[1]], speed_2, measured_speeds)
                spectrum_vibrations[target_angle_idx, target_speed_idx] = vibrations_1 + vibrations_2

        return spectrum_angles, spectrum_speeds, spectrum_vibrations

    def _compute_angle_powers(self, spectrogram_data: np.ndarray) -> np.ndarray:
        """Compute angle powers from spectrogram data"""
        angles_powers = trapezoid(spectrogram_data, axis=1)

        # The kernel width and padding below are rescaled by the angle-grid ratio so their
        # PHYSICAL (degrees) width stays exactly the same as at the reference resolution (720
        # angles -> kernel_width=15, pad=9); at N_ANGLES == _REFERENCE_N_ANGLES the ratio is 1.0
        # and these are bit-identical to the original hardcoded values.
        ratio = _angle_window_ratio(self._n_angles)
        kernel_width = _round_to_odd(15 * ratio)
        pad = max(kernel_width // 2, int(round(9 * ratio)))

        # Since we want to plot it on a continuous polar plot later on, we need to append parts of
        # the array to start and end of it to smooth transitions when doing the convolution
        # and get the same value at modulo 360. Then we return the array without the extras
        extended_angles_powers = np.concatenate([angles_powers[-pad:], angles_powers, angles_powers[:pad]])
        convolved_extended = np.convolve(extended_angles_powers, np.ones(kernel_width) / kernel_width, mode='same')

        return convolved_extended[pad:-pad]

    def _compute_speed_powers(self, spectrogram_data: np.ndarray, smoothing_window: int = 15) -> np.ndarray:
        """Compute speed powers from spectrogram data"""
        min_values = np.amin(spectrogram_data, axis=0)
        max_values = np.amax(spectrogram_data, axis=0)
        var_values = np.var(spectrogram_data, axis=0)

        # rescale the variance to the same range as max_values to plot it on the same graph
        var_values = var_values / var_values.max() * max_values.max()

        # Create a vibration metric that is the product of the max values and the variance to quantify the best
        # speeds that have at the same time a low global energy level that is also consistent at every angles
        vibration_metric = max_values * var_values

        # utility function to pad and smooth the data avoiding edge effects
        conv_filter = np.ones(smoothing_window) / smoothing_window
        window = int(smoothing_window / 2)

        def pad_and_smooth(data: np.ndarray) -> np.ndarray:
            data_padded = np.pad(data, (window,), mode='edge')
            smoothed_data = np.convolve(data_padded, conv_filter, mode='valid')
            return smoothed_data

        # Stack the arrays and apply padding and smoothing in batch
        data_arrays = np.stack([min_values, max_values, var_values, vibration_metric])
        smoothed_arrays = np.array([pad_and_smooth(data) for data in data_arrays])

        return smoothed_arrays

    def _check_stealthchop_transitions(self, all_speeds: np.ndarray, sp_max_energy: np.ndarray) -> List[Dict[str, Any]]:
        """Correlate each motor's stealthChop->spreadCycle transition velocity (when known) with the
        measured vibration energy to flag transitions that likely add vibrations in the tested range"""
        if self.motors is None:
            return []

        findings = []
        span = float(all_speeds.max() - all_speeds.min())
        if span <= 0:
            return []

        margin = STEALTHCHOP_RANGE_MARGIN_FRACTION * span
        lo = all_speeds.min() + margin
        hi = all_speeds.max() - margin

        for motor in self.motors:
            v_thr = motor.get_config('stealthchop_threshold_mm_s')
            if v_thr is None or not math.isfinite(v_thr) or not (lo <= v_thr <= hi):
                continue

            idx = int(np.searchsorted(all_speeds, v_thr))
            if idx <= 0 or idx >= len(all_speeds):
                continue

            energy_below = float(np.mean(sp_max_energy[:idx]))
            energy_above = float(np.mean(sp_max_energy[idx:]))
            lo_e, hi_e = min(energy_below, energy_above), max(energy_below, energy_above)
            if lo_e <= 0 or hi_e / lo_e < STEALTHCHOP_ENERGY_RATIO_THRESHOLD:
                continue

            ratio = hi_e / lo_e
            # In Klipper, speeds BELOW stealthchop_threshold run in stealthChop and speeds ABOVE it
            # run in spreadCycle (stealthchop_threshold: 0 disables stealthChop -> spreadCycle
            # everywhere). The fix is to push the whole tested range into the QUIETER mode, so the
            # advice must move the transition to the far side of the range from the noisier mode.
            if energy_above > energy_below:
                # spreadCycle (above the transition) is the noisier mode -> keep the range in the
                # quieter stealthChop by raising the threshold above the top tested speed.
                worse_side = 'spreadCycle (above the transition)'
                advice = (
                    f'the quieter mode here is stealthChop -- consider raising stealthchop_threshold '
                    f'above {all_speeds.max():.0f} mm/s so your print speeds stay in stealthChop'
                )
            else:
                # stealthChop (below the transition) is the noisier mode -> keep the range in the
                # quieter spreadCycle by lowering/zeroing the threshold.
                worse_side = 'stealthChop (below the transition)'
                advice = (
                    f'the quieter mode here is spreadCycle -- consider stealthchop_threshold: 0 '
                    f'(spreadCycle always) or lowering it below {all_speeds.min():.0f} mm/s'
                )

            msg = (
                f'{motor.name}: stealthChop->spreadCycle transition at ~{v_thr:.0f} mm/s lies inside your '
                f'tested speed range and vibration energy is {ratio:.1f}x higher on the {worse_side} side; {advice}'
            )
            findings.append(
                {
                    'motor': motor.name,
                    'threshold_mm_s': float(v_thr),
                    'energy_ratio': float(ratio),
                    'worse_side': worse_side,
                    'message': msg,
                }
            )

        return findings

    def _format_motor_diff(self, differences: Dict[str, Dict[str, Any]]) -> str:
        """Flatten a Motor.compare_to() diff dict into a short, human-readable comma-joined list"""
        parts = []
        for key, value in differences.get('config', {}).items():
            parts.append(f'{key}={value}')
        for register, reg_diffs in differences.get('registers', {}).items():
            for field_name, value in reg_diffs.items():
                parts.append(f'{register}.{field_name}={value}')
        return ', '.join(parts) if parts else 'unknown'

    def _filter_and_split_ranges(
        self,
        all_speeds: np.ndarray,
        good_speeds: List[Tuple[int, int, float]],
        peak_speed_indices: dict,
        deletion_range: int,
    ) -> List[Tuple[int, int, float]]:
        """Filter and split the good_speed ranges"""
        # Process each range to filter out and split based on peak indices
        filtered_good_speeds = []
        for start, end, energy in good_speeds:
            start_speed, end_speed = all_speeds[start], all_speeds[end]
            # Identify peaks that intersect with the current speed range
            intersecting_peaks_indices = [
                idx for speed, idx in peak_speed_indices.items() if start_speed <= speed <= end_speed
            ]

            if not intersecting_peaks_indices:
                filtered_good_speeds.append((start, end, energy))
            else:
                intersecting_peaks_indices.sort()
                current_start = start

                for peak_index in intersecting_peaks_indices:
                    before_peak_end = max(current_start, peak_index - deletion_range)
                    if current_start < before_peak_end:
                        filtered_good_speeds.append((current_start, before_peak_end, energy))
                    current_start = peak_index + deletion_range + 1

                if current_start < end:
                    filtered_good_speeds.append((current_start, end, energy))

        # Sorting by start point once and then merge overlapping ranges
        sorted_ranges = sorted(filtered_good_speeds, key=lambda x: x[0])
        merged_ranges = [sorted_ranges[0]]

        for current in sorted_ranges[1:]:
            last_merged_start, last_merged_end, last_merged_energy = merged_ranges[-1]
            if current[0] <= last_merged_end:
                new_end = max(last_merged_end, current[1])
                new_energy = min(last_merged_energy, current[2])
                merged_ranges[-1] = (last_merged_start, new_end, new_energy)
            else:
                merged_ranges.append(current)

        return merged_ranges

    def _compute_symmetry_analysis(
        self, all_angles: np.ndarray, spectrogram_data: np.ndarray, measured_angles: Optional[List[int]] = None
    ) -> float:
        """Compute symmetry score that reflects the spectrogram apparent symmetry"""
        if measured_angles is None:
            measured_angles = [0, 90]

        total_spectrogram_angles = len(all_angles)
        half_spectrogram_angles = total_spectrogram_angles // 2

        # Extend the spectrogram by adding half to the beginning (in order to not get an out of bounds error later)
        extended_spectrogram = np.concatenate((spectrogram_data[-half_spectrogram_angles:], spectrogram_data), axis=0)

        # Calculate the split index directly within the slicing
        midpoint_angle = np.mean(measured_angles)
        split_index = int(midpoint_angle * (total_spectrogram_angles / 360) + half_spectrogram_angles)
        half_segment_length = half_spectrogram_angles // 2

        # Slice out the two segments of the spectrogram and flatten them for comparison
        segment_1_flattened = extended_spectrogram[split_index - half_segment_length : split_index].flatten()
        segment_2_flattened = extended_spectrogram[split_index : split_index + half_segment_length].flatten()

        # Compute the correlation coefficient between the two segments of spectrogram
        correlation = np.corrcoef(segment_1_flattened, segment_2_flattened)[0, 1]
        percentage_correlation_biased = (100 * np.power(correlation, 0.75)) + 10

        return np.clip(0, 100, percentage_correlation_biased)

    def _extract_angle_and_speed(self, logname: str) -> Tuple[float, float]:
        """Extract from the measurement name the angle and speed of the tested movement"""
        try:
            match = re.search(r'an(\d+)_\d+sp(\d+)_\d+', os.path.basename(logname))
            if match:
                angle = match.group(1)
                speed = match.group(2)
            else:
                raise ValueError(
                    f'File {logname} does not match expected format. Clean your /tmp folder and start again!'
                )
        except AttributeError as err:
            raise ValueError(
                f'File {logname} does not match expected format. Clean your /tmp folder and start again!'
            ) from err
        return float(angle), float(speed)
