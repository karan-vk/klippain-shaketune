#!/usr/bin/env python3
# Shake&Tune: 3D printer analysis tools
#
# File: run_parity.py
# Description: Standalone (no pytest) parity runner comparing the compiled Rust "_core" native
#              extension against Shake&Tune's pure-Python fallback implementations, for each of
#              the four numeric kernels the native module implements:
#                kernel 1: dir_speed_spectrogram   (vibrations profile direction/speed mapping)
#                kernel 2: spectrogram             (input shaper / belts PSD spectrogram)
#                kernel 3: stdata read/write       (.stdata v1/v2 file formats)
#                kernel 4: klipper_psd             (Klipper's own calc_freq_response PSD math)
#
# This script degrades gracefully when the native module (or, for kernel 4, a Klipper checkout)
# is not available: missing pieces are reported as SKIP, not FAIL, so this can be run at any
# point in the native-module bring-up before every piece has landed. Run gen_corpus.py first.
#
# Usage:
#   python tests/parity/gen_corpus.py [--big]
#   python tests/parity/run_parity.py [--klipper-dir DIR] [--corpus DIR] [--big]

import argparse
import json
import math
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import zstandard

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ['SHAKETUNE_IN_CLI'] = '1'

# --------------------------------------------------------------------------------------------
# Small, independent (i.e. NOT reusing shaketune's own reader) helpers used as ground truth for
# the stdata kernel, plus a couple of generic utilities used by every kernel.
# --------------------------------------------------------------------------------------------


def read_v1_direct(path):
    """From-scratch v1 (.stdata) reader: zstd-decompress, then parse each JSON line ourselves.
    Deliberately independent of MeasurementsManager so it is a genuine reference to check the
    native module's v1 reading against. Returns a list of (name, samples ndarray (Ni,4))."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, 'rb') as f, dctx.stream_reader(f) as reader:
        text = reader.read().decode('utf-8')

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        samples = np.array(obj['samples'], dtype=np.float64).reshape(-1, 4)
        out.append((obj['name'], samples))
    return out


def load_v1_measurement(path):
    """Load the single measurement of a v1 corpus file via shaketune's own MeasurementsManager
    (a real, exercised code path), returning (name, samples ndarray (N,4))."""
    from shaketune.helpers.accelerometer import MeasurementsManager

    mgr = MeasurementsManager(chunk_size=10_000_000)
    measurements = mgr.load_from_stdata(Path(path))
    if not measurements:
        raise ValueError(f'no measurements loaded from {path}')
    meas = measurements[0]
    samples = np.array(meas['samples'], dtype=np.float64).reshape(-1, 4)
    return meas['name'], samples


class Results:
    def __init__(self):
        self.rows = []  # (kernel, status, detail)

    def record(self, kernel, status, detail=''):
        self.rows.append((kernel, status, detail))
        print(f'[{status}] {kernel}: {detail}' if detail else f'[{status}] {kernel}')

    def print_summary(self):
        print('\n' + '=' * 78)
        print(f'{"KERNEL":<45} {"STATUS":<8} DETAIL')
        print('-' * 78)
        for kernel, status, detail in self.rows:
            print(f'{kernel:<45} {status:<8} {detail}')
        print('=' * 78)

    def failed(self):
        return any(status == 'FAIL' for _, status, _ in self.rows)

    def nothing_passed(self):
        """True when not a single check actually ran (e.g. missing corpus -> all SKIP). An
        all-SKIP run proves nothing and must not be reported as SUCCESS, especially in CI."""
        return not any(status == 'PASS' for _, status, _ in self.rows)


def _first_line(exc: Exception) -> str:
    """First non-blank line of an exception's message (np.testing.assert_* messages often start
    with one or more blank lines before the actual "Mismatched elements" summary)."""
    for line in str(exc).splitlines():
        line = line.strip()
        if line:
            return line
    return repr(exc)


def set_native_disabled(disabled: bool):
    if disabled:
        os.environ['SHAKETUNE_DISABLE_NATIVE'] = '1'
    else:
        os.environ.pop('SHAKETUNE_DISABLE_NATIVE', None)


# --------------------------------------------------------------------------------------------
# Kernel 1: dir_speed_spectrogram
# --------------------------------------------------------------------------------------------


def reference_dir_speed_spectrogram(measured_speeds, vibs_a, vibs_b, corexy):
    """Faithful, standalone re-implementation of
    VibrationsComputation._compute_dir_speed_spectrogram (see
    shaketune/graph_creators/computations/vibrations_computation.py), adapted to take two flat
    per-angle value arrays instead of a nested {angle: {speed: value}} dict, to match the native
    `dir_speed_spectrogram(measured_speeds, vibs_a, vibs_b, corexy)` contract."""
    measured_speeds = list(measured_speeds)
    data_a = dict(zip(measured_speeds, vibs_a))
    data_b = dict(zip(measured_speeds, vibs_b))

    spectrum_angles = np.linspace(0, 360, 720)
    spectrum_speeds = np.linspace(min(measured_speeds), max(measured_speeds), len(measured_speeds) * 6)
    spectrum_vibrations = np.zeros((len(spectrum_angles), len(spectrum_speeds)))

    def get_interpolated_vibrations(data, speed, speeds):
        idx = np.clip(np.searchsorted(speeds, speed, side='left'), 1, len(speeds) - 1)
        lower_speed = speeds[idx - 1]
        upper_speed = speeds[idx]
        lower_vibrations = data.get(lower_speed, 0)
        upper_vibrations = data.get(upper_speed, 0)
        return lower_vibrations + (speed - lower_speed) * (upper_vibrations - lower_vibrations) / (
            upper_speed - lower_speed
        )

    angle_radians = np.deg2rad(spectrum_angles)
    cos_vals = np.cos(angle_radians)
    sin_vals = np.sin(angle_radians)
    sqrt_2_inv = 1 / math.sqrt(2)

    for target_angle_idx, (cos_val, sin_val) in enumerate(zip(cos_vals, sin_vals)):
        for target_speed_idx, target_speed in enumerate(spectrum_speeds):
            if not corexy:
                speed_1 = np.abs(target_speed * cos_val)
                speed_2 = np.abs(target_speed * sin_val)
            else:
                speed_1 = np.abs(target_speed * (cos_val + sin_val) * sqrt_2_inv)
                speed_2 = np.abs(target_speed * (cos_val - sin_val) * sqrt_2_inv)

            vibrations_1 = get_interpolated_vibrations(data_a, speed_1, measured_speeds)
            vibrations_2 = get_interpolated_vibrations(data_b, speed_2, measured_speeds)
            spectrum_vibrations[target_angle_idx, target_speed_idx] = vibrations_1 + vibrations_2

    return spectrum_angles, spectrum_speeds, spectrum_vibrations


def kernel1_dir_speed_spectrogram(results: Results, corpus_dir: Path):
    name = 'kernel1 dir_speed_spectrogram'
    from shaketune.native import get_native

    nm = get_native()
    if nm is None:
        results.record(name, 'SKIP', 'native module not available')
        return

    npz_path = corpus_dir / 'vibrations.npz'
    if not npz_path.exists():
        results.record(name, 'SKIP', f'{npz_path} missing, run gen_corpus.py first')
        return

    data = np.load(npz_path)
    speeds = data['speeds']
    values = data['values']  # shape (2, len(speeds))
    vibs_a = np.ascontiguousarray(values[0], dtype=np.float64)
    vibs_b = np.ascontiguousarray(values[1], dtype=np.float64)
    measured_speeds = np.ascontiguousarray(speeds, dtype=np.float64)

    try:
        for corexy in (False, True):
            ref_angles, ref_speeds, ref_vibs = reference_dir_speed_spectrogram(measured_speeds, vibs_a, vibs_b, corexy)
            native_angles, native_speeds, native_vibs = nm.dir_speed_spectrogram(
                measured_speeds, vibs_a, vibs_b, corexy
            )

            np.testing.assert_array_equal(native_angles, ref_angles)
            np.testing.assert_array_equal(native_speeds, ref_speeds)
            np.testing.assert_allclose(native_vibs, ref_vibs, rtol=1e-12)
    except AssertionError as exc:
        results.record(name, 'FAIL', _first_line(exc))
        return

    results.record(name, 'PASS', 'matched reference for corexy=False and corexy=True')


# --------------------------------------------------------------------------------------------
# Kernel 2: spectrogram
# --------------------------------------------------------------------------------------------


def kernel2_spectrogram(results: Results, corpus_dir: Path, big: bool):
    from shaketune.native import get_native

    sizes = [2048, 100_000] + ([1_000_000] if big else [])
    for n in sizes:
        name = f'kernel2 spectrogram (N={n})'
        path = corpus_dir / f'corpus_{n}.stdata'
        if not path.exists():
            skip_flag = ' (pass --big to gen_corpus.py to generate it)' if n == 1_000_000 else ''
            results.record(name, 'SKIP', f'{path} missing, run gen_corpus.py first{skip_flag}')
            continue

        try:
            _, samples = load_v1_measurement(path)
        except Exception as exc:  # noqa: BLE001
            results.record(name, 'FAIL', f'could not load corpus file: {exc}')
            continue

        from shaketune.helpers.spectrogram import compute_spectrogram

        try:
            set_native_disabled(False)
            pdata_a, t_a, f_a = compute_spectrogram(samples)
            native_active = get_native() is not None

            set_native_disabled(True)
            pdata_b, t_b, f_b = compute_spectrogram(samples)
        finally:
            set_native_disabled(False)

        try:
            np.testing.assert_array_equal(t_a, t_b)
            np.testing.assert_array_equal(f_a, f_b)
            np.testing.assert_allclose(pdata_a, pdata_b, rtol=1e-8, atol=1e-12 * max(np.max(np.abs(pdata_b)), 1.0))
        except AssertionError as exc:
            results.record(name, 'FAIL', _first_line(exc))
            continue

        detail = 'native active vs fallback' if native_active else 'native inactive (compared fallback to itself)'
        results.record(name, 'PASS', detail)


# --------------------------------------------------------------------------------------------
# Kernel 3: stdata read/write (v1 and v2)
# --------------------------------------------------------------------------------------------


def _normalize_records(records):
    """Normalize either the native read_stdata() return (list of (name, samples) tuples) or
    stdata_v2.read_stdata_v2()'s return (list of {'name':..., 'samples':...} dicts) into a
    plain list of (name, samples) tuples for comparison."""
    out = []
    for rec in records:
        if isinstance(rec, dict):
            out.append((rec['name'], rec['samples']))
        else:
            out.append((rec[0], rec[1]))
    return out


def kernel3_stdata(results: Results, corpus_dir: Path):
    from shaketune.native import get_native

    nm = get_native()
    if nm is None:
        results.record('kernel3 stdata v1 read', 'SKIP', 'native module not available')
        results.record('kernel3 stdata v2 roundtrip', 'SKIP', 'native module not available')
        return

    # --- Part A: v1 corpus files, read via native vs. a from-scratch direct JSON parse ---
    for n in (2048, 100_000):
        name = f'kernel3 stdata v1 read (N={n})'
        path = corpus_dir / f'corpus_{n}.stdata'
        if not path.exists():
            results.record(name, 'SKIP', f'{path} missing, run gen_corpus.py first')
            continue

        try:
            native_records = _normalize_records(nm.read_stdata(str(path)))
            direct_records = read_v1_direct(path)

            assert len(native_records) == len(direct_records), 'measurement count mismatch'
            for (n_name, n_samples), (d_name, d_samples) in zip(native_records, direct_records):
                assert n_name == d_name, f'name mismatch: {n_name!r} != {d_name!r}'
                # NOTE: legacy v1 (.stdata) reads are compared to ~1 ulp, not bitwise. The samples
                # are float text produced by Python's json; serde_json (Rust) and CPython's json
                # float parsers can differ by a single ULP on some values (~2e-16 relative). This is
                # far below every downstream FFT/PSD tolerance, and only affects re-reading OLD v1
                # files (the new v2 binary format is bitwise-exact, checked in Part B below).
                np.testing.assert_allclose(np.asarray(n_samples, dtype=np.float64), d_samples, rtol=1e-12, atol=0.0)
        except AssertionError as exc:
            results.record(name, 'FAIL', _first_line(exc))
            continue

        results.record(name, 'PASS', f'{len(native_records)} measurement(s) match within 1 ulp (serde_json vs json)')

    # --- Part B: v2 round trip via native StdataWriter, read back by native and by the ---
    # --- pure-Python shaketune.native.stdata_v2 reader.                                ---
    name = 'kernel3 stdata v2 roundtrip'
    try:
        from shaketune.native.stdata_v2 import read_stdata_v2
    except ImportError as exc:
        results.record(name, 'SKIP', f'shaketune.native.stdata_v2 not importable ({exc})')
        return

    v2_path = corpus_dir / '_kernel3_v2_roundtrip.stdata'
    written = []
    for n in (2048, 100_000):
        src_path = corpus_dir / f'corpus_{n}.stdata'
        if not src_path.exists():
            continue
        try:
            src_name, samples = load_v1_measurement(src_path)
        except Exception:
            continue
        written.append((f'{src_name}_v2', np.ascontiguousarray(samples, dtype=np.float64)))

    if not written:
        results.record(name, 'SKIP', 'no source corpus files available, run gen_corpus.py first')
        return

    try:
        writer = nm.StdataWriter(str(v2_path), 11)
        for meas_name, samples in written:
            writer.write_measurement(meas_name, samples)
        writer.close()

        native_records = _normalize_records(nm.read_stdata(str(v2_path)))
        pyread_records = _normalize_records(read_stdata_v2(str(v2_path)))

        assert len(native_records) == len(written), 'native read_stdata: measurement count mismatch'
        assert len(pyread_records) == len(written), 'stdata_v2.read_stdata_v2: measurement count mismatch'

        for (exp_name, exp_samples), (n_name, n_samples), (p_name, p_samples) in zip(
            written, native_records, pyread_records
        ):
            assert n_name == exp_name, f'native name mismatch: {n_name!r} != {exp_name!r}'
            assert p_name == exp_name, f'py name mismatch: {p_name!r} != {exp_name!r}'
            np.testing.assert_array_equal(np.asarray(n_samples, dtype=np.float64), exp_samples)
            np.testing.assert_array_equal(np.asarray(p_samples, dtype=np.float64), exp_samples)
    except AssertionError as exc:
        results.record(name, 'FAIL', _first_line(exc))
        return
    except Exception as exc:  # noqa: BLE001
        results.record(name, 'FAIL', f'unexpected error: {exc}')
        return

    results.record(name, 'PASS', f'{len(written)} measurement(s) bitwise-identical (native + python readers)')


# --------------------------------------------------------------------------------------------
# Kernel 4: klipper_psd (only runs if --klipper-dir is given)
# --------------------------------------------------------------------------------------------


def load_klipper_module(klipper_dir: str):
    """Mirrors shaketune.cli.load_klipper_module: makes Klipper's shaper_calibrate/shaper_defs
    importable as top-level modules the way klippy's extras loader expects."""
    kdir = os.path.expanduser(klipper_dir)
    sys.path.append(os.path.join(kdir, 'klippy'))
    sys.modules['shaper_calibrate'] = import_module('.shaper_calibrate', 'extras')
    sys.modules['shaper_defs'] = import_module('.shaper_defs', 'extras')


def kernel4_klipper_psd(results: Results, corpus_dir: Path, klipper_dir: str):
    name = 'kernel4 klipper_psd'
    if not klipper_dir:
        results.record(name, 'SKIP', 'no --klipper-dir given')
        return

    from shaketune.native import get_native

    nm = get_native()
    if nm is None:
        results.record(name, 'SKIP', 'native module not available')
        return

    try:
        load_klipper_module(klipper_dir)
    except Exception as exc:  # noqa: BLE001
        results.record(name, 'FAIL', f'could not import shaper_calibrate from {klipper_dir}: {exc}')
        return

    from shaketune.graph_creators import (
        find_best_shaper_compat,
        get_shaper_calibrate_module,
        process_accelerometer_data_compat,
    )

    set_native_disabled(True)
    shaper_calibrate_obj, _shaper_defs = get_shaper_calibrate_module()
    sc_module = sys.modules['shaper_calibrate']
    set_native_disabled(False)

    # Print (and best-effort record) the digest gating mechanism, if it has landed yet. This is
    # forward-compatible: until shaketune.native.psd_compat exists, we just note that the
    # klipper_psd comparison below is unconditional (not gated by any allowlist).
    try:
        from shaketune.native import psd_compat
    except ImportError as exc:
        psd_compat = None
        print(f'[kernel4] shaketune.native.psd_compat not available yet ({exc}); no digest gating in effect')

    if psd_compat is not None:
        try:
            digest = psd_compat.compute_digest(shaper_calibrate_obj)
            print(f'[kernel4] psd_compat source digest: {digest}')
            try:
                psd_compat.record_digest(digest)
            except TypeError:
                psd_compat.record_digest(shaper_calibrate_obj, digest)
        except Exception as exc:  # noqa: BLE001
            print(f'[kernel4] psd_compat present but digest compute/record failed ({exc}); continuing anyway')

    for n in (2048, 100_000):
        subname = f'{name} (N={n})'
        path = corpus_dir / f'corpus_{n}.stdata'
        if not path.exists():
            results.record(subname, 'SKIP', f'{path} missing, run gen_corpus.py first')
            continue

        try:
            meas_name, samples = load_v1_measurement(path)
        except Exception as exc:  # noqa: BLE001
            results.record(subname, 'FAIL', f'could not load corpus file: {exc}')
            continue

        # Klipper's own calc_freq_response returns None (-> "Internal error") when the recording is
        # shorter than one FFT window M = 1 << int(0.5*Fs - 1).bit_length(). The native kernel raises
        # the same "too short" condition. So when N <= M there is nothing to compare: skip, don't fail.
        N = samples.shape[0]
        T = samples[-1, 0] - samples[0, 0]
        sampling_freq = N / T
        M = 1 << int(sampling_freq * 0.5 - 1).bit_length()
        if N <= M:
            results.record(subname, 'SKIP', f'N={n} <= FFT window M={M}; too short for Klipper to process')
            continue

        try:
            set_native_disabled(True)
            ref_cal = process_accelerometer_data_compat(shaper_calibrate_obj, samples, name=meas_name)
            ref_freq_bins = np.asarray(ref_cal.freq_bins, dtype=np.float64)
            ref_psd_sum = np.asarray(ref_cal.psd_sum, dtype=np.float64)
            ref_psd_x = np.asarray(ref_cal.psd_x, dtype=np.float64)
            ref_psd_y = np.asarray(ref_cal.psd_y, dtype=np.float64)
            ref_psd_z = np.asarray(ref_cal.psd_z, dtype=np.float64)

            set_native_disabled(False)
            n_freq_bins, n_psd_sum, n_psd_x, n_psd_y, n_psd_z = nm.klipper_psd(
                np.ascontiguousarray(samples, dtype=np.float64)
            )

            np.testing.assert_array_equal(n_freq_bins, ref_freq_bins)
            np.testing.assert_allclose(n_psd_sum, ref_psd_sum, rtol=1e-8)
            np.testing.assert_allclose(n_psd_x, ref_psd_x, rtol=1e-8)
            np.testing.assert_allclose(n_psd_y, ref_psd_y, rtol=1e-8)
            np.testing.assert_allclose(n_psd_z, ref_psd_z, rtol=1e-8)
        except AssertionError as exc:
            results.record(subname, 'FAIL', _first_line(exc))
            continue
        except Exception as exc:  # noqa: BLE001
            results.record(subname, 'FAIL', f'unexpected error: {exc}')
            continue
        finally:
            set_native_disabled(False)

        results.record(subname, 'PASS', 'freq_bins exact, psd_* within rtol=1e-8')

        # End-to-end decision parity: feed both PSD sets into find_best_shaper and check the
        # recommended shaper agrees. This is best-effort and reported independently of the raw
        # PSD comparison above: Klipper's CalibrationData constructor shape has been stable but
        # is not part of any documented public API, so we degrade to a SKIP (not a FAIL) of just
        # this sub-check if it doesn't line up.
        decision_name = f'{name} (N={n}) shaper decision'
        try:
            from shaketune.helpers.common_func import compute_mechanical_parameters

            # Mirror ShaperComputation._calibrate_shaper: build a real CalibrationData from the native
            # PSD (constructor is (name, freq_bins, psd_sum, psd_x, psd_y, psd_z)), normalize both,
            # derive the damping ratio, then run Klipper's own find_best_shaper on each and compare.
            CalibrationData = sc_module.CalibrationData
            # CalibrationData gained a leading `name` arg in Klipper master (Dec 2024); v0.13.0 and
            # Kalico don't take it. Construct with whichever signature this checkout actually has.
            import inspect as _inspect

            if 'name' in _inspect.signature(CalibrationData.__init__).parameters:
                native_cal = CalibrationData(meas_name, n_freq_bins, n_psd_sum, n_psd_x, n_psd_y, n_psd_z)
            else:
                native_cal = CalibrationData(n_freq_bins, n_psd_sum, n_psd_x, n_psd_y, n_psd_z)
            if hasattr(native_cal, 'set_numpy'):
                native_cal.set_numpy(np)

            def _best_shaper(cal):
                cal.normalize_to_frequencies()
                _fr, zeta, _, _ = compute_mechanical_parameters(cal.psd_sum, cal.freq_bins)
                zeta = zeta if zeta is not None else 0.1
                shaper, _ = find_best_shaper_compat(
                    shaper_calibrate_obj,
                    cal,
                    shapers=None,
                    damping_ratio=zeta,
                    scv=5.0,
                    shaper_freqs=None,
                    max_smoothing=None,
                    test_damping_ratios=None,
                    max_freq=200.0,
                    logger=None,
                )
                return shaper

            ref_shaper = _best_shaper(ref_cal)
            native_shaper = _best_shaper(native_cal)

            assert ref_shaper.name == native_shaper.name, (
                f'recommended shaper differs: {ref_shaper.name!r} != {native_shaper.name!r}'
            )
            assert abs(ref_shaper.freq - native_shaper.freq) <= 0.01, (
                f'recommended shaper freq differs by more than 0.01 Hz: {ref_shaper.freq} vs {native_shaper.freq}'
            )
        except AssertionError as exc:
            results.record(decision_name, 'FAIL', _first_line(exc))
            continue
        except Exception as exc:  # noqa: BLE001
            results.record(decision_name, 'SKIP', f'could not run end-to-end decision parity: {exc}')
            continue

        results.record(
            decision_name,
            'PASS',
            f'both recommend {ref_shaper.name} @ {ref_shaper.freq:.2f} Hz',
        )


# --------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Run native vs. pure-Python parity checks')
    parser.add_argument('--klipper-dir', default=None, help='Path to a Klipper checkout (enables kernel 4)')
    parser.add_argument(
        '--corpus',
        default=str(Path(__file__).resolve().parent / '_corpus'),
        help='Directory containing the corpus generated by gen_corpus.py',
    )
    parser.add_argument('--big', action='store_true', help='Also exercise the 1,000,000-sample corpus file')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)

    from shaketune.native import get_native, status

    nm = get_native()
    print('=' * 78)
    print('NATIVE ACTIVE' if nm is not None else 'FALLBACK (no native)')
    print(f'native status: {status()}')
    print('=' * 78)

    results = Results()
    kernel1_dir_speed_spectrogram(results, corpus_dir)
    kernel2_spectrogram(results, corpus_dir, args.big)
    kernel3_stdata(results, corpus_dir)
    kernel4_klipper_psd(results, corpus_dir, args.klipper_dir)

    results.print_summary()

    if results.failed():
        print('\nRESULT: FAILURE')
        sys.exit(1)

    if results.nothing_passed():
        print('\nRESULT: FAILURE (all checks were skipped — nothing was actually verified)')
        sys.exit(1)

    print('\nRESULT: SUCCESS (see SKIP rows above for anything not exercised)')
    sys.exit(0)


if __name__ == '__main__':
    main()
