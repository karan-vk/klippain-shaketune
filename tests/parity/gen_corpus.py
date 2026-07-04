#!/usr/bin/env python3
# Shake&Tune: 3D printer analysis tools
#
# File: gen_corpus.py
# Description: Deterministic synthetic corpus generator for the native (Rust) vs pure-Python
#              parity tests in tests/parity/. Produces:
#                - accelerometer traces (N, 4) [t, x, y, z], written as v1 ".stdata" files
#                  (zstd-compressed JSON-lines, matching MeasurementsManager's on-disk format)
#                  for N in {2048, 100000} (and optionally 1000000 behind --big).
#                - a synthetic vibrations dataset {angle: {speed: value}} for the
#                  dir_speed_spectrogram parity kernel, saved as a .npz file.
#
# Everything here is generated from a single seeded numpy Generator (seed=1234), so re-running
# this script always produces byte-identical corpus files.

import argparse
import json
from pathlib import Path

import numpy as np
import zstandard

SEED = 1234
FS = 3200.0  # approximate accelerometer sampling frequency, in Hz
COMPRESSION_LEVEL = 11  # matches shaketune.helpers.accelerometer.COMPRESSION_LEVEL

DEFAULT_SIZES = (2048, 100_000)
BIG_SIZE = 1_000_000


def synth_measurement(rng: np.random.Generator, n: int) -> np.ndarray:
    """Build a synthetic (N, 4) [t, x, y, z] accelerometer-like trace: a 5-130 Hz linear chirp
    plus 2-3 damped resonances plus small Gaussian noise, mixed differently per axis so that
    x/y/z are not degenerate copies of one another."""
    t = np.arange(n, dtype=np.float64) / FS
    duration = n / FS

    # Linear chirp from f0 to f1 Hz over the full duration of the trace.
    f0, f1 = 5.0, 130.0
    k = (f1 - f0) / max(duration, 1e-9)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t**2)
    chirp = np.sin(phase)

    # 2-3 damped resonances at random (but seeded) frequencies/decays/amplitudes/phases.
    n_res = int(rng.integers(2, 4))  # 2 or 3
    freqs = rng.uniform(20.0, 150.0, size=n_res)
    decays = rng.uniform(0.5, 3.0, size=n_res)
    amps = rng.uniform(0.3, 1.2, size=n_res)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_res)

    resonances = np.zeros(n, dtype=np.float64)
    for i in range(n_res):
        resonances += amps[i] * np.exp(-decays[i] * t) * np.sin(2.0 * np.pi * freqs[i] * t + phases[i])

    base = chirp + resonances
    noise_sigma = 0.02

    x = base + rng.normal(0.0, noise_sigma, n)
    y = 0.8 * base + 0.3 * chirp + rng.normal(0.0, noise_sigma, n)
    z = 0.5 * resonances + 0.2 * chirp + rng.normal(0.0, noise_sigma, n)

    return np.stack([t, x, y, z], axis=1)


def write_v1_stdata(path: Path, measurements) -> None:
    """Write measurements (iterable of (name, samples ndarray (N,4))) as a v1 ".stdata" file:
    a single zstd frame containing one JSON line per measurement, matching the exact on-disk
    format produced by shaketune.helpers.accelerometer.MeasurementsManager._writer_loop."""
    cctx = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL)
    with open(path, 'wb') as f, cctx.stream_writer(f) as compressor:
        for name, samples in measurements:
            obj = {'name': name, 'samples': np.asarray(samples, dtype=np.float64).tolist()}
            line = (json.dumps(obj) + '\n').encode('utf-8')
            compressor.write(line)
        compressor.flush(zstandard.FLUSH_FRAME)


def gen_vibrations_dataset(rng: np.random.Generator):
    """Build a synthetic {angle: {speed: value}} vibrations dataset for angles [0, 90] and
    speeds 2..200 mm/s (step 2, ~100 points), with a smooth, always-positive value function of
    (speed, angle) that includes a resonance-like bump so the data isn't perfectly flat."""
    angles = np.array([0, 90], dtype=np.float64)
    speeds = np.arange(2, 202, 2, dtype=np.float64)  # 100 points: 2, 4, ..., 200

    values = np.empty((len(angles), len(speeds)), dtype=np.float64)
    # A tiny amount of seeded jitter keeps the surface from being perfectly analytic/symmetric,
    # which is a better stress test for interpolation code, while staying strictly positive.
    jitter = rng.normal(0.0, 0.5, size=values.shape)
    for ai, angle in enumerate(angles):
        base = 20.0 + 0.05 * speeds
        bump = 40.0 * np.exp(-((speeds - 80.0) ** 2) / (2.0 * 15.0**2))
        angle_mod = 1.0 + 0.3 * np.sin(np.deg2rad(angle) + speeds / 40.0)
        values[ai] = (base + bump) * angle_mod + jitter[ai]

    # Guard the "always positive" contract regardless of jitter draw.
    values = np.clip(values, 0.1, None)

    return angles, speeds, values


def main():
    parser = argparse.ArgumentParser(description='Generate the deterministic parity-test corpus')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / '_corpus'), help='Output directory')
    parser.add_argument('--big', action='store_true', help='Also generate the large (1,000,000-sample) corpus file')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    sizes = list(DEFAULT_SIZES)
    if args.big:
        sizes.append(BIG_SIZE)

    for n in sizes:
        samples = synth_measurement(rng, n)
        out_path = out_dir / f'corpus_{n}.stdata'
        write_v1_stdata(out_path, [(f'corpus_{n}', samples)])
        print(f'wrote {out_path} ({n} samples)')

    angles, speeds, values = gen_vibrations_dataset(rng)
    npz_path = out_dir / 'vibrations.npz'
    np.savez(npz_path, angles=angles, speeds=speeds, values=values)
    print(f'wrote {npz_path} (angles={angles.tolist()}, {len(speeds)} speeds)')

    print(f'\nCorpus generation complete in {out_dir}')


if __name__ == '__main__':
    main()
