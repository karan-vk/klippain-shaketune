#!/usr/bin/env python3
# Shake&Tune: 3D printer analysis tools
#
# File: bench.py
# Description: Standalone benchmark of the compiled Rust "_core" native extension against the
#              pure-Python fallback, for each of the four accelerated kernels, plus the in-memory
#              samples representation change. Reports median wall time over a few repeats and the
#              native/fallback speedup. Run gen_corpus.py (with --big for the 1M case) first.
#
# Usage:
#   python tests/parity/gen_corpus.py --big
#   python tests/parity/bench.py [--klipper-dir DIR] [--corpus DIR]
#
# NOTE: absolute times are machine-specific. The transferable figure is the speedup ratio; on
# interpreter-bound kernels (1 and 3-serialize) low-power SBCs (Pi Zero/1) typically see a LARGER
# relative win than a fast x86_64 dev box, since Python-loop/allocation overhead dominates there.

import argparse
import io
import json
import os
import sys
import time
import tracemalloc
from importlib import import_module
from pathlib import Path

import numpy as np
import zstandard

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for run_parity helpers
os.environ['SHAKETUNE_IN_CLI'] = '1'

from run_parity import load_v1_measurement, read_v1_direct, reference_dir_speed_spectrogram  # noqa: E402


def _set_native(enabled):
    if enabled:
        os.environ.pop('SHAKETUNE_DISABLE_NATIVE', None)
    else:
        os.environ['SHAKETUNE_DISABLE_NATIVE'] = '1'


def timeit(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]  # median


ROWS = []


def report(kernel, detail, native_s, py_s):
    speedup = py_s / native_s if native_s > 0 else float('inf')
    ROWS.append((kernel, detail, native_s, py_s, speedup))
    print(
        f'  {kernel:<28} {detail:<16} native={native_s * 1e3:9.2f} ms  '
        f'python={py_s * 1e3:9.2f} ms  speedup={speedup:6.1f}x'
    )


def bench_kernel1(corpus_dir):
    from shaketune.native import get_native

    nm = get_native()
    if nm is None:
        print('  kernel1: SKIP (no native)')
        return
    d = np.load(corpus_dir / 'vibrations.npz')
    ms = np.ascontiguousarray(d['speeds'], dtype=np.float64)
    va = np.ascontiguousarray(d['values'][0], dtype=np.float64)
    vb = np.ascontiguousarray(d['values'][1], dtype=np.float64)
    # Bench at the shipped default (1440 angles, 12x speed oversampling). The pure-Python
    # reference is a nested angle*speed loop, so benching the legacy (720, 6) grid as well would
    # roughly triple total runtime for little extra signal -- skip it (the parity check in
    # run_parity.py already covers that grid for correctness).
    n_angles, speed_oversampling = 1440, 12
    for corexy in (False, True):
        native_s = timeit(
            lambda c=corexy: nm.dir_speed_spectrogram(ms, va, vb, c, n_angles, speed_oversampling), repeats=5
        )
        py_s = timeit(
            lambda c=corexy: reference_dir_speed_spectrogram(ms, va, vb, c, n_angles, speed_oversampling), repeats=3
        )
        report(
            'kernel1 vibrations proj',
            f'{"corexy" if corexy else "cartesian"} {n_angles}x{speed_oversampling}',
            native_s,
            py_s,
        )


def bench_kernel2(corpus_dir):
    from shaketune.helpers.spectrogram import compute_spectrogram

    for n in (100_000, 1_000_000):
        path = corpus_dir / f'corpus_{n}.stdata'
        if not path.exists():
            continue
        _, samples = load_v1_measurement(path)
        _set_native(True)
        native_s = timeit(lambda s=samples: compute_spectrogram(s), repeats=5)
        _set_native(False)
        py_s = timeit(lambda s=samples: compute_spectrogram(s), repeats=5)
        _set_native(True)
        report('kernel2 spectrogram', f'N={n}', native_s, py_s)


def bench_kernel3(corpus_dir, tmp):
    from shaketune.native import get_native

    nm = get_native()
    if nm is None:
        print('  kernel3: SKIP (no native)')
        return
    path = corpus_dir / 'corpus_1000000.stdata'
    if not path.exists():
        path = corpus_dir / 'corpus_100000.stdata'
    name, samples = load_v1_measurement(path)
    n = samples.shape[0]
    lvl = 11

    # --- write (serialize + zstd) ---
    v2_path = Path(tmp) / 'bench_v2.stdata'

    def write_v2():
        w = nm.StdataWriter(str(v2_path), lvl)
        w.write_measurement(name, samples)
        w.close()

    def write_v1_json():
        # exactly what the pure-Python _writer_loop fallback does
        blob = (json.dumps({'name': name, 'samples': samples.tolist()}) + '\n').encode('utf-8')
        buf = io.BytesIO()
        cctx = zstandard.ZstdCompressor(level=lvl)
        with cctx.stream_writer(buf) as c:
            c.write(blob)

    w_native = timeit(write_v2, repeats=3)
    w_py = timeit(write_v1_json, repeats=3)
    report('kernel3 write (.stdata)', f'N={n}', w_native, w_py)

    # --- read (zstd + parse) ---
    v1_path = path
    r_native = timeit(lambda: nm.read_stdata(str(v2_path)), repeats=3)
    r_py = timeit(lambda: read_v1_direct(v1_path), repeats=3)
    report('kernel3 read (.stdata)', f'N={n}', r_native, r_py)

    # --- file size (v2 binary vs v1 json, both zstd-11) ---
    print(
        f'  kernel3 file size          N={n:<11} v2={v2_path.stat().st_size / 1e6:6.2f} MB  '
        f'v1={v1_path.stat().st_size / 1e6:6.2f} MB'
    )

    # --- in-memory samples footprint: old list-of-tuples vs new ndarray ---
    py_rows = [tuple(float(x) for x in row) for row in samples[:200_000]]  # sample to bound time
    per_row = sys.getsizeof(py_rows[0]) + sum(sys.getsizeof(v) for v in py_rows[0])
    old_bytes = sys.getsizeof(py_rows) / len(py_rows) * n + per_row * n  # list slots + tuple/float objs
    new_bytes = samples.nbytes
    print(
        f'  kernel3 in-mem samples     N={n:<11} ndarray={new_bytes / 1e6:6.1f} MB  '
        f'list-of-tuples~{old_bytes / 1e6:6.1f} MB  ({old_bytes / new_bytes:.1f}x smaller)'
    )
    del py_rows


def bench_kernel4(corpus_dir, klipper_dir):
    if not klipper_dir:
        print('  kernel4: SKIP (no --klipper-dir)')
        return
    kdir = os.path.expanduser(klipper_dir)
    sys.path.append(os.path.join(kdir, 'klippy'))
    sys.modules['shaper_calibrate'] = import_module('.shaper_calibrate', 'extras')
    sys.modules['shaper_defs'] = import_module('.shaper_defs', 'extras')

    from shaketune.graph_creators import get_shaper_calibrate_module, process_accelerometer_data_compat

    _set_native(True)
    sc, _ = get_shaper_calibrate_module()

    for n in (100_000, 1_000_000):
        path = corpus_dir / f'corpus_{n}.stdata'
        if not path.exists():
            continue
        _, samples = load_v1_measurement(path)
        _set_native(True)
        native_s = timeit(lambda s=samples: process_accelerometer_data_compat(sc, s, name='b'), repeats=5)
        _set_native(False)
        py_s = timeit(lambda s=samples: process_accelerometer_data_compat(sc, s, name='b'), repeats=5)
        _set_native(True)
        report('kernel4 klipper PSD', f'N={n}', native_s, py_s)


def main():
    ap = argparse.ArgumentParser(description='Benchmark native vs pure-Python kernels')
    ap.add_argument('--klipper-dir', default=None)
    ap.add_argument('--corpus', default=str(Path(__file__).resolve().parent / '_corpus'))
    args = ap.parse_args()
    corpus_dir = Path(args.corpus)

    from shaketune.native import get_native, status

    print('=' * 92)
    print('NATIVE ACTIVE' if get_native() is not None else 'FALLBACK (no native) — nothing to compare')
    print(f'status: {status()}')
    print('=' * 92)

    tracemalloc.start()  # not strictly needed but harmless; kernel3 uses sys.getsizeof
    bench_kernel1(corpus_dir)
    bench_kernel2(corpus_dir)
    bench_kernel3(corpus_dir, os.environ.get('TMPDIR', '/tmp'))
    bench_kernel4(corpus_dir, args.klipper_dir)
    print('=' * 92)


if __name__ == '__main__':
    main()
