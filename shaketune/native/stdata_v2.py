# Shake&Tune: 3D printer analysis tools
#
# Copyright (C) 2024 Félix Boisselier <felix@fboisselier.fr> (Frix_x on Discord)
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
#
# File: stdata_v2.py
# Description: Pure-Python reader for the Shake&Tune ".stdata" v2 binary format (a single
#              Zstandard frame containing a magic header followed by a sequence of named
#              (t, x, y, z) float64 sample-array records). This lets hosts without the
#              compiled native extension still read v2 files written by the Rust writer.

import struct
from typing import List

import numpy as np
import zstandard

MAGIC = b'STDATAV2'


def read_stdata_v2(path) -> List[dict]:
    """Read a v2 ".stdata" file and return a list of {'name': str, 'samples': ndarray} dicts,
    one per recorded measurement, where each ndarray has shape (Ni, 4) and dtype float64 with
    columns [t, x, y, z]."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, 'rb') as f, dctx.stream_reader(f) as reader:
        buf = reader.read()

    offset = 0
    magic = buf[offset : offset + 8]
    offset += 8
    if magic != MAGIC:
        raise ValueError(f'Invalid stdata v2 magic: {magic!r}')

    (version,) = struct.unpack_from('<H', buf, offset)
    offset += 2
    if version != 2:
        raise ValueError(f'Unsupported stdata v2 version: {version}')

    records = []
    buf_len = len(buf)
    while offset < buf_len:
        (name_len,) = struct.unpack_from('<I', buf, offset)
        offset += 4
        name_bytes = buf[offset : offset + name_len]
        if len(name_bytes) != name_len:
            raise ValueError('Truncated stdata v2 stream (incomplete record name)')
        name = name_bytes.decode('utf-8')
        offset += name_len

        (n_samples,) = struct.unpack_from('<Q', buf, offset)
        offset += 8

        n_bytes = n_samples * 4 * 8  # n_samples rows * 4 columns * 8 bytes per float64
        sample_bytes = buf[offset : offset + n_bytes]
        if len(sample_bytes) != n_bytes:
            raise ValueError(
                f'Truncated stdata v2 stream: record {name!r} declares {n_samples} samples '
                f'but only {len(sample_bytes)} of {n_bytes} bytes are present'
            )
        offset += n_bytes

        # bytearray() makes the resulting array writable (np.frombuffer on bytes yields a
        # read-only array, unlike the native reader which always returns writable arrays)
        samples = np.frombuffer(bytearray(sample_bytes), dtype='<f8').reshape(n_samples, 4)
        records.append({'name': name, 'samples': samples})

    return records
