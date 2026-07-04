// Shake&Tune native module
//
// File: stdata.rs
// Description: Kernel 3 - reader/writer for the `.stdata` measurement file format.
//
// Two on-disk formats are supported, both wrapped in a single Zstandard frame:
//   v1 (legacy): Zstandard-compressed JSON-lines, one `{"name": ..., "samples": [[t,x,y,z], ...]}`
//                object per line (see shaketune/helpers/accelerometer.py).
//   v2 (native): a compact binary format written by `StdataWriter`, described below.
//
// v2 binary layout (all integers little-endian), inside one zstd frame:
//   header: 8 bytes magic b"STDATAV2", then u16 version = 2
//   per record: u32 name_len | name (utf-8, name_len bytes) | u64 n_samples
//               | n_samples * 4 f64 (row-major [t, x, y, z])

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use numpy::ndarray::{Array2, ArrayView2};
use serde::Deserialize;

const MAGIC: &[u8; 8] = b"STDATAV2";
const VERSION: u16 = 2;

pub struct StdataWriterCore {
    encoder: Option<zstd::stream::write::Encoder<'static, File>>,
}

impl StdataWriterCore {
    pub fn create(path: &str, level: i32) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("unable to create '{path}': {e}"))?;
        let mut encoder =
            zstd::stream::write::Encoder::new(file, level).map_err(|e| format!("unable to start zstd stream: {e}"))?;
        encoder
            .write_all(MAGIC)
            .and_then(|_| encoder.write_all(&VERSION.to_le_bytes()))
            .map_err(|e| format!("unable to write stdata header: {e}"))?;
        Ok(StdataWriterCore { encoder: Some(encoder) })
    }

    pub fn write_measurement(&mut self, name: &str, samples: ArrayView2<f64>) -> Result<(), String> {
        if samples.ncols() != 4 {
            return Err(format!(
                "samples array must have shape (N, 4), got (N, {})",
                samples.ncols()
            ));
        }
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| "writer is already closed".to_string())?;

        let name_bytes = name.as_bytes();
        let n = samples.nrows();

        let mut buf = Vec::with_capacity(14 + name_bytes.len() + n * 4 * 8);
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        for row in samples.outer_iter() {
            for k in 0..4 {
                buf.extend_from_slice(&row[k].to_le_bytes());
            }
        }

        encoder.write_all(&buf).map_err(|e| format!("unable to write measurement: {e}"))
    }

    /// Finish the zstd frame and close the underlying file. Idempotent: calling this more than
    /// once is a no-op after the first successful call.
    pub fn close(&mut self) -> Result<(), String> {
        if let Some(encoder) = self.encoder.take() {
            encoder.finish().map_err(|e| format!("unable to finish stdata file: {e}"))?;
        }
        Ok(())
    }
}

impl Drop for StdataWriterCore {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[derive(Deserialize)]
struct V1Measurement {
    name: String,
    samples: Vec<Vec<f64>>,
}

fn samples_to_array2(samples: Vec<Vec<f64>>, context: &str) -> Result<Array2<f64>, String> {
    let n = samples.len();
    let mut flat = Vec::with_capacity(n * 4);
    for (i, row) in samples.into_iter().enumerate() {
        if row.len() != 4 {
            return Err(format!(
                "{context}: sample row {i} has {} columns, expected 4",
                row.len()
            ));
        }
        flat.extend_from_slice(&row);
    }
    Array2::from_shape_vec((n, 4), flat).map_err(|e| format!("{context}: {e}"))
}

struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        ByteCursor { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("unexpected end of stdata v2 stream".to_string());
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn take_u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn take_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn take_u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }
}

fn parse_v2(decoded: &[u8]) -> Result<Vec<(String, Array2<f64>)>, String> {
    let mut cursor = ByteCursor::new(decoded);
    cursor.take(MAGIC.len())?; // magic already verified by caller
    let version = cursor.take_u16()?;
    if version != VERSION {
        return Err(format!("unsupported stdata v2 version {version}"));
    }

    let mut records = Vec::new();
    while cursor.remaining() > 0 {
        let name_len = cursor.take_u32()? as usize;
        let name_bytes = cursor.take(name_len)?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|e| format!("invalid utf-8 measurement name: {e}"))?;
        let n_samples = cursor.take_u64()? as usize;
        let mut flat = Vec::with_capacity(n_samples * 4);
        for _ in 0..n_samples * 4 {
            let bytes = cursor.take(8)?;
            flat.push(f64::from_le_bytes(bytes.try_into().unwrap()));
        }
        let array = Array2::from_shape_vec((n_samples, 4), flat).map_err(|e| format!("record '{name}': {e}"))?;
        records.push((name, array));
    }

    Ok(records)
}

fn parse_v1(decoded: &[u8]) -> Result<Vec<(String, Array2<f64>)>, String> {
    let text = String::from_utf8_lossy(decoded);
    let mut records = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let meas: V1Measurement =
            serde_json::from_str(line).map_err(|e| format!("invalid stdata v1 JSON line: {e}"))?;
        let array = samples_to_array2(meas.samples, &meas.name)?;
        records.push((meas.name, array));
    }
    Ok(records)
}

pub fn read_stdata(path: &str) -> Result<Vec<(String, Array2<f64>)>, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("unable to open '{path}': {e}"))?;
    let mut decoded = Vec::new();
    zstd::stream::read::Decoder::new(BufReader::new(file))
        .map_err(|e| format!("unable to start zstd decoder: {e}"))?
        .read_to_end(&mut decoded)
        .map_err(|e| format!("unable to decompress '{path}': {e}"))?;

    if decoded.len() >= MAGIC.len() && &decoded[..MAGIC.len()] == MAGIC {
        parse_v2(&decoded)
    } else {
        parse_v1(&decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn roundtrip_v2_single_measurement() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("shaketune_test_{}.stdata", std::process::id()));
        let path_str = path.to_str().unwrap();

        let samples = array![[0.0, 1.0, 2.0, 3.0], [0.001, 1.1, 2.1, 3.1], [0.002, 1.2, 2.2, 3.2]];

        {
            let mut writer = StdataWriterCore::create(path_str, 3).unwrap();
            writer.write_measurement("meas_a", samples.view()).unwrap();
            writer.close().unwrap();
            // Idempotent close.
            writer.close().unwrap();
        }

        let records = read_stdata(path_str).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, "meas_a");
        assert_eq!(records[0].1, samples);

        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn roundtrip_v2_multiple_measurements() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("shaketune_test_multi_{}.stdata", std::process::id()));
        let path_str = path.to_str().unwrap();

        let s1 = array![[0.0, 1.0, 2.0, 3.0]];
        let s2 = array![[0.0, 4.0, 5.0, 6.0], [0.1, 4.1, 5.1, 6.1]];

        {
            let mut writer = StdataWriterCore::create(path_str, 1).unwrap();
            writer.write_measurement("first", s1.view()).unwrap();
            writer.write_measurement("second", s2.view()).unwrap();
            writer.close().unwrap();
        }

        let records = read_stdata(path_str).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, "first");
        assert_eq!(records[1].0, "second");
        assert_eq!(records[1].1, s2);

        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn write_measurement_rejects_wrong_shape() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("shaketune_test_bad_{}.stdata", std::process::id()));
        let path_str = path.to_str().unwrap();
        let mut writer = StdataWriterCore::create(path_str, 1).unwrap();
        let bad = Array2::<f64>::zeros((3, 3));
        assert!(writer.write_measurement("bad", bad.view()).is_err());
        writer.close().unwrap();
        let _ = std::fs::remove_file(path_str);
    }

    #[test]
    fn parse_v1_json_lines() {
        let json_line = r#"{"name": "abc", "samples": [[0.0, 1.0, 2.0, 3.0], [0.1, 1.1, 2.1, 3.1]]}"#;
        let compressed = zstd::stream::encode_all(json_line.as_bytes(), 3).unwrap();

        let dir = std::env::temp_dir();
        let path = dir.join(format!("shaketune_test_v1_{}.stdata", std::process::id()));
        std::fs::write(&path, &compressed).unwrap();

        let records = read_stdata(path.to_str().unwrap()).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, "abc");
        assert_eq!(records[0].1.shape(), &[2, 4]);

        let _ = std::fs::remove_file(&path);
    }
}
