// Shake&Tune: 3D printer analysis tools
//
// File: lib.rs
// Description: PyO3 native extension (`_core`) providing the performance-critical numeric
//              kernels used by Shake&Tune: PSD spectrogram computation, Klipper-parity Welch
//              PSD, the directional speed/vibrations spectrogram, and the `.stdata` file
//              format reader/writer. All heavy numeric work runs with the GIL released
//              (`py.allow_threads`) since this module is loaded inside Klipper's single
//              -threaded `klippy` process. No threads/rayon are used internally: this target
//              also runs on single-core Pi Zero hardware and must stay fork-safe.

mod kaiser;
mod psd;
mod spectrogram;
mod stdata;
mod vibrations;
mod welch;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// (pdata, t, f)
type SpectrogramPyOutput<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
/// (freq_bins, psd_sum, psd_x, psd_y, psd_z)
type KlipperPsdPyOutput<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);
/// (spectrum_angles, spectrum_speeds, spectrum_vibrations)
type DirSpeedSpectrogramPyOutput<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// Compute a power spectral density spectrogram from raw accelerometer `data` of shape
/// (N, 4) with columns [time, x, y, z]. Mirrors
/// `shaketune/helpers/spectrogram.py::compute_spectrogram` exactly.
#[pyfunction]
#[pyo3(name = "spectrogram")]
fn spectrogram_fn<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, f64>) -> PyResult<SpectrogramPyOutput<'py>> {
    // Copy the input while the GIL is held (the source buffer is only guaranteed valid under
    // the GIL), then run the actual FFT-heavy computation with the GIL released.
    let owned = data.as_array().to_owned();
    let (pdata, t, f) = py
        .allow_threads(move || crate::spectrogram::compute_spectrogram(owned.view()))
        .map_err(PyValueError::new_err)?;

    Ok((pdata.into_pyarray(py), t.into_pyarray(py), f.into_pyarray(py)))
}

/// Compute the Klipper-parity Welch-averaged PSDs (full-length, no max_freq filtering) from
/// raw accelerometer `data` of shape (N, 4). Mirrors Klipper's
/// `shaper_calibrate.py::ShaperCalibrate._psd` / `calc_freq_response` exactly.
#[pyfunction]
fn klipper_psd<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, f64>) -> PyResult<KlipperPsdPyOutput<'py>> {
    let owned = data.as_array().to_owned();
    let result = py
        .allow_threads(move || psd::compute_klipper_psd(owned.view()))
        .map_err(PyValueError::new_err)?;

    Ok((
        result.freqs.into_pyarray(py),
        result.psd_sum.into_pyarray(py),
        result.psd_x.into_pyarray(py),
        result.psd_y.into_pyarray(py),
        result.psd_z.into_pyarray(py),
    ))
}

/// Compute the directional speed/vibrations spectrogram used by the machine vibrations
/// analysis tool. Mirrors `VibrationsComputation._compute_dir_speed_spectrogram` exactly.
#[pyfunction]
fn dir_speed_spectrogram<'py>(
    py: Python<'py>,
    measured_speeds: PyReadonlyArray1<'py, f64>,
    vibs_a: PyReadonlyArray1<'py, f64>,
    vibs_b: PyReadonlyArray1<'py, f64>,
    corexy: bool,
) -> PyResult<DirSpeedSpectrogramPyOutput<'py>> {
    let speeds: Vec<f64> = measured_speeds.as_array().iter().copied().collect();
    let a: Vec<f64> = vibs_a.as_array().iter().copied().collect();
    let b: Vec<f64> = vibs_b.as_array().iter().copied().collect();

    let (angles, out_speeds, vib) = py
        .allow_threads(move || vibrations::compute_dir_speed_spectrogram(&speeds, &a, &b, corexy))
        .map_err(PyValueError::new_err)?;

    Ok((angles.into_pyarray(py), out_speeds.into_pyarray(py), vib.into_pyarray(py)))
}

/// Load all measurements from a `.stdata` file (auto-sniffing the legacy v1 zstd/JSON-lines
/// format and the native v2 zstd/binary format).
#[pyfunction]
fn read_stdata<'py>(py: Python<'py>, path: String) -> PyResult<Vec<(String, Bound<'py, PyArray2<f64>>)>> {
    let records = py
        .allow_threads(move || stdata::read_stdata(&path))
        .map_err(PyValueError::new_err)?;

    Ok(records
        .into_iter()
        .map(|(name, arr)| (name, arr.into_pyarray(py)))
        .collect())
}

/// Streaming writer for the native v2 `.stdata` binary format (a single zstd frame containing
/// a compact binary encoding of each measurement).
#[pyclass]
struct StdataWriter {
    inner: stdata::StdataWriterCore,
}

#[pymethods]
impl StdataWriter {
    #[new]
    fn new(path: String, level: i32) -> PyResult<Self> {
        let inner = stdata::StdataWriterCore::create(&path, level).map_err(PyValueError::new_err)?;
        Ok(StdataWriter { inner })
    }

    fn write_measurement(&mut self, py: Python<'_>, name: String, samples: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let owned = samples.as_array().to_owned();
        let inner = &mut self.inner;
        py.allow_threads(move || inner.write_measurement(&name, owned.view()))
            .map_err(PyValueError::new_err)
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.allow_threads(move || inner.close()).map_err(PyValueError::new_err)
    }
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__source_tree_hash__",
        option_env!("SHAKETUNE_RUST_TREE_HASH").unwrap_or("dev"),
    )?;

    m.add_function(wrap_pyfunction!(spectrogram_fn, m)?)?;
    m.add_function(wrap_pyfunction!(klipper_psd, m)?)?;
    m.add_function(wrap_pyfunction!(dir_speed_spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(read_stdata, m)?)?;
    m.add_class::<StdataWriter>()?;

    Ok(())
}
