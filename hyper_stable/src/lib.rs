use pyo3::prelude::*;
use numpy::PyReadonlyArray3;
use image::{ImageBuffer, Rgb};
use std::io::Cursor;

// 1. THE RUST FUNCTION
// It takes a NumPy array (uint8, 3 dimensions: H, W, C)
// It returns a Python Bytes object (the JPEG data)
#[pyfunction]
fn process_frame(
    py: Python, 
    array: PyReadonlyArray3<u8>, 
    target_width: u32, 
    target_height: u32, 
    quality: u8
) -> PyResult<PyObject> {

    // A. Convert Numpy to Rust View (Zero Copy!)
    let array_view = array.as_array();
    let (height, width, _) = array_view.dim();

    // B. Create an Image Buffer from the raw data
    // We have to be careful here to match the shape perfectly.
    // numpy gives us [row, col, channel].
    // We flatten this into a raw slice for the image crate.
    let raw_slice = array_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Array is not contiguous")
    })?;

    // Create container
    let img_buffer: ImageBuffer<Rgb<u8>, _> = 
        ImageBuffer::from_raw(width as u32, height as u32, raw_slice)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Could not create image buffer"))?;

    // C. RELEASE THE GIL! 
    // This allows Python to do other work while Rust crunches the numbers.
    let jpeg_bytes = py.allow_threads(move || {
        // 1. Resize (High quality Lanczos3 filter, or Nearest for speed)
        let resized = image::imageops::resize(
            &img_buffer, 
            target_width, 
            target_height, 
            image::imageops::FilterType::Nearest 
        );

        // 2. Encode to JPEG
        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut bytes);
        
        // Write the JPEG to the memory buffer
        resized.write_to(&mut cursor, image::ImageOutputFormat::Jpeg(quality)).unwrap();
        
        bytes
    });

    // D. Return as Python Bytes
    Ok(pyo3::types::PyBytes::new(py, &jpeg_bytes).into())
}

// 2. THE MODULE DEFINITION
#[pymodule]
fn hyper_stable(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_frame, m)?)?;
    Ok(())
}
