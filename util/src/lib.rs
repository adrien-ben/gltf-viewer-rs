use image::hdr::HdrDecoder;
use image::Rgb;
use std::{fs::File, io::BufReader, path::Path};

/// Return a `&[u8]` for any sized object passed in.
pub unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}

pub fn load_hdr_image<P: AsRef<Path>>(path: P) -> (u32, u32, Vec<f32>) {
    let decoder = HdrDecoder::new(BufReader::new(File::open(path).unwrap())).unwrap();

    let w = decoder.metadata().width;
    let h = decoder.metadata().height;
    let rgb = decoder.read_image_hdr().unwrap();
    let mut data = Vec::with_capacity(rgb.len() * 4);
    for Rgb(p) in rgb.iter() {
        data.extend_from_slice(p);
        data.push(0.0);
    }
    (w, h, data)
}
