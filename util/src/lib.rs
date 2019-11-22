pub mod fs;

use image::{hdr::HDRDecoder, imageops::resize, FilterType, ImageBuffer, Rgb, Rgba};
use std::path::Path;

/// Return a `&[u8]` for any sized object passed in.
pub unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}

#[derive(Clone, Debug)]
pub struct RgbaHdrImageData {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

impl RgbaHdrImageData {
    /// Load an HDR image from a path
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let decoder = HDRDecoder::new(fs::load(path)).unwrap();

        let width = decoder.metadata().width;
        let height = decoder.metadata().height;
        let rgb = decoder.read_image_hdr().unwrap();
        let mut pixels = Vec::with_capacity(rgb.len() * 4);
        for Rgb(p) in rgb.iter() {
            pixels.extend_from_slice(p);
            pixels.push(0.0);
        }

        Self {
            width,
            height,
            pixels,
        }
    }

    /// Generate mipmaps for the image
    pub fn get_mipmaps(&self) -> Vec<RgbaHdrImageData> {
        match self.get_half_res() {
            None => vec![],
            Some(first_mipmap) => {
                let mut mipmaps = vec![first_mipmap];
                while let Some(next_mipmap) = mipmaps.last().unwrap().get_half_res() {
                    mipmaps.push(next_mipmap);
                }
                mipmaps
            }
        }
    }

    fn get_half_res(&self) -> Option<Self> {
        if self.width == 1 || self.height == 1 {
            return None;
        }

        let width = self.width / 2;
        let height = self.height / 2;

        let image: ImageBuffer<_, _> = self.into();
        let image = resize(&image, width, height, FilterType::Triangle);

        Some(RgbaHdrImageData {
            width,
            height,
            pixels: image.into_vec(),
        })
    }
}

impl Into<ImageBuffer<Rgba<f32>, Vec<f32>>> for &RgbaHdrImageData {
    fn into(self) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
        ImageBuffer::from_vec(self.width, self.height, self.pixels.clone())
            .expect("Failed to build image from data")
    }
}

/// Compute the number of mipmap levels (including top level) from the dimensions of an image.
pub fn compute_mipmap_levels(width: u32, height: u32) -> u32 {
    (width.min(height) as f32).log2().floor() as u32 + 1
}
