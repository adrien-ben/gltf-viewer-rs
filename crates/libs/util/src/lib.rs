use std::path::Path;

/// Return a `&[u8]` for any sized object passed in.
pub unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    unsafe {
        let ptr = (any as *const T) as *const u8;
        std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
    }
}

pub fn load_hdr_image<P: AsRef<Path>>(path: P) -> (u32, u32, Vec<f32>) {
    let img = image::open(path).unwrap();
    let w = img.width();
    let h = img.height();
    let data = img.into_rgba32f().into_raw();

    (w, h, data)
}
