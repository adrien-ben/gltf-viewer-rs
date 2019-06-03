
use ash::{util::Align, vk::DeviceSize};
use std::{ffi::c_void, mem::size_of};

/// Utility function that copy the content of a slice at the position of a given pointer.
pub unsafe fn mem_copy<T: Copy>(ptr: *mut c_void, data: &[T]) {
    let elem_size = size_of::<T>() as DeviceSize;
    let size = data.len() as DeviceSize * elem_size;
    let mut align = Align::new(ptr, elem_size, size);
    align.copy_from_slice(data);
}

/// Utility function that copy the content of a slice at the position of a given pointer and pad elements to respect the requested alignment.
pub unsafe fn mem_copy_aligned<T: Copy>(ptr: *mut c_void, alignment: DeviceSize, data: &[T]) {
    let size = data.len() as DeviceSize * alignment;
    let mut align = Align::new(ptr, alignment, size);
    align.copy_from_slice(data);
}
