use gltf::{buffer::Data, Accessor};

pub fn read_accessor(accessor: &Accessor, buffers: &[Data]) -> Vec<u8> {
    let view = accessor.view();
    let buffer = view.buffer();
    let data = &buffers[buffer.index()];

    let offset = view.offset() + accessor.offset();
    let stride = view.stride().unwrap_or_else(|| accessor.size());

    let mut vertices = Vec::<u8>::new();
    for component_index in 0..accessor.count() {
        let offset = offset + component_index * stride;
        for byte_index in 0..accessor.size() {
            vertices.push(data[offset + byte_index]);
        }
    }
    vertices
}
