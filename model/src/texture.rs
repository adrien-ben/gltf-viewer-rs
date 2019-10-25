use ash::vk;
use gltf::image::{Data, Format};
use gltf::iter::Textures;
use std::sync::Arc;
use vulkan::*;

/// Create
pub fn create_textures_from_gltf(
    context: &Arc<Context>,
    command_buffer: vk::CommandBuffer,
    textures: Textures,
    images: &[Data],
) -> (Vec<Texture>, Vec<Buffer>) {
    textures
        .map(|texture| texture.source().index())
        .map(|texture_index| &images[texture_index])
        .map(|image| (image.width, image.height, build_rgba_buffer(image)))
        .map(|(width, height, pixels)| {
            Texture::cmd_from_rgba(&context, command_buffer, width, height, &pixels)
        })
        .unzip()
}

fn build_rgba_buffer(image: &Data) -> Vec<u8> {
    let mut buffer = Vec::new();
    let size = image.width * image.height;
    for index in 0..size {
        let rgba = get_next_rgba(&image.pixels, image.format, index as usize);
        buffer.extend_from_slice(&rgba);
    }
    buffer
}

fn get_next_rgba(pixels: &[u8], format: Format, index: usize) -> [u8; 4] {
    match format {
        Format::R8 => [pixels[index], 0, 0, std::u8::MAX],
        Format::R8G8 => [pixels[index * 2], pixels[index * 2 + 1], 0, std::u8::MAX],
        Format::R8G8B8 => [
            pixels[index * 3],
            pixels[index * 3 + 1],
            pixels[index * 3 + 2],
            std::u8::MAX,
        ],
        Format::B8G8R8 => [
            pixels[index * 3 + 2],
            pixels[index * 3 + 1],
            pixels[index * 3],
            std::u8::MAX,
        ],
        Format::R8G8B8A8 => [
            pixels[index * 4],
            pixels[index * 4 + 1],
            pixels[index * 4 + 2],
            pixels[index * 4 + 3],
        ],
        Format::B8G8R8A8 => [
            pixels[index * 4 + 2],
            pixels[index * 4 + 1],
            pixels[index * 4],
            pixels[index * 4 + 3],
        ],
    }
}
