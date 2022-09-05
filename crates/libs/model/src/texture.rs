use gltf::image::{Data, Format};
use gltf::iter::{Materials, Textures as GltfTextures};
use gltf::json::texture::{MagFilter, MinFilter, WrappingMode};
use gltf::texture::Sampler;
use std::collections::HashSet;
use std::sync::Arc;
use vulkan::ash::vk;
use vulkan::{Buffer, Context, Image, Texture as VulkanTexture};

pub(crate) struct Textures {
    _images: Vec<VulkanTexture>,
    pub textures: Vec<Texture>,
}

pub struct Texture {
    context: Arc<Context>,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

impl Texture {
    pub fn get_view(&self) -> vk::ImageView {
        self.view
    }

    pub fn get_sampler(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.context.device().destroy_sampler(self.sampler, None);
        }
    }
}

/// Create
pub(crate) fn create_textures_from_gltf(
    context: &Arc<Context>,
    command_buffer: vk::CommandBuffer,
    textures: GltfTextures,
    materials: Materials,
    images: &[Data],
) -> (Textures, Vec<Buffer>) {
    let srgb_image_indices = {
        let mut indices = HashSet::new();

        for m in materials {
            if let Some(t) = m.pbr_metallic_roughness().base_color_texture() {
                indices.insert(t.texture().source().index());
            }

            if let Some(pbr_specular_glossiness) = m.pbr_specular_glossiness() {
                if let Some(t) = pbr_specular_glossiness.diffuse_texture() {
                    indices.insert(t.texture().source().index());
                }

                if let Some(t) = pbr_specular_glossiness.specular_glossiness_texture() {
                    indices.insert(t.texture().source().index());
                }
            }

            if let Some(t) = m.emissive_texture() {
                indices.insert(t.texture().source().index());
            }
        }

        indices
    };

    let (images, buffers) = images
        .iter()
        .enumerate()
        .map(|(index, image)| {
            let pixels = build_rgba_buffer(image);
            let is_srgb = srgb_image_indices.contains(&index);
            VulkanTexture::cmd_from_rgba(
                context,
                command_buffer,
                image.width,
                image.height,
                &pixels,
                !is_srgb,
            )
        })
        .unzip::<_, _, Vec<_>, _>();

    let textures = textures
        .map(|t| {
            let context = Arc::clone(context);
            let image = &images[t.source().index()];
            let view = image.view;
            let sampler = map_sampler(&context, &image.image, &t.sampler());
            Texture {
                context,
                view,
                sampler,
            }
        })
        .collect();

    (
        Textures {
            _images: images,
            textures,
        },
        buffers,
    )
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
    use Format::*;
    match format {
        // actually luma8
        R8 => [pixels[index], pixels[index], pixels[index], std::u8::MAX],
        // actually luma8 with alpha
        R8G8 => [
            pixels[index * 2],
            pixels[index * 2],
            pixels[index * 2],
            pixels[index * 2 + 1],
        ],
        R8G8B8 => [
            pixels[index * 3],
            pixels[index * 3 + 1],
            pixels[index * 3 + 2],
            std::u8::MAX,
        ],
        R8G8B8A8 => [
            pixels[index * 4],
            pixels[index * 4 + 1],
            pixels[index * 4 + 2],
            pixels[index * 4 + 3],
        ],
        R16 | R16G16 | R16G16B16 | R16G16B16A16 | R32G32B32FLOAT | R32G32B32A32FLOAT => {
            panic!("16/32 bits colors are not supported for model textures")
        }
    }
}

fn map_sampler(context: &Arc<Context>, image: &Image, sampler: &Sampler) -> vk::Sampler {
    let min_filter = sampler.min_filter().unwrap_or(MinFilter::Linear);
    let mag_filter = sampler.mag_filter().unwrap_or(MagFilter::Linear);
    let has_mipmaps = has_mipmaps(min_filter);
    let max_lod = if has_mipmaps {
        image.get_mip_levels() as f32
    } else {
        0.25
    };

    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(map_mag_filter(mag_filter))
        .min_filter(map_min_filter(min_filter))
        .address_mode_u(map_wrap_mode(sampler.wrap_s()))
        .address_mode_v(map_wrap_mode(sampler.wrap_t()))
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(has_mipmaps)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(map_mipmap_filter(min_filter))
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(max_lod);

    unsafe {
        context
            .device()
            .create_sampler(&sampler_info, None)
            .expect("Failed to create sampler")
    }
}

fn has_mipmaps(filter: MinFilter) -> bool {
    filter != MinFilter::Linear && filter != MinFilter::Nearest
}

fn map_wrap_mode(wrap_mode: WrappingMode) -> vk::SamplerAddressMode {
    match wrap_mode {
        WrappingMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        WrappingMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        WrappingMode::Repeat => vk::SamplerAddressMode::REPEAT,
    }
}

fn map_min_filter(min_filter: MinFilter) -> vk::Filter {
    match min_filter {
        MinFilter::Nearest => vk::Filter::NEAREST,
        MinFilter::Linear => vk::Filter::LINEAR,
        MinFilter::NearestMipmapNearest => vk::Filter::NEAREST,
        MinFilter::LinearMipmapNearest => vk::Filter::LINEAR,
        MinFilter::NearestMipmapLinear => vk::Filter::NEAREST,
        MinFilter::LinearMipmapLinear => vk::Filter::LINEAR,
    }
}

fn map_mag_filter(mag_filter: MagFilter) -> vk::Filter {
    match mag_filter {
        MagFilter::Nearest => vk::Filter::NEAREST,
        MagFilter::Linear => vk::Filter::LINEAR,
    }
}

fn map_mipmap_filter(min_filter: MinFilter) -> vk::SamplerMipmapMode {
    match min_filter {
        MinFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
        MinFilter::Linear => vk::SamplerMipmapMode::NEAREST,
        MinFilter::NearestMipmapNearest => vk::SamplerMipmapMode::NEAREST,
        MinFilter::LinearMipmapNearest => vk::SamplerMipmapMode::NEAREST,
        MinFilter::NearestMipmapLinear => vk::SamplerMipmapMode::LINEAR,
        MinFilter::LinearMipmapLinear => vk::SamplerMipmapMode::LINEAR,
    }
}
