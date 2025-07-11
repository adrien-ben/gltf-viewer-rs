mod brdf;
mod cubemap;
mod irradiance;
mod pre_filtered;

use brdf::create_brdf_lookup;
use cgmath::{Matrix4, Point3, Vector3};
use cubemap::create_skybox_cubemap;
use irradiance::create_irradiance_map;
use math::*;
use pre_filtered::create_pre_filtered_map;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use vulkan::ash::vk;
use vulkan::{
    Buffer, Context, Descriptors, PipelineParameters, ShaderParameters, Texture, Vertex,
    create_device_local_buffer_with_data, create_pipeline,
};

pub const PRE_FILTERED_MAP_SIZE: u32 = 512;

pub struct Environment {
    skybox: Texture,
    irradiance: Texture,
    pre_filtered: Texture,
    brdf_lookup: Texture,
}

impl Environment {
    pub fn new<P: AsRef<Path>>(context: &Arc<Context>, path: P, resolution: u32) -> Self {
        let skybox = create_skybox_cubemap(context, path, resolution);
        let irradiance = create_irradiance_map(context, &skybox, 32);
        let pre_filtered = create_pre_filtered_map(context, &skybox, 512);
        let brdf_lookup = create_brdf_lookup(context, PRE_FILTERED_MAP_SIZE);

        Self {
            skybox,
            irradiance,
            pre_filtered,
            brdf_lookup,
        }
    }
}

impl Environment {
    pub fn skybox(&self) -> &Texture {
        &self.skybox
    }

    pub fn irradiance(&self) -> &Texture {
        &self.irradiance
    }

    pub fn pre_filtered(&self) -> &Texture {
        &self.pre_filtered
    }

    pub fn brdf_lookup(&self) -> &Texture {
        &self.brdf_lookup
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct SkyboxVertex {
    position: [f32; 3],
}

impl Vertex for SkyboxVertex {
    fn get_bindings_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: size_of::<SkyboxVertex>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attributes_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        }]
    }
}

pub struct SkyboxModel {
    vertices: Buffer,
    indices: Buffer,
}

impl SkyboxModel {
    pub fn new(context: &Arc<Context>) -> Self {
        let indices: [u32; 36] = [
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ];
        let indices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );
        let vertices: [f32; 24] = [
            -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5,
            0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
        ];
        let vertices = create_device_local_buffer_with_data::<u8, _>(
            context,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        SkyboxModel { vertices, indices }
    }
}

impl SkyboxModel {
    pub fn vertices(&self) -> &Buffer {
        &self.vertices
    }

    pub fn indices(&self) -> &Buffer {
        &self.indices
    }
}

fn get_view_matrices() -> [Matrix4<f32>; 6] {
    [
        Matrix4::<f32>::look_at_rh(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Matrix4::<f32>::look_at_rh(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Matrix4::<f32>::look_at_rh(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ),
        Matrix4::<f32>::look_at_rh(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ),
        Matrix4::<f32>::look_at_rh(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Matrix4::<f32>::look_at_rh(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 0.0),
        ),
    ]
}

fn create_descriptors(context: &Arc<Context>, texture: &Texture) -> Descriptors {
    let device = context.device();

    let layout = {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    };

    let pool = {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    };

    let sets = {
        let layouts = [layout];

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let sets = unsafe {
            context
                .device()
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
        };

        let cubemap_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.view)
            .sampler(texture.sampler.unwrap())];

        let descriptor_writes = [vk::WriteDescriptorSet::default()
            .dst_set(sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&cubemap_info)];

        unsafe {
            context
                .device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }

        sets
    };

    Descriptors::new(Arc::clone(context), layout, pool, sets)
}

#[derive(Copy, Clone)]
struct EnvPipelineParameters<'a> {
    vertex_shader_name: &'static str,
    fragment_shader_name: &'static str,
    viewport_info: &'a vk::PipelineViewportStateCreateInfo<'a>,
    rasterizer_info: &'a vk::PipelineRasterizationStateCreateInfo<'a>,
    dynamic_state_info: Option<&'a vk::PipelineDynamicStateCreateInfo<'a>>,
    layout: vk::PipelineLayout,
    format: vk::Format,
}

fn create_env_pipeline<V: Vertex>(
    context: &Arc<Context>,
    params: EnvPipelineParameters,
) -> vk::Pipeline {
    let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];

    create_pipeline::<V>(
        context,
        PipelineParameters {
            vertex_shader_params: ShaderParameters::new(params.vertex_shader_name),
            fragment_shader_params: ShaderParameters::new(params.fragment_shader_name),
            multisampling_info: &multisampling_info,
            viewport_info: params.viewport_info,
            rasterizer_info: params.rasterizer_info,
            dynamic_state_info: params.dynamic_state_info,
            depth_stencil_info: None,
            color_blend_attachments: &color_blend_attachments,
            color_attachment_formats: &[params.format],
            depth_attachment_format: None,
            layout: params.layout,
            parent: None,
            allow_derivatives: false,
        },
    )
}
