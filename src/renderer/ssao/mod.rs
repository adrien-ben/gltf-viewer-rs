mod renderpass;

use super::fullscreen::*;
use super::{create_renderer_pipeline, RendererPipelineParameters};
use math::{
    cgmath::{InnerSpace, Vector3, Vector4},
    lerp::Lerp,
    rand,
};
pub use renderpass::RenderPass as SSAORenderPass;
use std::mem::size_of;
use std::sync::Arc;
use util::any_as_u8_slice;
use vulkan::ash::{version::DeviceV1_0, vk, Device};
use vulkan::{
    create_device_local_buffer_with_data, Buffer, Context, SamplerParameters, SwapchainProperties,
    Texture,
};

const NOISE_SIZE: u32 = 8;
const DEFAULT_KERNEL_SIZE: u32 = 32;
const DEFAULT_RADIUS: f32 = 0.15;
const DEFAULT_STRENGTH: f32 = 1.0;

const STATIC_SET_INDEX: u32 = 0;
const DYNAMIC_SET_INDEX: u32 = 1;
const PER_FRAME_SET_INDEX: u32 = 2;
const NORMALS_SAMPLER_BINDING: u32 = 0;
const DEPTH_SAMPLER_BINDING: u32 = 1;
const NOISE_SAMPLER_BINDING: u32 = 2;
const KERNEL_UBO_BINDING: u32 = 3;
const CAMERA_UBO_BINDING: u32 = 4;

pub struct SSAOPass {
    context: Arc<Context>,
    kernel_buffer: Buffer,
    noise_texture: Texture,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    kernel_size: u32,
    ssao_radius: f32,
    ssao_strength: f32,
}

impl SSAOPass {
    pub fn create(
        context: Arc<Context>,
        swapchain_props: SwapchainProperties,
        render_pass: &SSAORenderPass,
        normals: &Texture,
        depth: &Texture,
        camera_buffers: &[Buffer],
    ) -> Self {
        let kernel_buffer = create_kernel_buffer(&context, DEFAULT_KERNEL_SIZE);

        let noise_texture = {
            let size = NOISE_SIZE * NOISE_SIZE;
            let mut noise = Vec::<f32>::new();

            (0..size).for_each(|_| {
                let x = rand::random::<f32>() * 2.0 - 1.0;
                let y = rand::random::<f32>() * 2.0 - 1.0;
                let z = 0.0;
                let w = 0.0;

                let v: [f32; 4] = Vector4::new(x, y, z, w).normalize().into();
                noise.extend_from_slice(&v);
            });

            Texture::from_rgba_32(
                &context,
                NOISE_SIZE,
                NOISE_SIZE,
                false,
                &noise,
                Some(SamplerParameters {
                    mag_filter: vk::Filter::NEAREST,
                    min_filter: vk::Filter::NEAREST,
                    ..Default::default()
                }),
            )
        };

        let descriptors = create_descriptors(
            &context,
            normals,
            depth,
            &noise_texture,
            &kernel_buffer,
            camera_buffers,
        );
        let pipeline_layout = create_pipeline_layout(context.device(), &descriptors);
        let pipeline = create_pipeline(
            &context,
            swapchain_props,
            render_pass.get_render_pass(),
            pipeline_layout,
            DEFAULT_KERNEL_SIZE,
            DEFAULT_RADIUS,
            DEFAULT_STRENGTH,
        );

        SSAOPass {
            context,
            kernel_buffer,
            noise_texture,
            descriptors,
            pipeline_layout,
            pipeline,
            kernel_size: DEFAULT_KERNEL_SIZE,
            ssao_radius: DEFAULT_RADIUS,
            ssao_strength: DEFAULT_STRENGTH,
        }
    }
}

fn create_kernel_buffer(context: &Arc<Context>, kernel_size: u32) -> Buffer {
    let kernel = (0..kernel_size)
        .map(|i| {
            let x = rand::random::<f32>() * 2.0 - 1.0;
            let y = rand::random::<f32>() * 2.0 - 1.0;
            let z = rand::random::<f32>();

            let scale = i as f32 / kernel_size as f32;
            let scale = (0.1f32).lerp(1.0f32, scale * scale);

            let v = Vector3::new(x, y, z).normalize() * rand::random::<f32>() * scale;
            [v.x, v.y, v.z, 0.0]
        })
        .collect::<Vec<_>>();

    create_device_local_buffer_with_data::<u8, _>(
        &context,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        &kernel,
    )
}

impl SSAOPass {
    pub fn set_inputs(&mut self, normals: &Texture, depth: &Texture) {
        unsafe {
            self.context
                .device()
                .free_descriptor_sets(self.descriptors.pool, &[self.descriptors.static_set]);
        }
        self.descriptors.static_set = create_static_set(
            &self.context,
            self.descriptors.pool,
            self.descriptors.static_set_layout,
            normals,
            depth,
            &self.noise_texture,
        );
    }

    pub fn set_ssao_kernel_size(&mut self, kernel_size: u32) {
        unsafe {
            self.context
                .device()
                .free_descriptor_sets(self.descriptors.pool, &[self.descriptors.dynamic_set]);
        }

        self.kernel_size = kernel_size;
        self.kernel_buffer = create_kernel_buffer(&self.context, kernel_size);
        self.descriptors.dynamic_set = create_dynamic_set(
            &self.context,
            self.descriptors.pool,
            self.descriptors.dynamic_set_layout,
            &self.kernel_buffer,
        );
    }

    pub fn set_ssao_radius(&mut self, radius: f32) {
        self.ssao_radius = radius;
    }

    pub fn set_ssao_strength(&mut self, strength: f32) {
        self.ssao_strength = strength;
    }

    pub fn rebuild_pipelines(
        &mut self,
        swapchain_properties: SwapchainProperties,
        render_pass: &SSAORenderPass,
    ) {
        let device = self.context.device();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }

        self.pipeline = create_pipeline(
            &self.context,
            swapchain_properties,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            self.kernel_size,
            self.ssao_radius,
            self.ssao_strength,
        )
    }

    pub fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        quad_model: &QuadModel,
        frame_index: usize,
    ) {
        let device = self.context.device();
        // Bind pipeline
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            )
        };

        // Bind buffers
        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[quad_model.vertices.buffer], &[0]);
            device.cmd_bind_index_buffer(
                command_buffer,
                quad_model.indices.buffer,
                0,
                vk::IndexType::UINT16,
            );
        }

        // Bind static descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                STATIC_SET_INDEX,
                &[self.descriptors.static_set],
                &[],
            )
        };

        // Bind dynamic descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                DYNAMIC_SET_INDEX,
                &[self.descriptors.dynamic_set],
                &[],
            )
        };

        // Bind per frame descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                PER_FRAME_SET_INDEX,
                &self.descriptors.per_frame_sets[frame_index..=frame_index],
                &[],
            )
        };

        // Draw
        unsafe { device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1) };
    }
}

impl Drop for SSAOPass {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub struct Descriptors {
    context: Arc<Context>,
    pool: vk::DescriptorPool,
    static_set_layout: vk::DescriptorSetLayout,
    static_set: vk::DescriptorSet,
    dynamic_set_layout: vk::DescriptorSetLayout,
    dynamic_set: vk::DescriptorSet,
    per_frame_set_layout: vk::DescriptorSetLayout,
    per_frame_sets: Vec<vk::DescriptorSet>,
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.static_set_layout, None);
            device.destroy_descriptor_set_layout(self.dynamic_set_layout, None);
            device.destroy_descriptor_set_layout(self.per_frame_set_layout, None);
        }
    }
}

fn create_descriptors(
    context: &Arc<Context>,
    normals: &Texture,
    depth: &Texture,
    noise_texture: &Texture,
    kernel_buffer: &Buffer,
    camera_buffers: &[Buffer],
) -> Descriptors {
    let pool = create_descriptor_pool(context.device(), camera_buffers.len() as _);

    let static_set_layout = create_static_set_layout(context.device());
    let static_set = create_static_set(
        context,
        pool,
        static_set_layout,
        normals,
        depth,
        noise_texture,
    );

    let dynamic_set_layout = create_dynamic_set_layout(context.device());
    let dynamic_set = create_dynamic_set(context, pool, dynamic_set_layout, kernel_buffer);

    let per_frame_set_layout = create_per_frame_set_layout(context.device());
    let per_frame_sets = create_per_frame_sets(context, pool, per_frame_set_layout, camera_buffers);

    Descriptors {
        context: Arc::clone(context),
        pool,
        static_set_layout,
        static_set,
        dynamic_set_layout,
        dynamic_set,
        per_frame_set_layout,
        per_frame_sets,
    }
}

fn create_descriptor_pool(device: &Device, descriptor_count: u32) -> vk::DescriptorPool {
    const TEXTURE_COUNT: u32 = 3;
    const DYNAMIC_UNIFORM_BUFFER_COUNT: u32 = 1;
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: TEXTURE_COUNT,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: descriptor_count + DYNAMIC_UNIFORM_BUFFER_COUNT,
        },
    ];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(descriptor_count + 2)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_static_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(DEPTH_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(NOISE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_static_set(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    normals: &Texture,
    depth: &Texture,
    noise_texture: &Texture,
) -> vk::DescriptorSet {
    let layouts = [layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let set = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()[0]
    };

    let normals_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(normals.view)
        .sampler(
            normals
                .sampler
                .expect("SSAO input normals must have a sampler"),
        )
        .build()];

    let depth_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(depth.view)
        .sampler(depth.sampler.expect("SSAO input depth must have a sampler"))
        .build()];

    let noise_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(noise_texture.view)
        .sampler(
            noise_texture
                .sampler
                .expect("SSAO input depth must have a sampler"),
        )
        .build()];

    let descriptor_writes = [
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&normals_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(DEPTH_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(NOISE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&noise_info)
            .build(),
    ];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }

    set
}

fn create_dynamic_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(KERNEL_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build()];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_dynamic_set(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    kernel_buffer: &Buffer,
) -> vk::DescriptorSet {
    let layouts = [layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let set = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()[0]
    };

    let kernel_info = [vk::DescriptorBufferInfo::builder()
        .buffer(kernel_buffer.buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)
        .build()];

    let descriptor_writes = [vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(KERNEL_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&kernel_info)
        .build()];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }

    set
}

fn create_per_frame_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(CAMERA_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)
        .build()];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_per_frame_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    camera_buffers: &[Buffer],
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..camera_buffers.len())
        .map(|_| layout)
        .collect::<Vec<_>>();
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()
    };

    sets.iter()
        .zip(camera_buffers.iter())
        .for_each(|(set, buffer)| {
            let buffer_info = [vk::DescriptorBufferInfo::builder()
                .buffer(buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()];

            let descriptor_writes = [vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(CAMERA_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info)
                .build()];

            unsafe {
                context
                    .device()
                    .update_descriptor_sets(&descriptor_writes, &[])
            }
        });

    sets
}

fn create_pipeline_layout(device: &Device, descriptors: &Descriptors) -> vk::PipelineLayout {
    let layouts = [
        descriptors.static_set_layout,
        descriptors.dynamic_set_layout,
        descriptors.per_frame_set_layout,
    ];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_pipeline(
    context: &Arc<Context>,
    swapchain_properties: SwapchainProperties,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    kernel_size: u32,
    ssao_radius: f32,
    ssao_strength: f32,
) -> vk::Pipeline {
    let (specialization_info, _map_entries, _data) =
        create_ssao_frag_shader_specialization(kernel_size, ssao_radius, ssao_strength);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];

    create_renderer_pipeline::<QuadVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "ssao",
            fragment_shader_name: "ssao",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            swapchain_properties,
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            render_pass,
            subpass: 0,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling: true,
            parent: None,
        },
    )
}

fn create_ssao_frag_shader_specialization(
    saao_samples_count: u32,
    ssao_radius: f32,
    ssao_strength: f32,
) -> (
    vk::SpecializationInfo,
    Vec<vk::SpecializationMapEntry>,
    Vec<u8>,
) {
    let map_entries = vec![
        // Kernel size
        vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: size_of::<u32>(),
        },
        // Radius
        vk::SpecializationMapEntry {
            constant_id: 1,
            offset: size_of::<u32>() as _,
            size: size_of::<f32>(),
        },
        // Strength
        vk::SpecializationMapEntry {
            constant_id: 2,
            offset: (2 * size_of::<u32>()) as _,
            size: size_of::<f32>(),
        },
    ];

    let mut data = Vec::new();
    data.extend_from_slice(unsafe { any_as_u8_slice(&saao_samples_count) });
    data.extend_from_slice(unsafe { any_as_u8_slice(&ssao_radius) });
    data.extend_from_slice(unsafe { any_as_u8_slice(&ssao_strength) });

    let specialization_info = vk::SpecializationInfo::builder()
        .map_entries(&map_entries)
        .data(&data)
        .build();

    (specialization_info, map_entries, data)
}
