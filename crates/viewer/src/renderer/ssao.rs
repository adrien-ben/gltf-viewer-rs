use super::attachments::Attachments;
use super::fullscreen::*;
use super::{create_renderer_pipeline, RendererPipelineParameters, RendererSettings};
use math::{
    cgmath::{InnerSpace, Vector3, Vector4},
    lerp::Lerp,
    rand,
};
use std::mem::size_of;
use std::sync::Arc;
use util::any_as_u8_slice;
use vulkan::ash::vk::{RenderingAttachmentInfo, RenderingInfo};
use vulkan::ash::{vk, Device};
use vulkan::{create_device_local_buffer_with_data, Buffer, Context, SamplerParameters, Texture};

const AO_MAP_FORMAT: vk::Format = vk::Format::R8_UNORM;

const NOISE_SIZE: u32 = 4;

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

#[allow(dead_code)]
struct ConfigUniform {
    ssao_radius: f32,
    ssao_strength: f32,
}

impl SSAOPass {
    pub fn create(
        context: Arc<Context>,
        normals: &Texture,
        depth: &Texture,
        camera_buffers: &[Buffer],
        settings: RendererSettings,
    ) -> Self {
        let kernel_buffer = create_kernel_buffer(&context, settings.ssao_kernel_size);

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
        let pipeline = create_pipeline(&context, pipeline_layout, settings.ssao_kernel_size);

        SSAOPass {
            context,
            kernel_buffer,
            noise_texture,
            descriptors,
            pipeline_layout,
            pipeline,
            kernel_size: settings.ssao_kernel_size,
            ssao_radius: settings.ssao_radius,
            ssao_strength: settings.ssao_strength,
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
            let scale = Lerp::lerp(0.1f32, 1.0f32, scale * scale);

            let v = Vector3::new(x, y, z).normalize() * rand::random::<f32>() * scale;
            [v.x, v.y, v.z, 0.0]
        })
        .collect::<Vec<_>>();

    create_device_local_buffer_with_data::<u8, _>(
        context,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        &kernel,
    )
}

impl SSAOPass {
    pub fn set_inputs(&mut self, normals: &Texture, depth: &Texture) {
        update_static_set(
            &self.context,
            self.descriptors.static_set,
            normals,
            depth,
            &self.noise_texture,
        );
    }

    pub fn set_ssao_kernel_size(&mut self, kernel_size: u32) {
        self.kernel_size = kernel_size;
        self.kernel_buffer = create_kernel_buffer(&self.context, kernel_size);
        update_dynamic_set(
            &self.context,
            self.descriptors.dynamic_set,
            &self.kernel_buffer,
        );
        self.rebuild_pipelines();
    }

    pub fn set_ssao_radius(&mut self, radius: f32) {
        self.ssao_radius = radius;
    }

    pub fn set_ssao_strength(&mut self, strength: f32) {
        self.ssao_strength = strength;
    }

    pub fn rebuild_pipelines(&mut self) {
        let device = self.context.device();

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }

        self.pipeline = create_pipeline(&self.context, self.pipeline_layout, self.kernel_size)
    }

    pub fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        attachments: &Attachments,
        quad_model: &QuadModel,
        frame_index: usize,
    ) {
        let device = self.context.device();

        let extent = vk::Extent2D {
            width: attachments.ssao.image.extent.width,
            height: attachments.ssao.image.extent.height,
        };

        unsafe {
            self.context.device().cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    width: extent.width as _,
                    height: extent.height as _,
                    max_depth: 1.0,
                    ..Default::default()
                }],
            );
            self.context.device().cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    extent,
                    ..Default::default()
                }],
            )
        }

        {
            let attachment_info = RenderingAttachmentInfo::default()
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                })
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(attachments.ssao.view)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = RenderingInfo::default()
                .color_attachments(std::slice::from_ref(&attachment_info))
                .layer_count(1)
                .render_area(vk::Rect2D {
                    extent,
                    ..Default::default()
                });

            unsafe {
                self.context
                    .dynamic_rendering()
                    .cmd_begin_rendering(command_buffer, &rendering_info)
            };
        }

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

        // Push material constants
        unsafe {
            let config = ConfigUniform {
                ssao_radius: self.ssao_radius,
                ssao_strength: self.ssao_strength,
            };
            let data = any_as_u8_slice(&config);

            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                data,
            );
        };

        // Draw
        unsafe { device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1) };

        unsafe {
            self.context
                .dynamic_rendering()
                .cmd_end_rendering(command_buffer)
        };
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

    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(descriptor_count + 2)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_static_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(DEPTH_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(NOISE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

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
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let set = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()[0]
    };

    update_static_set(context, set, normals, depth, noise_texture);

    set
}

fn update_static_set(
    context: &Arc<Context>,
    set: vk::DescriptorSet,
    normals: &Texture,
    depth: &Texture,
    noise_texture: &Texture,
) {
    let normals_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(normals.view)
        .sampler(
            normals
                .sampler
                .expect("SSAO input normals must have a sampler"),
        )];

    let depth_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(depth.view)
        .sampler(depth.sampler.expect("SSAO input depth must have a sampler"))];

    let noise_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(noise_texture.view)
        .sampler(
            noise_texture
                .sampler
                .expect("SSAO input depth must have a sampler"),
        )];

    let descriptor_writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&normals_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(DEPTH_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(NOISE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&noise_info),
    ];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }
}

fn create_dynamic_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(KERNEL_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

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
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let set = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()[0]
    };

    update_dynamic_set(context, set, kernel_buffer);

    set
}

fn update_dynamic_set(context: &Arc<Context>, set: vk::DescriptorSet, kernel_buffer: &Buffer) {
    let kernel_info = [vk::DescriptorBufferInfo::default()
        .buffer(kernel_buffer.buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)];

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(KERNEL_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&kernel_info)];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }
}

fn create_per_frame_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(CAMERA_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

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
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
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
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];

            let descriptor_writes = [vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(CAMERA_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info)];

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

    let push_constant_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: size_of::<ConfigUniform>() as _,
    }];

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_ranges);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_pipeline(
    context: &Arc<Context>,
    layout: vk::PipelineLayout,
    kernel_size: u32,
) -> vk::Pipeline {
    // create_ssao_frag_shader_specialization
    let map_entries = vec![
        // Kernel size
        vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: size_of::<u32>(),
        },
    ];

    let data = unsafe { any_as_u8_slice(&kernel_size) }.to_vec();

    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&map_entries)
        .data(&data);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

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

    create_renderer_pipeline::<QuadVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "ssao",
            fragment_shader_name: "ssao",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            color_attachment_formats: &[AO_MAP_FORMAT],
            depth_attachment_format: None,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling: true,
            parent: None,
        },
    )
}
