mod renderpass;

pub use renderpass::RenderPass as GBufferRenderPass;

use super::{JointsBuffer, ModelData};
use crate::renderer::{create_renderer_pipeline, RendererPipelineParameters};
use math::cgmath::Matrix4;
use model::{Material, Model, ModelVertex, Primitive, Texture};
use std::{mem::size_of, sync::Arc};
use util::any_as_u8_slice;
use vulkan::ash::{version::DeviceV1_0, vk, Device};
use vulkan::{Buffer, Context, SwapchainProperties, Texture as VulkanTexture};

const DYNAMIC_DATA_SET_INDEX: u32 = 0;
const PER_PRIMITIVE_DATA_SET_INDEX: u32 = 1;

const CAMERA_UBO_BINDING: u32 = 0;
const TRANSFORMS_UBO_BINDING: u32 = 1;
const SKINS_UBO_BINDING: u32 = 2;
const COLOR_SAMPLER_BINDING: u32 = 3;

pub struct GBufferPass {
    context: Arc<Context>,
    _dummy_texture: VulkanTexture,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    culled_pipeline: vk::Pipeline,
    unculled_pipeline: vk::Pipeline,
}

impl GBufferPass {
    pub fn create(
        context: Arc<Context>,
        model_data: &ModelData,
        camera_buffers: &[Buffer],
        swapchain_props: SwapchainProperties,
        render_pass: &GBufferRenderPass,
    ) -> Self {
        let dummy_texture = VulkanTexture::from_rgba(&context, 1, 1, &[std::u8::MAX; 4]);

        let model_rc = model_data
            .model
            .upgrade()
            .expect("Cannot create model renderer because model was dropped");

        let descriptors = create_descriptors(
            &context,
            DescriptorsResources {
                camera_buffers,
                model_transform_buffers: &model_data.transform_ubos,
                model_skin_buffers: &model_data.skin_ubos,
                model: &model_rc.borrow(),
                dummy_texture: &dummy_texture,
            },
        );

        let pipeline_layout = create_pipeline_layout(context.device(), &descriptors);
        let culled_pipeline = create_pipeline(
            &context,
            swapchain_props,
            render_pass.get_render_pass(),
            pipeline_layout,
            true,
        );
        let unculled_pipeline = create_pipeline(
            &context,
            swapchain_props,
            render_pass.get_render_pass(),
            pipeline_layout,
            false,
        );

        GBufferPass {
            context,
            _dummy_texture: dummy_texture,
            descriptors,
            pipeline_layout,
            culled_pipeline,
            unculled_pipeline,
        }
    }

    pub fn rebuild_pipelines(
        &mut self,
        swapchain_props: SwapchainProperties,
        render_pass: &GBufferRenderPass,
    ) {
        let device = self.context.device();

        unsafe {
            device.destroy_pipeline(self.unculled_pipeline, None);
            device.destroy_pipeline(self.culled_pipeline, None);
        }

        self.culled_pipeline = create_pipeline(
            &self.context,
            swapchain_props,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            true,
        );

        self.unculled_pipeline = create_pipeline(
            &self.context,
            swapchain_props,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            false,
        );
    }
}

impl GBufferPass {
    pub fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        model_data: &ModelData,
    ) {
        let device = self.context.device();
        let model = model_data
            .model
            .upgrade()
            .expect("Cannot register draw commands because model was dropped");
        let model = model.borrow();

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.culled_pipeline,
            )
        };

        // Draw opaque primitives
        register_model_draw_commands(
            &self.context,
            self.pipeline_layout,
            command_buffer,
            &model,
            &self.descriptors.dynamic_data_sets[frame_index..=frame_index],
            &self.descriptors.per_primitive_sets,
            |p| !p.material().is_transparent() && !p.material().is_double_sided(),
        );

        // Bind opaque without culling pipeline
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.unculled_pipeline,
            )
        };

        // Draw opaque, double sided primitives
        register_model_draw_commands(
            &self.context,
            self.pipeline_layout,
            command_buffer,
            &model,
            &self.descriptors.dynamic_data_sets[frame_index..=frame_index],
            &self.descriptors.per_primitive_sets,
            |p| !p.material().is_transparent() && p.material().is_double_sided(),
        );
    }
}

impl Drop for GBufferPass {
    fn drop(&mut self) {
        self.context.graphics_queue_wait_idle();
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.unculled_pipeline, None);
            device.destroy_pipeline(self.culled_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[derive(Copy, Clone)]
struct DescriptorsResources<'a> {
    camera_buffers: &'a [Buffer],
    model_transform_buffers: &'a [Buffer],
    model_skin_buffers: &'a [Buffer],
    model: &'a Model,
    dummy_texture: &'a VulkanTexture,
}

pub struct Descriptors {
    context: Arc<Context>,
    pool: vk::DescriptorPool,
    dynamic_data_layout: vk::DescriptorSetLayout,
    dynamic_data_sets: Vec<vk::DescriptorSet>,
    per_primitive_layout: vk::DescriptorSetLayout,
    per_primitive_sets: Vec<vk::DescriptorSet>,
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.dynamic_data_layout, None);
        }
    }
}

fn create_descriptors(context: &Arc<Context>, resources: DescriptorsResources) -> Descriptors {
    let pool = create_descriptor_pool(context.device(), resources);

    let dynamic_data_layout = create_dynamic_data_descriptor_set_layout(context.device());
    let dynamic_data_sets =
        create_dynamic_data_descriptor_sets(context, pool, dynamic_data_layout, resources);

    let per_primitive_layout = create_per_primitive_descriptor_set_layout(context.device());
    let per_primitive_sets =
        create_per_primitive_descriptor_sets(context, pool, per_primitive_layout, resources);

    Descriptors {
        context: Arc::clone(context),
        pool,
        dynamic_data_layout,
        dynamic_data_sets,
        per_primitive_layout,
        per_primitive_sets,
    }
}

fn create_descriptor_pool(
    device: &Device,
    descriptors_resources: DescriptorsResources,
) -> vk::DescriptorPool {
    let descriptor_count = descriptors_resources.camera_buffers.len() as u32;
    let primitive_count = descriptors_resources.model.primitive_count() as u32;

    let pool_sizes = [
        // Camera
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count,
        },
        // Transforms & skins
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: descriptor_count * 2,
        },
        // Color sampler
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: primitive_count,
        },
    ];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(descriptor_count + primitive_count);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_dynamic_data_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(CAMERA_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(TRANSFORMS_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(SKINS_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build(),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_dynamic_data_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..resources.camera_buffers.len())
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

    sets.iter().enumerate().for_each(|(i, set)| {
        let camera_ubo = &resources.camera_buffers[i];
        let model_transform_ubo = &resources.model_transform_buffers[i];
        let model_skin_ubo = &resources.model_skin_buffers[i];

        let camera_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(camera_ubo.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()];

        let model_transform_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(model_transform_ubo.buffer)
            .offset(0)
            .range(size_of::<Matrix4<f32>>() as _)
            .build()];

        let model_skin_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(model_skin_ubo.buffer)
            .offset(0)
            .range(size_of::<JointsBuffer>() as _)
            .build()];

        let descriptor_writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(CAMERA_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&camera_buffer_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(TRANSFORMS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&model_transform_buffer_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(SKINS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&model_skin_buffer_info)
                .build(),
        ];

        unsafe {
            context
                .device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }
    });

    sets
}

fn create_per_primitive_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::builder()
        .binding(COLOR_SAMPLER_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
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

fn create_per_primitive_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..resources.model.primitive_count())
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

    let model = resources.model;
    let textures = resources.model.textures();
    let mut primitive_index = 0;
    for mesh in model.meshes() {
        for primitive in mesh.primitives() {
            let material = primitive.material();
            let albedo_info = create_descriptor_image_info(
                material.get_color_texture_index(),
                textures,
                resources.dummy_texture,
            );

            let set = sets[primitive_index];
            primitive_index += 1;

            let descriptor_writes = [vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(COLOR_SAMPLER_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&albedo_info)
                .build()];

            unsafe {
                context
                    .device()
                    .update_descriptor_sets(&descriptor_writes, &[])
            }
        }
    }

    sets
}

fn create_descriptor_image_info(
    index: Option<usize>,
    textures: &[Texture],
    dummy_texture: &VulkanTexture,
) -> [vk::DescriptorImageInfo; 1] {
    let (view, sampler) = index
        .map(|i| &textures[i])
        .map_or((dummy_texture.view, dummy_texture.sampler.unwrap()), |t| {
            (t.get_view(), t.get_sampler())
        });

    [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler)
        .build()]
}

fn create_pipeline_layout(device: &Device, descriptors: &Descriptors) -> vk::PipelineLayout {
    let layouts = [
        descriptors.dynamic_data_layout,
        descriptors.per_primitive_layout,
    ];
    let constant_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: size_of::<MaterialUniform>() as _,
    }];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .push_constant_ranges(&constant_ranges);

    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_pipeline(
    context: &Arc<Context>,
    swapchain_properties: SwapchainProperties,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    enable_face_culling: bool,
) -> vk::Pipeline {
    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
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

    create_renderer_pipeline::<ModelVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "gbuffer",
            fragment_shader_name: "gbuffer",
            vertex_shader_specialization: None,
            fragment_shader_specialization: None,
            swapchain_properties,
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            render_pass,
            subpass: 0,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling,
            parent: None,
        },
    )
}

fn register_model_draw_commands<F>(
    context: &Context,
    pipeline_layout: vk::PipelineLayout,
    command_buffer: vk::CommandBuffer,
    model: &Model,
    dynamic_descriptors: &[vk::DescriptorSet],
    per_primitive_descriptors: &[vk::DescriptorSet],
    primitive_filter: F,
) where
    F: FnMut(&&Primitive) -> bool + Copy,
{
    let device = context.device();
    let model_transform_ubo_offset = context.get_ubo_alignment::<Matrix4<f32>>();
    let model_skin_ubo_offset = context.get_ubo_alignment::<JointsBuffer>();

    for (index, node) in model
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.mesh_index().is_some())
        .enumerate()
    {
        let mesh = model.mesh(node.mesh_index().unwrap());
        let skin_index = node.skin_index().unwrap_or(0);

        // Bind descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                DYNAMIC_DATA_SET_INDEX,
                &dynamic_descriptors,
                &[
                    model_transform_ubo_offset * index as u32,
                    model_skin_ubo_offset * skin_index as u32,
                ],
            )
        };

        for primitive in mesh.primitives().iter().filter(primitive_filter) {
            let primitive_index = primitive.index();

            unsafe {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    PER_PRIMITIVE_DATA_SET_INDEX,
                    &per_primitive_descriptors[primitive_index..=primitive_index],
                    &[],
                )
            };

            unsafe {
                device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[primitive.vertices().buffer().buffer],
                    &[primitive.vertices().offset()],
                );
            }

            if let Some(index_buffer) = primitive.indices() {
                unsafe {
                    device.cmd_bind_index_buffer(
                        command_buffer,
                        index_buffer.buffer().buffer,
                        index_buffer.offset(),
                        index_buffer.index_type(),
                    );
                }
            }

            // Push material constants
            unsafe {
                let material: MaterialUniform = primitive.material().into();
                let material_contants = any_as_u8_slice(&material);
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &material_contants,
                );
            };

            // Draw geometry
            match primitive.indices() {
                Some(index_buffer) => {
                    unsafe {
                        device.cmd_draw_indexed(
                            command_buffer,
                            index_buffer.element_count(),
                            1,
                            0,
                            0,
                            0,
                        )
                    };
                }
                None => {
                    unsafe {
                        device.cmd_draw(
                            command_buffer,
                            primitive.vertices().element_count(),
                            1,
                            0,
                            0,
                        )
                    };
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct MaterialUniform {
    alpha: f32,
    color_texture_channel: u32,
    alpha_mode: u32,
    alpha_cutoff: f32,
}

impl MaterialUniform {
    const NO_TEXTURE_ID: u32 = std::u8::MAX as u32;
}

impl<'a> From<Material> for MaterialUniform {
    fn from(material: Material) -> MaterialUniform {
        let alpha = material.get_color()[3];
        let color_texture_channel = material
            .get_color_texture()
            .map_or(Self::NO_TEXTURE_ID, |info| info.get_channel());
        let alpha_mode = material.get_alpha_mode();
        let alpha_cutoff = material.get_alpha_cutoff();

        MaterialUniform {
            alpha,
            color_texture_channel,
            alpha_mode,
            alpha_cutoff,
        }
    }
}
