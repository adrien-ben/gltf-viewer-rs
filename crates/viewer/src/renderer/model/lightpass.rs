use super::{uniform::*, JointsBuffer, ModelData};
use crate::renderer::attachments::SCENE_COLOR_FORMAT;
use crate::renderer::{create_renderer_pipeline, RendererPipelineParameters, RendererSettings};
use environment::*;
use math::cgmath::Matrix4;
use model::{Model, ModelVertex, Primitive, Texture, Workflow};
use std::mem::offset_of;
use std::{mem::size_of, sync::Arc};
use util::*;
use vulkan::ash::{vk, Device};
use vulkan::{Buffer, Context, Texture as VulkanTexture};

const SAMPLERS_PER_PRIMITIVE: u32 = 8;

const PER_NODE_DYNAMIC_DATA_SET_INDEX: u32 = 0;
const STATIC_DATA_SET_INDEX: u32 = 1;
const PER_PRIMITIVE_DATA_SET_INDEX: u32 = 2;
const INPUT_SET_INDEX: u32 = 3;
const PER_PRIMITIVE_DYNAMIC_DATA_SET_INDEX: u32 = 4;

const CAMERA_UBO_BINDING: u32 = 0;
const CONFIG_UBO_BINDING: u32 = 1;
const LIGHT_UBO_BINDING: u32 = 2;
const TRANSFORMS_UBO_BINDING: u32 = 3;
const SKINS_UBO_BINDING: u32 = 4;
const IRRADIANCE_SAMPLER_BINDING: u32 = 5;
const PRE_FILTERED_SAMPLER_BINDING: u32 = 6;
const BRDF_SAMPLER_BINDING: u32 = 7;
const COLOR_SAMPLER_BINDING: u32 = 8;
const NORMALS_SAMPLER_BINDING: u32 = 9;
const MATERIAL_SAMPLER_BINDING: u32 = 10;
const OCCLUSION_SAMPLER_BINDING: u32 = 11;
const EMISSIVE_SAMPLER_BINDING: u32 = 12;
const CLEARCOAT_FACTOR_SAMPLER_BINDING: u32 = 13;
const CLEARCOAT_ROUGHNESS_SAMPLER_BINDING: u32 = 14;
const CLEARCOAT_NORMAL_SAMPLER_BINDING: u32 = 15;
const AO_MAP_SAMPLER_BINDING: u32 = 16;
const MATERIAL_UBO_BINDING: u32 = 17;

pub struct LightPass {
    context: Arc<Context>,
    dummy_texture: VulkanTexture,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    opaque_pipeline: vk::Pipeline,
    opaque_unculled_pipeline: vk::Pipeline,
    opaque_transparent_pipeline: vk::Pipeline,
    transparent_pipeline: vk::Pipeline,
    output_mode: OutputMode,
    emissive_intensity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Final = 0,
    Color,
    Emissive,
    Metallic,
    Specular,
    Roughness,
    Occlusion,
    Normal,
    Alpha,
    TexCoord0,
    TexCoord1,
    Ssao,
    ClearcoatFactor,
    ClearcoatRoughness,
    ClearcoatNormal,
}

impl OutputMode {
    pub fn all() -> [OutputMode; 15] {
        use OutputMode::*;
        [
            Final,
            Color,
            Emissive,
            Metallic,
            Specular,
            Roughness,
            Occlusion,
            Normal,
            Alpha,
            TexCoord0,
            TexCoord1,
            Ssao,
            ClearcoatFactor,
            ClearcoatRoughness,
            ClearcoatNormal,
        ]
    }

    pub fn from_value(value: usize) -> Option<Self> {
        use OutputMode::*;
        match value {
            0 => Some(Final),
            1 => Some(Color),
            2 => Some(Emissive),
            3 => Some(Metallic),
            4 => Some(Specular),
            5 => Some(Roughness),
            6 => Some(Occlusion),
            7 => Some(Normal),
            8 => Some(Alpha),
            9 => Some(TexCoord0),
            10 => Some(TexCoord1),
            11 => Some(Ssao),
            12 => Some(ClearcoatFactor),
            13 => Some(ClearcoatRoughness),
            14 => Some(ClearcoatNormal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Pass {
    /// Opaque geometry only, will discard masked fragments.
    Opaque = 0,
    /// Transparent geometry but non opaque pixels will be discarded.
    /// It seems required for some models whose materials are BLEND but
    /// alpha is actually 1.0.
    OpaqueTransparent,
    /// Actually transparent gerometry, opaque fragments will be discarded.
    Transparent,
}

impl LightPass {
    pub fn create(
        context: Arc<Context>,
        model_data: &ModelData,
        camera_ubos: &[Buffer],
        config_ubos: &[Buffer],
        environment: &Environment,
        ao_map: Option<&VulkanTexture>,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
        settings: RendererSettings,
    ) -> Self {
        let dummy_texture = VulkanTexture::from_rgba(&context, 1, 1, &[u8::MAX; 4], true);

        let model_rc = model_data
            .model
            .upgrade()
            .expect("Cannot create model renderer because model was dropped");

        let descriptors = create_descriptors(
            &context,
            DescriptorsResources {
                camera_ubos,
                config_ubos,
                model_transform_buffers: &model_data.transform_ubos,
                model_skin_buffers: &model_data.skin_ubos,
                model_materials_buffer: &model_data.materials_ubo,
                light_buffers: &model_data.light_ubos,
                dummy_texture: &dummy_texture,
                environment,
                model: &model_rc.borrow(),
                ao_map: ao_map.unwrap_or(&dummy_texture),
            },
        );

        let pipeline_layout = create_pipeline_layout(context.device(), &descriptors);
        let opaque_pipeline = create_opaque_pipeline(
            &context,
            Pass::Opaque,
            msaa_samples,
            true,
            depth_format,
            pipeline_layout,
        );

        let opaque_unculled_pipeline = create_opaque_pipeline(
            &context,
            Pass::Opaque,
            msaa_samples,
            false,
            depth_format,
            pipeline_layout,
        );

        let opaque_transparent_pipeline = create_opaque_pipeline(
            &context,
            Pass::OpaqueTransparent,
            msaa_samples,
            false,
            depth_format,
            pipeline_layout,
        );

        let transparent_pipeline = create_transparent_pipeline(
            &context,
            msaa_samples,
            depth_format,
            pipeline_layout,
            opaque_pipeline,
        );

        LightPass {
            context,
            dummy_texture,
            descriptors,
            pipeline_layout,
            opaque_pipeline,
            opaque_unculled_pipeline,
            opaque_transparent_pipeline,
            transparent_pipeline,
            output_mode: settings.output_mode,
            emissive_intensity: settings.emissive_intensity,
        }
    }

    pub fn set_ao_map(&mut self, ao_map: Option<&VulkanTexture>) {
        update_input_descriptor_set(
            &self.context,
            self.descriptors.input_set,
            ao_map.unwrap_or(&self.dummy_texture),
        );
    }

    pub fn set_output_mode(&mut self, output_mode: OutputMode) {
        self.output_mode = output_mode;
    }

    pub fn set_emissive_intensity(&mut self, emissive_intensity: f32) {
        self.emissive_intensity = emissive_intensity;
    }
}

impl LightPass {
    pub fn set_model(
        &mut self,
        model_data: &ModelData,
        camera_ubos: &[Buffer],
        config_ubos: &[Buffer],
        environment: &Environment,
        ao_map: Option<&VulkanTexture>,
    ) {
        let model_rc = model_data
            .model
            .upgrade()
            .expect("Cannot create model renderer because model was dropped");

        self.descriptors = create_descriptors(
            &self.context,
            DescriptorsResources {
                camera_ubos,
                config_ubos,
                model_transform_buffers: &model_data.transform_ubos,
                model_skin_buffers: &model_data.skin_ubos,
                model_materials_buffer: &model_data.materials_ubo,
                light_buffers: &model_data.light_ubos,
                dummy_texture: &self.dummy_texture,
                environment,

                model: &model_rc.borrow(),
                ao_map: ao_map.unwrap_or(&self.dummy_texture),
            },
        );
    }

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

        // Bind static data
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                STATIC_DATA_SET_INDEX,
                &[self.descriptors.static_data_set],
                &[],
            )
        };

        // Bind input data
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                INPUT_SET_INDEX,
                &[self.descriptors.input_set],
                &[],
            )
        };

        // Draw opaque primitives
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.opaque_pipeline,
            )
        };

        self.register_model_draw_commands(command_buffer, frame_index, &model, |p| {
            !p.material().is_transparent() && !p.material().is_double_sided()
        });

        // Draw opaque, double sided primitives
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.opaque_unculled_pipeline,
            )
        };

        self.register_model_draw_commands(command_buffer, frame_index, &model, |p| {
            !p.material().is_transparent() && p.material().is_double_sided()
        });

        // Draw opaque transparent
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.opaque_transparent_pipeline,
            )
        };

        self.register_model_draw_commands(command_buffer, frame_index, &model, |p| {
            p.material().is_transparent()
        });

        // Draw transparent primitives
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.transparent_pipeline,
            )
        };

        self.register_model_draw_commands(command_buffer, frame_index, &model, |p| {
            p.material().is_transparent()
        });
    }

    fn register_model_draw_commands<F>(
        &self,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        model: &Model,
        primitive_filter: F,
    ) where
        F: FnMut(&&Primitive) -> bool + Copy,
    {
        let device = self.context.device();
        let model_transform_ubo_offset = self.context.get_ubo_alignment::<Matrix4<f32>>();
        let model_skin_ubo_offset = self.context.get_ubo_alignment::<JointsBuffer>();

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
                    self.pipeline_layout,
                    PER_NODE_DYNAMIC_DATA_SET_INDEX,
                    &self.descriptors.per_node_dynamic_data_sets[frame_index..=frame_index],
                    &[
                        model_transform_ubo_offset * index as u32,
                        model_skin_ubo_offset * skin_index as u32,
                    ],
                )
            };

            for primitive in mesh.primitives().iter().filter(primitive_filter) {
                let primitive_index = primitive.index();
                let material_index = primitive
                    .material_index()
                    .map(|i| i + 1)
                    .unwrap_or_default();
                let material_ubo_offset = self.context.get_ubo_alignment::<MaterialUniform>();

                // Bind descriptor sets
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        PER_PRIMITIVE_DATA_SET_INDEX,
                        &self.descriptors.per_primitive_sets[primitive_index..=primitive_index],
                        &[],
                    );

                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        PER_PRIMITIVE_DYNAMIC_DATA_SET_INDEX,
                        &[self.descriptors.per_primitive_dynamic_data_set],
                        &[material_index as u32 * material_ubo_offset],
                    );
                }

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
}

impl Drop for LightPass {
    fn drop(&mut self) {
        self.context.graphics_queue_wait_idle();
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.opaque_pipeline, None);
            device.destroy_pipeline(self.opaque_unculled_pipeline, None);
            device.destroy_pipeline(self.opaque_transparent_pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

#[derive(Copy, Clone)]
struct DescriptorsResources<'a> {
    camera_ubos: &'a [Buffer],
    config_ubos: &'a [Buffer],
    model_transform_buffers: &'a [Buffer],
    model_skin_buffers: &'a [Buffer],
    model_materials_buffer: &'a Buffer,
    light_buffers: &'a [Buffer],
    dummy_texture: &'a VulkanTexture,
    environment: &'a Environment,
    model: &'a Model,
    ao_map: &'a VulkanTexture,
}

pub struct Descriptors {
    context: Arc<Context>,
    pool: vk::DescriptorPool,
    per_node_dynamic_data_layout: vk::DescriptorSetLayout,
    per_node_dynamic_data_sets: Vec<vk::DescriptorSet>,
    static_data_layout: vk::DescriptorSetLayout,
    static_data_set: vk::DescriptorSet,
    per_primitive_layout: vk::DescriptorSetLayout,
    per_primitive_sets: Vec<vk::DescriptorSet>,
    input_layout: vk::DescriptorSetLayout,
    input_set: vk::DescriptorSet,
    per_primitive_dynamic_data_layout: vk::DescriptorSetLayout,
    per_primitive_dynamic_data_set: vk::DescriptorSet,
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.per_primitive_dynamic_data_layout, None);
            device.destroy_descriptor_set_layout(self.input_layout, None);
            device.destroy_descriptor_set_layout(self.per_node_dynamic_data_layout, None);
            device.destroy_descriptor_set_layout(self.static_data_layout, None);
            device.destroy_descriptor_set_layout(self.per_primitive_layout, None);
        }
    }
}

fn create_descriptors(context: &Arc<Context>, resources: DescriptorsResources) -> Descriptors {
    let pool = create_descriptor_pool(context.device(), resources);

    let per_node_dynamic_data_layout =
        create_per_node_dynamic_data_descriptor_set_layout(context.device());
    let per_node_dynamic_data_sets = create_per_node_dynamic_data_descriptor_sets(
        context,
        pool,
        per_node_dynamic_data_layout,
        resources,
    );

    let static_data_layout = create_static_data_descriptor_set_layout(context.device());
    let static_data_set =
        create_static_data_descriptor_set(context, pool, static_data_layout, resources);

    let per_primitive_layout = create_per_primitive_descriptor_set_layout(context.device());
    let per_primitive_sets =
        create_per_primitive_descriptor_sets(context, pool, per_primitive_layout, resources);

    let input_layout = create_input_descriptor_set_layout(context.device());
    let input_set = create_input_descriptor_set(context, pool, input_layout, resources.ao_map);

    let per_primitive_dynamic_data_layout =
        create_per_primitive_dynamic_data_descriptor_set_layout(context.device());
    let per_primitive_dynamic_data_set = create_per_primitive_dynamic_data_descriptor_set(
        context,
        pool,
        per_primitive_dynamic_data_layout,
        resources,
    );

    Descriptors {
        context: Arc::clone(context),
        pool,
        per_node_dynamic_data_layout,
        per_node_dynamic_data_sets,
        static_data_layout,
        static_data_set,
        per_primitive_layout,
        per_primitive_sets,
        input_layout,
        input_set,
        per_primitive_dynamic_data_layout,
        per_primitive_dynamic_data_set,
    }
}

fn create_descriptor_pool(
    device: &Device,
    descriptors_resources: DescriptorsResources,
) -> vk::DescriptorPool {
    const GLOBAL_TEXTURES_COUNT: u32 = 4; // irradiance, prefiltered, brdf lut, ao
    const PER_FRAME_SETS_COUNT: u32 = 3; // camera, config, lights
    const STATIC_SETS_COUNT: u32 = 1;
    const INPUT_SETS_COUNT: u32 = 1;
    const PER_NODE_SETS_COUNT: u32 = 2;
    const MATERIAL_SETS_COUNT: u32 = 1;

    let descriptor_count = descriptors_resources.camera_ubos.len() as u32;
    let primitive_count = descriptors_resources.model.primitive_count() as u32;
    let textures_desc_count = primitive_count * SAMPLERS_PER_PRIMITIVE;

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: descriptor_count * PER_FRAME_SETS_COUNT,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: descriptor_count * PER_NODE_SETS_COUNT + MATERIAL_SETS_COUNT,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: textures_desc_count + GLOBAL_TEXTURES_COUNT,
        },
    ];

    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(
            descriptor_count
                + STATIC_SETS_COUNT
                + INPUT_SETS_COUNT
                + primitive_count
                + MATERIAL_SETS_COUNT,
        )
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_per_node_dynamic_data_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(CAMERA_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(CONFIG_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(LIGHT_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(TRANSFORMS_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        vk::DescriptorSetLayoutBinding::default()
            .binding(SKINS_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_per_node_dynamic_data_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..resources.camera_ubos.len())
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

    sets.iter().enumerate().for_each(|(i, set)| {
        let camera_ubo = &resources.camera_ubos[i];
        let config_ubo = &resources.config_ubos[i];
        let light_buffer = &resources.light_buffers[i];
        let model_transform_ubo = &resources.model_transform_buffers[i];
        let model_skin_ubo = &resources.model_skin_buffers[i];

        let camera_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(camera_ubo.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let config_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(config_ubo.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let light_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(light_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];

        let model_transform_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(model_transform_ubo.buffer)
            .offset(0)
            .range(size_of::<Matrix4<f32>>() as _)];

        let model_skin_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(model_skin_ubo.buffer)
            .offset(0)
            .range(size_of::<JointsBuffer>() as _)];

        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(CAMERA_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&camera_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(CONFIG_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&config_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(LIGHT_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&light_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(TRANSFORMS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&model_transform_buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(SKINS_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&model_skin_buffer_info),
        ];

        unsafe {
            context
                .device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }
    });

    sets
}

fn create_static_data_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(IRRADIANCE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(PRE_FILTERED_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(BRDF_SAMPLER_BINDING)
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

fn create_static_data_descriptor_set(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
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

    let irradiance_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(resources.environment.irradiance().view)
        .sampler(resources.environment.irradiance().sampler.unwrap())];

    let pre_filtered_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(resources.environment.pre_filtered().view)
        .sampler(resources.environment.pre_filtered().sampler.unwrap())];

    let brdf_lookup_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(resources.environment.brdf_lookup().view)
        .sampler(resources.environment.brdf_lookup().sampler.unwrap())];

    let descriptor_writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(IRRADIANCE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&irradiance_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(PRE_FILTERED_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&pre_filtered_info),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(BRDF_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&brdf_lookup_info),
    ];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }

    set
}

fn create_per_primitive_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(COLOR_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(MATERIAL_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(OCCLUSION_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(EMISSIVE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(CLEARCOAT_FACTOR_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(CLEARCOAT_ROUGHNESS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(CLEARCOAT_NORMAL_SAMPLER_BINDING)
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

fn create_per_primitive_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..resources.model.primitive_count())
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
            let normals_info = create_descriptor_image_info(
                material.get_normals_texture_index(),
                textures,
                resources.dummy_texture,
            );

            let material_texture = match material.get_workflow() {
                Workflow::MetallicRoughness(workflow) => workflow.get_metallic_roughness_texture(),
                Workflow::SpecularGlossiness(workflow) => {
                    workflow.get_specular_glossiness_texture()
                }
            };
            let material_info = create_descriptor_image_info(
                material_texture.map(|t| t.get_index()),
                textures,
                resources.dummy_texture,
            );
            let occlusion_info = create_descriptor_image_info(
                material.get_occlusion_texture_index(),
                textures,
                resources.dummy_texture,
            );
            let emissive_info = create_descriptor_image_info(
                material.get_emissive_texture_index(),
                textures,
                resources.dummy_texture,
            );

            let clearcoat = material.get_clearcoat().unwrap_or_default();
            let clearcoat_factor_info = create_descriptor_image_info(
                clearcoat.factor_texture_index(),
                textures,
                resources.dummy_texture,
            );
            let clearcoat_roughness_info = create_descriptor_image_info(
                clearcoat.roughness_texture_index(),
                textures,
                resources.dummy_texture,
            );
            let clearcoat_normal_info = create_descriptor_image_info(
                clearcoat.normal_texture_index(),
                textures,
                resources.dummy_texture,
            );

            let set = sets[primitive_index];
            primitive_index += 1;

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(COLOR_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&albedo_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(NORMALS_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&normals_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(MATERIAL_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&material_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(OCCLUSION_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&occlusion_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(EMISSIVE_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&emissive_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(CLEARCOAT_FACTOR_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&clearcoat_factor_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(CLEARCOAT_ROUGHNESS_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&clearcoat_roughness_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(CLEARCOAT_NORMAL_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&clearcoat_normal_info),
            ];

            unsafe {
                context
                    .device()
                    .update_descriptor_sets(&descriptor_writes, &[])
            }
        }
    }

    sets
}

fn create_input_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(AO_MAP_SAMPLER_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_input_descriptor_set(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    ao_map: &VulkanTexture,
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

    update_input_descriptor_set(context, set, ao_map);

    set
}

fn create_per_primitive_dynamic_data_descriptor_set_layout(
    device: &Device,
) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(MATERIAL_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_per_primitive_dynamic_data_descriptor_set(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
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

    let material_buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(resources.model_materials_buffer.buffer)
        .offset(0)
        .range(size_of::<MaterialUniform>() as _)];

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(MATERIAL_UBO_BINDING)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .buffer_info(&material_buffer_info)];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }

    set
}

fn update_input_descriptor_set(
    context: &Arc<Context>,
    set: vk::DescriptorSet,
    ao_map: &VulkanTexture,
) {
    let ao_map_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(ao_map.view)
        .sampler(ao_map.sampler.unwrap())];

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(AO_MAP_SAMPLER_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&ao_map_info)];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }
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

    [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(view)
        .sampler(sampler)]
}

fn create_pipeline_layout(device: &Device, descriptors: &Descriptors) -> vk::PipelineLayout {
    let layouts = [
        descriptors.per_node_dynamic_data_layout,
        descriptors.static_data_layout,
        descriptors.per_primitive_layout,
        descriptors.input_layout,
        descriptors.per_primitive_dynamic_data_layout,
    ];

    let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);

    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_opaque_pipeline(
    context: &Arc<Context>,
    pass: Pass,
    msaa_samples: vk::SampleCountFlags,
    enable_face_culling: bool,
    depth_format: vk::Format,
    layout: vk::PipelineLayout,
) -> vk::Pipeline {
    // create_model_frag_shader_specialization
    let map_entries = vec![
        vk::SpecializationMapEntry {
            constant_id: 0,
            offset: offset_of!(ModelShaderConstants, max_light_count) as _,
            size: size_of::<u32>(),
        },
        vk::SpecializationMapEntry {
            constant_id: 1,
            offset: offset_of!(ModelShaderConstants, max_reflection_lod) as _,
            size: size_of::<u32>(),
        },
        vk::SpecializationMapEntry {
            constant_id: 2,
            offset: offset_of!(ModelShaderConstants, pass) as _,
            size: size_of::<u32>(),
        },
    ];

    let max_reflection_lod = (PRE_FILTERED_MAP_SIZE as f32).log2().floor() as u32;
    let constants = ModelShaderConstants {
        max_light_count: MAX_LIGHT_COUNT as _,
        max_reflection_lod,
        pass: pass as _,
    };

    let data = Vec::from(unsafe { any_as_u8_slice(&constants) });
    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&map_entries)
        .data(&data);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
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

    create_renderer_pipeline::<ModelVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "model",
            fragment_shader_name: "model",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            msaa_samples,
            color_attachment_formats: &[SCENE_COLOR_FORMAT],
            depth_attachment_format: Some(depth_format),
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling,
            parent: None,
        },
    )
}

fn create_transparent_pipeline(
    context: &Arc<Context>,
    msaa_samples: vk::SampleCountFlags,
    depth_format: vk::Format,
    layout: vk::PipelineLayout,
    parent: vk::Pipeline,
) -> vk::Pipeline {
    // create_model_frag_shader_specialization
    let map_entries = vec![
        vk::SpecializationMapEntry {
            constant_id: 0,
            offset: offset_of!(ModelShaderConstants, max_light_count) as _,
            size: size_of::<u32>(),
        },
        vk::SpecializationMapEntry {
            constant_id: 1,
            offset: offset_of!(ModelShaderConstants, max_reflection_lod) as _,
            size: size_of::<u32>(),
        },
        vk::SpecializationMapEntry {
            constant_id: 2,
            offset: offset_of!(ModelShaderConstants, pass) as _,
            size: size_of::<u32>(),
        },
    ];

    let max_reflection_lod = (PRE_FILTERED_MAP_SIZE as f32).log2().floor() as u32;
    let constants = ModelShaderConstants {
        max_light_count: MAX_LIGHT_COUNT as _,
        max_reflection_lod,
        pass: Pass::Transparent as _,
    };

    let data = Vec::from(unsafe { any_as_u8_slice(&constants) });
    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(&map_entries)
        .data(&data);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
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
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)];

    create_renderer_pipeline::<ModelVertex>(
        context,
        RendererPipelineParameters {
            vertex_shader_name: "model",
            fragment_shader_name: "model",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            msaa_samples,
            color_attachment_formats: &[SCENE_COLOR_FORMAT],
            depth_attachment_format: Some(depth_format),
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachments: &color_blend_attachments,
            enable_face_culling: false,
            parent: Some(parent),
        },
    )
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct ModelShaderConstants {
    max_light_count: u32,
    max_reflection_lod: u32,
    pass: u32,
}
