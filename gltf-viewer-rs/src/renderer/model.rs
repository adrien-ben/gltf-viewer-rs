use super::{create_renderer_pipeline, RendererPipelineParameters};
use ash::{version::DeviceV1_0, vk, Device};
use environment::*;
use math::cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector4};
use model::*;
use std::{mem::size_of, sync::Arc};
use util::*;
use vulkan::*;

const DYNAMIC_DATA_SET_INDEX: u32 = 0;
const STATIC_DATA_SET_INDEX: u32 = 1;
const PER_PRIMITIVE_DATA_SET_INDEX: u32 = 2;

const CAMERA_UBO_BINDING: u32 = 0;
const LIGHT_UBO_BINDING: u32 = 1;
const TRANSFORMS_UBO_BINDING: u32 = 2;
const SKINS_UBO_BINDING: u32 = 3;
const IRRADIANCE_SAMPLER_BINDING: u32 = 4;
const PRE_FILTERED_SAMPLER_BINDING: u32 = 5;
const BRDF_SAMPLER_BINDING: u32 = 6;
const COLOR_SAMPLER_BINDING: u32 = 7;
const NORMALS_SAMPLER_BINDING: u32 = 8;
const METALLIC_ROUGHNESS_SAMPLER_BINDING: u32 = 9;
const OCCLUSION_SAMPLER_BINDING: u32 = 10;
const EMISSIVE_SAMPLER_BINDING: u32 = 11;

const DIRECTIONAL_LIGHT_TYPE: u32 = 0;
const POINT_LIGHT_TYPE: u32 = 1;
const SPOT_LIGHT_TYPE: u32 = 2;

const DEFAULT_LIGHT_DIRECTION: [f32; 4] = [0.0, 0.0, -1.0, 0.0];

type JointsBuffer = [Matrix4<f32>; MAX_JOINTS_PER_MESH];

pub struct ModelRenderer {
    context: Arc<Context>,
    model: Model,
    _dummy_texture: Texture,
    transform_ubos: Vec<Buffer>,
    skin_ubos: Vec<Buffer>,
    skin_matrices: Vec<Vec<JointsBuffer>>,
    light_buffers: Vec<Buffer>,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    opaque_pipeline: vk::Pipeline,
    transparent_pipeline: vk::Pipeline,
}

impl ModelRenderer {
    pub fn create(
        context: Arc<Context>,
        model: Model,
        camera_buffers: &[Buffer],
        swapchain_props: SwapchainProperties,
        environment: &Environment,
        msaa_samples: vk::SampleCountFlags,
        render_pass: &RenderPass,
    ) -> Self {
        let dummy_texture = Texture::from_rgba(&context, 1, 1, &[0, 0, 0, 0]);

        // UBOS
        let transform_ubos = create_transform_ubos(&context, &model, swapchain_props.image_count);
        let (skin_ubos, skin_matrices) =
            create_skin_ubos(&context, &model, swapchain_props.image_count);
        let light_buffers = create_lights_ubos(&context, &model, swapchain_props.image_count);

        let descriptors = create_descriptors(
            &context,
            DescriptorsResources {
                camera_buffers,
                model_transform_buffers: &transform_ubos,
                model_skin_buffers: &skin_ubos,
                light_buffers: &light_buffers,
                dummy_texture: &dummy_texture,
                environment,
                model: &model,
            },
        );
        let pipeline_layout = create_pipeline_layout(context.device(), &descriptors);
        let opaque_pipeline = create_opaque_pipeline(
            &context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            pipeline_layout,
            &model,
        );
        let transparent_pipeline = create_transparent_pipeline(
            &context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            pipeline_layout,
            opaque_pipeline,
            &model,
        );

        Self {
            context,
            model,
            _dummy_texture: dummy_texture,
            transform_ubos,
            skin_ubos,
            skin_matrices,
            light_buffers,
            descriptors,
            pipeline_layout,
            opaque_pipeline,
            transparent_pipeline,
        }
    }

    pub fn rebuild_pipelines(
        &mut self,
        swapchain_props: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: &RenderPass,
    ) {
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.opaque_pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline, None);
        }

        self.opaque_pipeline = create_opaque_pipeline(
            &self.context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            &self.model,
        );
        self.transparent_pipeline = create_transparent_pipeline(
            &self.context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            self.opaque_pipeline,
            &self.model,
        );
    }
}

impl ModelRenderer {
    pub fn update_model(&mut self, delta_s: f32) {
        self.model.update(delta_s);
    }

    pub fn update_buffers(&mut self, frame_index: usize) {
        // Update transform buffers
        {
            let mesh_nodes = self
                .model
                .nodes()
                .nodes()
                .iter()
                .filter(|n| n.mesh_index().is_some());

            let transforms = mesh_nodes.map(|n| n.transform()).collect::<Vec<_>>();

            let elem_size = &self.context.get_ubo_alignment::<Matrix4<f32>>();
            let buffer = &mut self.transform_ubos[frame_index];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy_aligned(data_ptr, u64::from(*elem_size), &transforms);
            }
        }

        // Update skin buffers
        {
            let skins = self.model.skins();
            let skin_matrices = &mut self.skin_matrices[frame_index];

            for (index, skin) in skins.iter().enumerate() {
                let matrices = &mut skin_matrices[index];
                for (index, joint) in skin.joints().iter().take(MAX_JOINTS_PER_MESH).enumerate() {
                    let joint_matrix = joint.matrix();
                    matrices[index] = joint_matrix;
                }
            }

            let elem_size = &self.context.get_ubo_alignment::<JointsBuffer>();
            let buffer = &mut self.skin_ubos[frame_index];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy_aligned(data_ptr, u64::from(*elem_size), &skin_matrices);
            }
        }

        // Update light buffers
        {
            let model = &self.model;

            let uniforms = model
                .nodes()
                .nodes()
                .iter()
                .filter(|n| n.light_index().is_some())
                .map(|n| (n.transform(), n.light_index().unwrap()))
                .map(|(t, i)| (t, model.lights()[i]).into())
                .collect::<Vec<LightUniform>>();

            if !uniforms.is_empty() {
                let buffer = &mut self.light_buffers[frame_index];
                let data_ptr = buffer.map_memory();
                unsafe { mem_copy(data_ptr, &uniforms) };
            }
        }
    }

    pub fn cmd_draw(&self, command_buffer: vk::CommandBuffer, frame_index: usize) {
        let device = self.context.device();
        // Bind opaque pipeline
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.opaque_pipeline,
            )
        };

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

        // Draw opaque primitives
        register_model_draw_commands(
            &self.context,
            self.pipeline_layout,
            command_buffer,
            &self.model,
            &self.descriptors.dynamic_data_sets[frame_index..=frame_index],
            &self.descriptors.per_primitive_sets,
            |p| !p.material().is_transparent(),
        );

        // Bind transparent pipeline
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.transparent_pipeline,
            )
        };
        // Draw transparent primitives
        register_model_draw_commands(
            &self.context,
            self.pipeline_layout,
            command_buffer,
            &self.model,
            &self.descriptors.dynamic_data_sets[frame_index..=frame_index],
            &self.descriptors.per_primitive_sets,
            |p| p.material().is_transparent(),
        );
    }
}

impl Drop for ModelRenderer {
    fn drop(&mut self) {
        self.context.graphics_queue_wait_idle();
        let device = self.context.device();
        unsafe {
            device.destroy_pipeline(self.opaque_pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

fn create_transform_ubos(context: &Arc<Context>, model: &Model, count: u32) -> Vec<Buffer> {
    let mesh_node_count = model
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.mesh_index().is_some())
        .count() as u32;
    let elem_size = context.get_ubo_alignment::<Matrix4<f32>>();

    (0..count)
        .map(|_| {
            let mut buffer = Buffer::create(
                Arc::clone(context),
                u64::from(elem_size * mesh_node_count),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffer.map_memory();
            buffer
        })
        .collect::<Vec<_>>()
}

fn create_skin_ubos(
    context: &Arc<Context>,
    model: &Model,
    count: u32,
) -> (Vec<Buffer>, Vec<Vec<JointsBuffer>>) {
    let skin_node_count = model.skins().len().max(1);
    let elem_size = context.get_ubo_alignment::<JointsBuffer>();

    let buffers = (0..count)
        .map(|_| {
            let mut buffer = Buffer::create(
                Arc::clone(context),
                u64::from(elem_size * skin_node_count as u32),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffer.map_memory();
            buffer
        })
        .collect();

    let matrices = (0..count)
        .map(|_| {
            let mut matrices = Vec::with_capacity(skin_node_count);
            for _ in 0..skin_node_count {
                matrices.push([Matrix4::<f32>::identity(); MAX_JOINTS_PER_MESH]);
            }
            matrices
        })
        .collect();

    (buffers, matrices)
}

fn create_lights_ubos(context: &Arc<Context>, model: &Model, count: u32) -> Vec<Buffer> {
    let light_count = model
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.light_index().is_some())
        .count();

    // Buffer size cannot be 0 so we allocate at least anough space for one light
    // Probably a bad idea but I'd rather avoid creating a specific shader
    let buffer_size = std::cmp::max(1, light_count) * size_of::<LightUniform>();

    (0..count)
        .map(|_| {
            Buffer::create(
                Arc::clone(context),
                buffer_size as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .collect::<Vec<_>>()
}

#[derive(Copy, Clone)]
struct DescriptorsResources<'a> {
    camera_buffers: &'a [Buffer],
    model_transform_buffers: &'a [Buffer],
    model_skin_buffers: &'a [Buffer],
    light_buffers: &'a [Buffer],
    dummy_texture: &'a Texture,
    environment: &'a Environment,
    model: &'a Model,
}

pub struct Descriptors {
    context: Arc<Context>,
    pool: vk::DescriptorPool,
    dynamic_data_layout: vk::DescriptorSetLayout,
    dynamic_data_sets: Vec<vk::DescriptorSet>,
    static_data_layout: vk::DescriptorSetLayout,
    static_data_set: vk::DescriptorSet,
    per_primitive_layout: vk::DescriptorSetLayout,
    per_primitive_sets: Vec<vk::DescriptorSet>,
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.dynamic_data_layout, None);
            device.destroy_descriptor_set_layout(self.static_data_layout, None);
            device.destroy_descriptor_set_layout(self.per_primitive_layout, None);
        }
    }
}

fn create_descriptors(context: &Arc<Context>, resources: DescriptorsResources) -> Descriptors {
    let pool = create_descriptor_pool(context.device(), resources);

    let dynamic_data_layout = create_dynamic_data_descriptor_set_layout(context.device());
    let dynamic_data_sets =
        create_dynamic_data_descriptor_sets(context, pool, dynamic_data_layout, resources);

    let static_data_layout = create_static_data_descriptor_set_layout(context.device());
    let static_data_set =
        create_static_data_descriptor_sets(context, pool, static_data_layout, resources);

    let per_primitive_layout = create_per_primitive_descriptor_set_layout(context.device());
    let per_primitive_sets =
        create_per_primitive_descriptor_sets(context, pool, per_primitive_layout, resources);

    Descriptors {
        context: Arc::clone(context),
        pool,
        dynamic_data_layout,
        dynamic_data_sets,
        static_data_layout,
        static_data_set,
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
    let textures_desc_count = primitive_count * 5;

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: descriptor_count * 2,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: descriptor_count * 2,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: textures_desc_count + 3,
        },
    ];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(descriptor_count + 1 + primitive_count);

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
            .binding(LIGHT_UBO_BINDING)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
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
        let light_buffer = &resources.light_buffers[i];
        let model_transform_ubo = &resources.model_transform_buffers[i];
        let model_skin_ubo = &resources.model_skin_buffers[i];

        let camera_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(camera_ubo.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()];

        let light_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(light_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()];

        let model_transform_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(model_transform_ubo.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
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
                .dst_binding(LIGHT_UBO_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&light_buffer_info)
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

fn create_static_data_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(IRRADIANCE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(PRE_FILTERED_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(BRDF_SAMPLER_BINDING)
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

fn create_static_data_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    resources: DescriptorsResources,
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

    let irradiance_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(resources.environment.irradiance().view)
        .sampler(resources.environment.irradiance().sampler.unwrap())
        .build()];

    let pre_filtered_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(resources.environment.pre_filtered().view)
        .sampler(resources.environment.pre_filtered().sampler.unwrap())
        .build()];

    let brdf_lookup_info = [vk::DescriptorImageInfo::builder()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(resources.environment.brdf_lookup().view)
        .sampler(resources.environment.brdf_lookup().sampler.unwrap())
        .build()];

    let descriptor_writes = [
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(IRRADIANCE_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&irradiance_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(PRE_FILTERED_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&pre_filtered_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(BRDF_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&brdf_lookup_info)
            .build(),
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
        vk::DescriptorSetLayoutBinding::builder()
            .binding(COLOR_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(METALLIC_ROUGHNESS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(OCCLUSION_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(EMISSIVE_SAMPLER_BINDING)
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

            let albedo = material
                .get_color_texture_index()
                .map_or(resources.dummy_texture, |i| &textures[i]);
            let normals = material
                .get_normals_texture_index()
                .map_or(resources.dummy_texture, |i| &textures[i]);
            let metallic_roughness = material
                .get_metallic_roughness_texture_index()
                .map_or(resources.dummy_texture, |i| &textures[i]);
            let occlusion = material
                .get_occlusion_texture_index()
                .map_or(resources.dummy_texture, |i| &textures[i]);
            let emissive = material
                .get_emissive_texture_index()
                .map_or(resources.dummy_texture, |i| &textures[i]);

            let albedo_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(albedo.view)
                .sampler(albedo.sampler.unwrap())
                .build()];
            let normals_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(normals.view)
                .sampler(normals.sampler.unwrap())
                .build()];
            let metallic_roughness_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(metallic_roughness.view)
                .sampler(metallic_roughness.sampler.unwrap())
                .build()];
            let occlusion_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(occlusion.view)
                .sampler(occlusion.sampler.unwrap())
                .build()];
            let emissive_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(emissive.view)
                .sampler(emissive.sampler.unwrap())
                .build()];

            let set = sets[primitive_index];
            primitive_index += 1;

            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(COLOR_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&albedo_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(NORMALS_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&normals_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(METALLIC_ROUGHNESS_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&metallic_roughness_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(OCCLUSION_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&occlusion_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(EMISSIVE_SAMPLER_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&emissive_info)
                    .build(),
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

fn create_pipeline_layout(device: &Device, descriptors: &Descriptors) -> vk::PipelineLayout {
    let layouts = [
        descriptors.dynamic_data_layout,
        descriptors.static_data_layout,
        descriptors.per_primitive_layout,
    ];
    let push_constant_range = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        offset: 0,
        size: size_of::<Material>() as _,
    }];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_constant_range);

    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_opaque_pipeline(
    context: &Arc<Context>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    model: &Model,
) -> vk::Pipeline {
    let (specialization_info, _map_entries, _data) = create_model_frag_shader_specialization(model);

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

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    create_renderer_pipeline::<ModelVertex>(
        context,
        RendererPipelineParameters {
            shader_name: "model",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            swapchain_properties,
            msaa_samples,
            render_pass,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachment: &color_blend_attachment,
            parent: None,
        },
    )
}

fn create_transparent_pipeline(
    context: &Arc<Context>,
    swapchain_properties: SwapchainProperties,
    msaa_samples: vk::SampleCountFlags,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    parent: vk::Pipeline,
    model: &Model,
) -> vk::Pipeline {
    let (specialization_info, _map_entries, _data) = create_model_frag_shader_specialization(model);

    let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
        .front(Default::default())
        .back(Default::default());

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    create_renderer_pipeline::<ModelVertex>(
        context,
        RendererPipelineParameters {
            shader_name: "model",
            vertex_shader_specialization: None,
            fragment_shader_specialization: Some(&specialization_info),
            swapchain_properties,
            msaa_samples,
            render_pass,
            layout,
            depth_stencil_info: &depth_stencil_info,
            color_blend_attachment: &color_blend_attachment,
            parent: Some(parent),
        },
    )
}

fn create_model_frag_shader_specialization(
    model: &Model,
) -> (
    vk::SpecializationInfo,
    Vec<vk::SpecializationMapEntry>,
    Vec<u8>,
) {
    let map_entries = vec![vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: size_of::<u32>(),
    }];

    let light_count = model
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.light_index().is_some())
        .count() as u32;

    let data = Vec::from(unsafe { any_as_u8_slice(&[light_count]) });

    let specialization_info = vk::SpecializationInfo::builder()
        .map_entries(&map_entries)
        .data(&data)
        .build();

    (specialization_info, map_entries, data)
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

            // Bind descriptor sets
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

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct LightUniform {
    position: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    intensity: f32,
    range: f32,
    angle_scale: f32,
    angle_offset: f32,
    light_type: u32,
    pad: [u32; 3],
}

impl From<(Matrix4<f32>, Light)> for LightUniform {
    fn from((transform, light): (Matrix4<f32>, Light)) -> Self {
        let position = [transform.w.x, transform.w.y, transform.w.z, 0.0];

        let direction = (transform * Vector4::from(DEFAULT_LIGHT_DIRECTION))
            .normalize()
            .into();

        let color = light.color();
        let color = [color[0], color[1], color[2], 0.0];

        let intensity = light.intensity();

        let range = light.range().unwrap_or(-1.0);

        let (angle_scale, angle_offset) = match light.light_type() {
            Type::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => {
                let outer_cos = outer_cone_angle.cos();
                let angle_scale = 1.0 / math::max(0.001, inner_cone_angle.cos() - outer_cos);
                let angle_offset = -outer_cos * angle_scale;
                (angle_scale, angle_offset)
            }
            _ => (-1.0, -1.0),
        };

        let light_type = match light.light_type() {
            Type::Directional => DIRECTIONAL_LIGHT_TYPE,
            Type::Point => POINT_LIGHT_TYPE,
            Type::Spot { .. } => SPOT_LIGHT_TYPE,
        };

        Self {
            position,
            direction,
            color,
            intensity,
            range,
            angle_scale,
            angle_offset,
            light_type,
            pad: [0, 0, 0],
        }
    }
}
