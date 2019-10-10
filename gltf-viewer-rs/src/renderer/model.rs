use super::{create_renderer_pipeline, RendererPipelineParameters};
use ash::{version::DeviceV1_0, vk, Device};
use environment::*;
use math::cgmath::{Matrix4, SquareMatrix};
use model::*;
use std::{mem::size_of, sync::Arc};
use util::*;
use vulkan::*;

type JointsBuffer = [Matrix4<f32>; MAX_JOINTS_PER_MESH];

pub struct ModelRenderer {
    context: Arc<Context>,
    model: Model,
    _dummy_texture: Texture,
    descriptors: Descriptors,
    transform_ubos: Vec<Buffer>,
    skin_ubos: Vec<Buffer>,
    skin_matrices: Vec<Vec<JointsBuffer>>,
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
        let transform_ubos = create_transform_ubos(&context, &model, swapchain_props.image_count);
        let (skin_ubos, skin_matrices) =
            create_skin_ubos(&context, &model, swapchain_props.image_count);
        let descriptors = create_descriptors(
            &context,
            DescriptorsResources {
                camera_buffers,
                model_transform_buffers: &transform_ubos,
                model_skin_buffers: &skin_ubos,
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
        );
        let transparent_pipeline = create_transparent_pipeline(
            &context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            pipeline_layout,
            opaque_pipeline,
        );

        Self {
            context,
            model,
            _dummy_texture: dummy_texture,
            descriptors,
            transform_ubos,
            skin_ubos,
            skin_matrices,
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
        );
        self.transparent_pipeline = create_transparent_pipeline(
            &self.context,
            swapchain_props,
            msaa_samples,
            render_pass.get_render_pass(),
            self.pipeline_layout,
            self.opaque_pipeline,
        );
    }
}

impl ModelRenderer {
    pub fn update_model(&mut self, delta_s: f32) {
        self.model.update(delta_s);
    }

    pub fn update_buffers(&mut self, frame_index: usize) {
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
                1,
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

#[derive(Copy, Clone)]
struct DescriptorsResources<'a> {
    camera_buffers: &'a [Buffer],
    model_transform_buffers: &'a [Buffer],
    model_skin_buffers: &'a [Buffer],
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

impl Descriptors {
    pub fn new(
        context: Arc<Context>,
        pool: vk::DescriptorPool,
        dynamic_data_layout: vk::DescriptorSetLayout,
        dynamic_data_sets: Vec<vk::DescriptorSet>,
        static_data_layout: vk::DescriptorSetLayout,
        static_data_set: vk::DescriptorSet,
        per_primitive_layout: vk::DescriptorSetLayout,
        per_primitive_sets: Vec<vk::DescriptorSet>,
    ) -> Self {
        Self {
            context,
            pool,
            dynamic_data_layout,
            dynamic_data_sets,
            static_data_layout,
            static_data_set,
            per_primitive_layout,
            per_primitive_sets,
        }
    }
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

fn create_descriptors(context: &Arc<Context>, resources: DescriptorsResources) -> Descriptors {
    let pool = create_descriptor_pool(context.device(), resources);

    let dynamic_data_layout = create_dynamic_data_descriptor_set_layout(context.device());
    let dynamic_data_sets =
        create_dynamic_data_descriptor_sets(context, pool, dynamic_data_layout, resources);

    let static_data_layout = create_static_data_descriptor_set_layout(context.device());
    let static_data_sets =
        create_static_data_descriptor_sets(context, pool, static_data_layout, resources);

    let per_primitive_layout = create_per_primitive_descriptor_set_layout(context.device());
    let per_primitive_sets =
        create_per_primitive_descriptor_sets(context, pool, per_primitive_layout, resources);

    Descriptors::new(
        Arc::clone(context),
        pool,
        dynamic_data_layout,
        dynamic_data_sets,
        static_data_layout,
        static_data_sets,
        per_primitive_layout,
        per_primitive_sets,
    )
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
            descriptor_count,
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
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&camera_buffer_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&model_transform_buffer_info)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(2)
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

fn create_dynamic_data_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
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

fn create_static_data_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
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
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&irradiance_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&pre_filtered_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(5)
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
            .binding(6)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(8)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(9)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(10)
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
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&albedo_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&normals_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(8)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&metallic_roughness_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(9)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&occlusion_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(10)
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
) -> vk::Pipeline {
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
                0,
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
                    2,
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