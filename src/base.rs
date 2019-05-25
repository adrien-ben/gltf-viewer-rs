use crate::{
    camera::*, config::*, controls::*, environment::*, math, model::*, pipelines::*, util::*,
    vulkan::*,
};
use ash::{version::DeviceV1_0, vk, Device};
use cgmath::{Deg, Matrix4, Point3, Vector3};
use std::{
    mem::{align_of, size_of},
    path::Path,
    rc::Rc,
};
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub struct BaseApp {
    events_loop: EventsLoop,
    _window: Window,
    config: Config,
    resize_dimensions: Option<[u32; 2]>,

    camera: Camera,
    input_state: InputState,
    context: Rc<Context>,
    _environment: Environment,
    render_pass: vk::RenderPass,
    model_descriptors: Descriptors,
    skybox_descriptors: Descriptors,
    uniform_buffers: Vec<Buffer>,
    skybox_model: SkyboxModel,
    model: Model,
    _dummy_texture: Texture,
    pipelines: Pipelines,
    msaa_samples: vk::SampleCountFlags,
    color_texture: Texture,
    depth_format: vk::Format,
    depth_texture: Texture,
    swapchain: Swapchain,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,
}

impl BaseApp {
    pub fn new<P: AsRef<Path>>(config: Config, path: P) -> Self {
        log::debug!("Creating application.");

        let resolution = [config.resolution().width(), config.resolution().height()];

        let events_loop = EventsLoop::new();
        let window = WindowBuilder::new()
            .with_title("GLTF Viewer")
            .with_dimensions(LogicalSize::new(
                f64::from(resolution[0]),
                f64::from(resolution[1]),
            ))
            .build(&events_loop)
            .unwrap();

        let context = Rc::new(Context::new(&window));

        let environment = Environment::new(&context, config.env());

        let swapchain_support_details = SwapchainSupportDetails::new(
            context.physical_device(),
            context.surface(),
            context.surface_khr(),
        );
        let swapchain_properties =
            swapchain_support_details.get_ideal_swapchain_properties(resolution, config.vsync());

        let msaa_samples = context.get_max_usable_sample_count(config.msaa());
        log::debug!("msaa: {} - preferred was {}", msaa_samples, config.msaa());

        let depth_format = Self::find_depth_format(&context);

        let render_pass = Self::create_render_pass(
            context.device(),
            swapchain_properties,
            msaa_samples,
            depth_format,
        );

        let uniform_buffers =
            Self::create_uniform_buffers(&context, swapchain_properties.image_count);

        let skybox_model = SkyboxModel::new(&context);

        let model = Model::create_from_file(&context, path).unwrap();

        let dummy_texture = Texture::from_rgba(&context, 1, 1, &vec![0, 0, 0, 0]);

        let model_descriptors = Self::create_model_descriptors(
            &context,
            &uniform_buffers,
            model.textures(),
            &dummy_texture,
            &environment,
        );

        let skybox_descriptors =
            Self::create_skybox_descriptors(&context, &uniform_buffers, &environment);

        let pipelines = Pipelines::build(
            Rc::clone(&context),
            swapchain_properties,
            msaa_samples,
            render_pass,
            &skybox_descriptors,
            &model_descriptors,
        );

        let color_texture =
            Self::create_color_texture(&context, swapchain_properties, msaa_samples);

        let depth_texture = Self::create_depth_texture(
            &context,
            depth_format,
            swapchain_properties.extent,
            msaa_samples,
        );

        let swapchain = Swapchain::create(
            Rc::clone(&context),
            swapchain_support_details,
            resolution,
            config.vsync(),
            &color_texture,
            &depth_texture,
            render_pass,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            &context,
            &swapchain,
            render_pass,
            &pipelines,
            &skybox_descriptors.sets(),
            &skybox_model,
            &model_descriptors.sets(),
            &model,
        );

        let in_flight_frames = Self::create_sync_objects(context.device());

        Self {
            events_loop,
            _window: window,
            config,
            resize_dimensions: None,
            camera: Default::default(),
            input_state: Default::default(),
            context,
            _environment: environment,
            render_pass,
            model_descriptors,
            skybox_descriptors,
            uniform_buffers,
            skybox_model,
            model,
            _dummy_texture: dummy_texture,
            pipelines,
            msaa_samples,
            color_texture,
            depth_format,
            depth_texture,
            swapchain,
            command_buffers,
            in_flight_frames,
        }
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let depth_attachement_desc = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();
        let resolve_attachment_desc = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();
        let attachment_descs = [
            color_attachment_desc,
            depth_attachement_desc,
            resolve_attachment_desc,
        ];

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let resolve_attachment_refs = [resolve_attachment_ref];

        let subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .resolve_attachments(&resolve_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .build();
        let subpass_descs = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();
        let subpass_deps = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps)
            .build();

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    fn create_uniform_buffers(context: &Rc<Context>, count: u32) -> Vec<Buffer> {
        (0..count)
            .map(|_| {
                Buffer::create(
                    Rc::clone(context),
                    size_of::<CameraUBO>() as _,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_model_descriptors(
        context: &Rc<Context>,
        uniform_buffers: &[Buffer],
        textures: &[Texture],
        dummy_texture: &Texture,
        environment: &Environment,
    ) -> Descriptors {
        let layout = Self::create_model_descriptor_set_layout(context.device());
        let pool = Self::create_model_descriptor_pool(context.device(), uniform_buffers.len() as _);
        let sets = Self::create_model_descriptor_sets(
            context,
            pool,
            layout,
            uniform_buffers,
            textures,
            dummy_texture,
            environment,
        );
        Descriptors::new(Rc::clone(context), layout, pool, sets)
    }

    fn create_model_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_TEXTURE_COUNT)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
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
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    fn create_model_descriptor_pool(device: &Device, descriptor_count: u32) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: descriptor_count * (MAX_TEXTURE_COUNT + 3),
            },
        ];

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(descriptor_count)
            .build();

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    }

    fn create_model_descriptor_sets(
        context: &Rc<Context>,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        buffers: &[Buffer],
        textures: &[Texture],
        dummy_texture: &Texture,
        environment: &Environment,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..buffers.len()).map(|_| layout).collect::<Vec<_>>();

        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .build();
        let sets = unsafe {
            context
                .device()
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
        };

        sets.iter().zip(buffers.iter()).for_each(|(set, buffer)| {
            let buffer_info = [vk::DescriptorBufferInfo::builder()
                .buffer(buffer.buffer)
                .offset(0)
                .range(size_of::<CameraUBO>() as _)
                .build()];

            let image_info = {
                let mut infos = textures
                    .iter()
                    .take(MAX_TEXTURE_COUNT as _)
                    .map(|texture| {
                        vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(texture.view)
                            .sampler(texture.sampler.unwrap())
                            .build()
                    })
                    .collect::<Vec<_>>();

                while infos.len() < MAX_TEXTURE_COUNT as _ {
                    infos.push(
                        vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(dummy_texture.view)
                            .sampler(dummy_texture.sampler.unwrap())
                            .build(),
                    )
                }

                infos
            };

            let irradiance_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(environment.irradiance().view)
                .sampler(environment.irradiance().sampler.unwrap())
                .build()];

            let pre_filtered_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(environment.pre_filtered().view)
                .sampler(environment.pre_filtered().sampler.unwrap())
                .build()];

            let brdf_lookup_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(environment.brdf_lookup().view)
                .sampler(environment.brdf_lookup().sampler.unwrap())
                .build()];

            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&irradiance_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&pre_filtered_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&brdf_lookup_info)
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

    fn create_skybox_descriptors(
        context: &Rc<Context>,
        uniform_buffers: &[Buffer],
        environment: &Environment,
    ) -> Descriptors {
        let layout = Self::create_skybox_descriptor_set_layout(context.device());
        let pool =
            Self::create_skybox_descriptor_pool(context.device(), uniform_buffers.len() as _);
        let sets = Self::create_skybox_descriptor_sets(
            context,
            pool,
            layout,
            uniform_buffers,
            environment.skybox(),
        );
        Descriptors::new(Rc::clone(context), layout, pool, sets)
    }

    fn create_skybox_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    fn create_skybox_descriptor_pool(device: &Device, descriptor_count: u32) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: descriptor_count,
            },
        ];

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(descriptor_count)
            .build();

        unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
    }

    fn create_skybox_descriptor_sets(
        context: &Rc<Context>,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        buffers: &[Buffer],
        cubemap: &Texture,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..buffers.len()).map(|_| layout).collect::<Vec<_>>();

        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .build();
        let sets = unsafe {
            context
                .device()
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
        };

        sets.iter().zip(buffers.iter()).for_each(|(set, buffer)| {
            let buffer_info = [vk::DescriptorBufferInfo::builder()
                .buffer(buffer.buffer)
                .offset(0)
                .range(size_of::<CameraUBO>() as _)
                .build()];

            let cubemap_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(cubemap.view)
                .sampler(cubemap.sampler.unwrap())
                .build()];

            let descriptor_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&cubemap_info)
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

    fn create_color_texture(
        context: &Rc<Context>,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let format = swapchain_properties.format.format;
        let image = Image::create(
            Rc::clone(context),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            swapchain_properties.extent,
            1,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        );

        image.transition_image_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::COLOR);

        Texture::new(Rc::clone(context), image, view, None)
    }

    /// Create the depth buffer texture (image, memory and view).
    ///
    /// This function also transitions the image to be ready to be used
    /// as a depth/stencil attachement.
    fn create_depth_texture(
        context: &Rc<Context>,
        format: vk::Format,
        extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let image = Image::create(
            Rc::clone(context),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            1,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        );

        image.transition_image_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let view = image.create_view(vk::ImageViewType::TYPE_2D, vk::ImageAspectFlags::DEPTH);

        Texture::new(Rc::clone(context), image, view, None)
    }

    fn find_depth_format(context: &Context) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        context
            .find_supported_format(
                &candidates,
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            )
            .expect("Failed to find a supported depth format")
    }

    fn create_and_register_command_buffers(
        context: &Context,
        swapchain: &Swapchain,
        render_pass: vk::RenderPass,
        pipelines: &Pipelines,
        skybox_descriptor_sets: &[vk::DescriptorSet],
        skybox_model: &SkyboxModel,
        descriptor_sets: &[vk::DescriptorSet],
        model: &Model,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(context.general_command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain.image_count() as _)
            .build();

        let device = context.device();
        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = swapchain.framebuffers()[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    .build();
                unsafe {
                    device
                        .begin_command_buffer(buffer, &command_buffer_begin_info)
                        .unwrap()
                };
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.7, 0.7, 0.7, 1.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];
                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain.properties().extent,
                    })
                    .clear_values(&clear_values)
                    .build();

                unsafe {
                    device.cmd_begin_render_pass(
                        buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    )
                };
            }

            // Bind skybox pipeline
            unsafe {
                device.cmd_bind_pipeline(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.skybox_pipeline(),
                )
            };

            // Bind skybox descriptor sets
            unsafe {
                device.cmd_bind_descriptor_sets(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.skybox_layout(),
                    0,
                    &skybox_descriptor_sets[i..=i],
                    &[],
                )
            };

            unsafe {
                device.cmd_bind_vertex_buffers(buffer, 0, &[skybox_model.vertices().buffer], &[0]);
            }

            unsafe {
                device.cmd_bind_index_buffer(
                    buffer,
                    skybox_model.indices().buffer,
                    0,
                    vk::IndexType::UINT32,
                );
            }

            // Draw skybox
            unsafe { device.cmd_draw_indexed(buffer, 36, 1, 0, 0, 0) };

            // Bind opaque pipeline
            unsafe {
                device.cmd_bind_pipeline(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.opaque_pipeline(),
                )
            };

            // Bind descriptor sets
            unsafe {
                device.cmd_bind_descriptor_sets(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.model_layout(),
                    0,
                    &descriptor_sets[i..=i],
                    &[],
                )
            };

            // Draw opaque primitives
            Self::register_model_draw_commands(
                device,
                pipelines.model_layout(),
                buffer,
                model,
                |p| !p.material().is_transparent(),
            );

            // Bind transparent pipeline
            unsafe {
                device.cmd_bind_pipeline(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipelines.transparent_pipeline(),
                )
            };

            // Draw transparent primitives
            Self::register_model_draw_commands(
                device,
                pipelines.model_layout(),
                buffer,
                model,
                |p| p.material().is_transparent(),
            );

            // End render pass
            unsafe { device.cmd_end_render_pass(buffer) };

            // End command buffer
            unsafe { device.end_command_buffer(buffer).unwrap() };
        });

        buffers
    }

    fn register_model_draw_commands<F>(
        device: &Device,
        pipeline_layout: vk::PipelineLayout,
        command_buffer: vk::CommandBuffer,
        model: &Model,
        primitive_filter: F,
    ) where
        F: FnMut(&&Primitive) -> bool + Copy,
    {
        for node in model.nodes() {
            let mesh = model.mesh(node.mesh_index());

            // Push transform constants
            unsafe {
                let transform = node.transform();
                let transform_contants = any_as_u8_slice(&transform);
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &transform_contants,
                );
            };

            for primitive in mesh.primitives().iter().filter(primitive_filter) {
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
                    let material = primitive.material();
                    let material_contants = any_as_u8_slice(&material);
                    device.cmd_push_constants(
                        command_buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        size_of::<Matrix4<f32>>() as _,
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

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info = vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build();
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects)
        }

        InFlightFrames::new(sync_objects_vec)
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");
        loop {
            if self.process_event() {
                break;
            }
            self.camera.update(&self.input_state);
            self.draw_frame();
        }
        unsafe { self.context.device().device_wait_idle().unwrap() };
    }

    /// Process the events from the `EventsLoop` and return whether the
    /// main loop should stop.
    fn process_event(&mut self) -> bool {
        let mut should_stop = false;
        let mut resize_dimensions = None;
        let mut input_state = self.input_state;
        input_state.reset();

        self.events_loop.poll_events(|event| {
            input_state = input_state.update(&event);
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => should_stop = true,
                    WindowEvent::Resized(LogicalSize { width, height }) => {
                        resize_dimensions = Some([width as u32, height as u32]);
                    }
                    _ => {}
                }
            }
        });

        self.resize_dimensions = resize_dimensions;
        self.input_state = input_state;
        should_stop
    }

    fn draw_frame(&mut self) {
        log::trace!("Drawing frame.");
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.context
                .device()
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap()
        };

        let result = self
            .swapchain
            .acquire_next_image(None, Some(image_available_semaphore), None);
        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.context.device().reset_fences(&wait_fences).unwrap() };

        self.update_uniform_buffers(image_index);

        let device = self.context.device();
        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                device
                    .queue_submit(
                        self.context.graphics_queue(),
                        &submit_infos,
                        in_flight_fence,
                    )
                    .unwrap()
            };
        }

        let swapchains = [self.swapchain.swapchain_khr()];
        let images_indices = [image_index];

        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&images_indices)
                .build();
            let result = self.swapchain.present(present_info);
            match result {
                Ok(is_suboptimal) if is_suboptimal => {
                    self.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain();
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }

            if self.resize_dimensions.is_some() {
                self.recreate_swapchain();
            }
        }
    }

    /// Recreates the swapchain.
    ///
    /// If the window has been resized, then the new size is used
    /// otherwise, the size of the current swapchain is used.
    ///
    /// If the window has been minimized, then the functions block until
    /// the window is maximized. This is because a width or height of 0
    /// is not legal.
    fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain.");

        if self.has_window_been_minimized() {
            while !self.has_window_been_maximized() {
                self.process_event();
            }
        }

        unsafe { self.context.device().device_wait_idle().unwrap() };

        self.cleanup_swapchain();

        let device = self.context.device();

        let dimensions = self.resize_dimensions.unwrap_or([
            self.swapchain.properties().extent.width,
            self.swapchain.properties().extent.height,
        ]);

        let swapchain_support_details = SwapchainSupportDetails::new(
            self.context.physical_device(),
            self.context.surface(),
            self.context.surface_khr(),
        );
        let swapchain_properties = swapchain_support_details
            .get_ideal_swapchain_properties(dimensions, self.config.vsync());

        let render_pass = Self::create_render_pass(
            device,
            swapchain_properties,
            self.msaa_samples,
            self.depth_format,
        );

        let pipelines = Pipelines::build(
            Rc::clone(&self.context),
            swapchain_properties,
            self.msaa_samples,
            render_pass,
            &self.skybox_descriptors,
            &self.model_descriptors,
        );

        let color_texture =
            Self::create_color_texture(&self.context, swapchain_properties, self.msaa_samples);

        let depth_texture = Self::create_depth_texture(
            &self.context,
            self.depth_format,
            swapchain_properties.extent,
            self.msaa_samples,
        );

        let swapchain = Swapchain::create(
            Rc::clone(&self.context),
            swapchain_support_details,
            dimensions,
            self.config.vsync(),
            &color_texture,
            &depth_texture,
            render_pass,
        );

        let command_buffers = Self::create_and_register_command_buffers(
            &self.context,
            &swapchain,
            render_pass,
            &pipelines,
            &self.skybox_descriptors.sets(),
            &self.skybox_model,
            &self.model_descriptors.sets(),
            &self.model,
        );

        self.swapchain = swapchain;
        self.render_pass = render_pass;
        self.pipelines = pipelines;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.command_buffers = command_buffers;
    }

    fn has_window_been_minimized(&self) -> bool {
        match self.resize_dimensions {
            Some([x, y]) if x == 0 || y == 0 => true,
            _ => false,
        }
    }

    fn has_window_been_maximized(&self) -> bool {
        match self.resize_dimensions {
            Some([x, y]) if x > 0 && y > 0 => true,
            _ => false,
        }
    }

    /// Clean up the swapchain and all resources that depends on it.
    fn cleanup_swapchain(&mut self) {
        let device = self.context.device();
        unsafe {
            device.free_command_buffers(self.context.general_command_pool(), &self.command_buffers);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.swapchain.destroy();
    }

    fn update_uniform_buffers(&mut self, current_image: u32) {
        let aspect = self.swapchain.properties().extent.width as f32
            / self.swapchain.properties().extent.height as f32;

        let view = Matrix4::look_at(
            self.camera.position(),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let proj = math::perspective(Deg(45.0), aspect, 0.01, 10.0);

        let ubos = [CameraUBO::new(view, proj, self.camera.position())];
        let buffer_mem = self.uniform_buffers[current_image as usize].memory;
        let size = size_of::<CameraUBO>() as vk::DeviceSize;
        unsafe {
            let device = self.context.device();
            let data_ptr = device
                .map_memory(buffer_mem, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<f32>() as _, size);
            align.copy_from_slice(&ubos);
            device.unmap_memory(buffer_mem);
        }
    }
}

impl Drop for BaseApp {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        self.cleanup_swapchain();
        let device = self.context.device();
        self.in_flight_frames.destroy(device);
    }
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(&device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}
