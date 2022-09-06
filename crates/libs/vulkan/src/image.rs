use super::{buffer::*, context::*, swapchain::SwapchainProperties};
use ash::{vk, Device};
use std::sync::Arc;

#[derive(Copy, Clone)]
pub struct ImageParameters {
    pub mem_properties: vk::MemoryPropertyFlags,
    pub extent: vk::Extent2D,
    pub layers: u32,
    pub mip_levels: u32,
    pub sample_count: vk::SampleCountFlags,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub create_flags: vk::ImageCreateFlags,
}

impl Default for ImageParameters {
    fn default() -> Self {
        Self {
            mem_properties: vk::MemoryPropertyFlags::empty(),
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            layers: 1,
            mip_levels: 1,
            sample_count: vk::SampleCountFlags::TYPE_1,
            format: vk::Format::R8G8B8A8_UNORM,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::SAMPLED,
            create_flags: vk::ImageCreateFlags::empty(),
        }
    }
}

pub struct Image {
    context: Arc<Context>,
    pub image: vk::Image,
    memory: Option<vk::DeviceMemory>,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub layers: u32,
    managed: bool,
}

impl Image {
    fn new(
        context: Arc<Context>,
        image: vk::Image,
        memory: Option<vk::DeviceMemory>,
        extent: vk::Extent3D,
        format: vk::Format,
        mip_levels: u32,
        layers: u32,
        managed: bool,
    ) -> Self {
        Self {
            context,
            image,
            memory,
            extent,
            format,
            mip_levels,
            layers,
            managed,
        }
    }

    pub fn create(context: Arc<Context>, parameters: ImageParameters) -> Self {
        let extent = vk::Extent3D {
            width: parameters.extent.width,
            height: parameters.extent.height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(parameters.mip_levels)
            .array_layers(parameters.layers)
            .format(parameters.format)
            .tiling(parameters.tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(parameters.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(parameters.sample_count)
            .flags(parameters.create_flags);

        let device = context.device();
        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create image")
        };
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index = find_memory_type(
            mem_requirements,
            context.get_mem_properties(),
            parameters.mem_properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        let memory = unsafe {
            let mem = device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate image memory");
            device
                .bind_image_memory(image, mem, 0)
                .expect("Failed to bind image memory");
            mem
        };

        Image::new(
            context,
            image,
            Some(memory),
            extent,
            parameters.format,
            parameters.mip_levels,
            parameters.layers,
            false,
        )
    }

    pub fn create_swapchain_image(
        context: Arc<Context>,
        image: vk::Image,
        swapchain_properties: SwapchainProperties,
    ) -> Self {
        Self::new(
            context,
            image,
            None,
            vk::Extent3D {
                width: swapchain_properties.extent.width,
                height: swapchain_properties.extent.height,
                depth: 1,
            },
            swapchain_properties.format.format,
            1,
            1,
            true,
        )
    }
}

impl Image {
    pub fn create_view(
        &self,
        view_type: vk::ImageViewType,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        create_image_view(
            self.context.device(),
            self.image,
            view_type,
            self.layers,
            self.mip_levels,
            0,
            self.format,
            aspect_mask,
        )
    }

    pub fn create_mips_views(
        &self,
        view_type: vk::ImageViewType,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Vec<vk::ImageView> {
        (0..self.mip_levels)
            .map(|mip| {
                create_image_view(
                    self.context.device(),
                    self.image,
                    view_type,
                    self.layers,
                    1,
                    mip,
                    self.format,
                    aspect_mask,
                )
            })
            .collect()
    }

    pub fn transition_image_layout(
        &self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        self.context.execute_one_time_commands(|buffer| {
            self.cmd_transition_image_layout(buffer, old_layout, new_layout)
        });
    }

    pub fn cmd_transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        self.cmd_transition_image_mips_layout(
            command_buffer,
            0,
            self.mip_levels,
            old_layout,
            new_layout,
        )
    }

    pub fn cmd_transition_image_mips_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        base_mip_level: u32,
        level_count: u32,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let barrier = self.get_barrier(base_mip_level, level_count, old_layout, new_layout);

        let dependency_info =
            vk::DependencyInfo::builder().image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            self.context
                .synchronization2()
                .cmd_pipeline_barrier2(command_buffer, &dependency_info)
        };
    }

    fn get_barrier(
        &self,
        base_mip_level: u32,
        level_count: u32,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2 {
        let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
            match (old_layout, new_layout) {
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                    vk::AccessFlags2::NONE,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::NONE,
                    vk::PipelineStageFlags2::TRANSFER,
                ),
                (
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                ),
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => {
                    (
                        vk::AccessFlags2::NONE,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags2::NONE,
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                    )
                }
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                    vk::AccessFlags2::NONE,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                ),
                (
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                ),
                (
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ) => (
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags2::TRANSFER,
                ),
                (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                ),
                (
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                ),
                (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                    vk::AccessFlags2::NONE,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::NONE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                ),
                (
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ) => (
                    vk::AccessFlags2::SHADER_READ,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                ),
                (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::PipelineStageFlags2::TRANSFER,
                ),
                (
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ) => (
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::AccessFlags2::SHADER_READ,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                ),
                _ => {
                    log::warn!("Undefined layout transition {old_layout:?} -> {new_layout:?}");

                    (
                        vk::AccessFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::NONE,
                        vk::PipelineStageFlags2::NONE,
                    )
                }
            };

        let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            || old_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        {
            let mut mask = vk::ImageAspectFlags::DEPTH;
            if has_stencil_component(self.format) {
                mask |= vk::ImageAspectFlags::STENCIL;
            }
            mask
        } else {
            vk::ImageAspectFlags::COLOR
        };

        vk::ImageMemoryBarrier2::builder()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access_mask)
            .old_layout(old_layout)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access_mask)
            .new_layout(new_layout)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level,
                level_count,
                base_array_layer: 0,
                layer_count: self.layers,
            })
            .build()
    }

    pub fn copy_buffer(&self, buffer: &Buffer, extent: vk::Extent2D) {
        self.context.execute_one_time_commands(|command_buffer| {
            self.cmd_copy_buffer(command_buffer, buffer, extent)
        })
    }

    pub fn cmd_copy_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: &Buffer,
        extent: vk::Extent2D,
    ) {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: self.layers,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .build();
        let regions = [region];
        unsafe {
            self.context.device().cmd_copy_buffer_to_image(
                command_buffer,
                buffer.buffer,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            )
        }
    }

    /// Record command to copy [src_image] into this image.
    ///
    /// The full extent of the passed in layer will be copied, so the target image
    /// should be big enough to contain the content of the source image.
    ///
    /// Source image layout should be TRANSFER_SRC_OPTIMAL and target TRANSFER_DST_OPTIMAL.
    pub fn cmd_copy(
        &self,
        command_buffer: vk::CommandBuffer,
        src_image: &Image,
        subresource_layers: vk::ImageSubresourceLayers,
    ) {
        let image_copy_info = [vk::ImageCopy::builder()
            .src_subresource(subresource_layers)
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_subresource(subresource_layers)
            .dst_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .extent(src_image.extent)
            .build()];

        unsafe {
            self.context.device().cmd_copy_image(
                command_buffer,
                src_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &image_copy_info,
            );
        };
    }

    pub fn generate_mipmaps(&self, extent: vk::Extent2D) {
        let format_properties = unsafe {
            self.context
                .instance()
                .get_physical_device_format_properties(self.context.physical_device(), self.format)
        };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!(
                "Linear blitting is not supported for format {:?}.",
                self.format
            )
        }

        self.context.execute_one_time_commands(|buffer| {
            self.cmd_generate_mipmaps(buffer, extent);
        });
    }

    pub fn cmd_generate_mipmaps(&self, command_buffer: vk::CommandBuffer, extent: vk::Extent2D) {
        let format_properties = unsafe {
            self.context
                .instance()
                .get_physical_device_format_properties(self.context.physical_device(), self.format)
        };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!(
                "Linear blitting is not supported for format {:?}.",
                self.format
            )
        }

        let mut mip_width = extent.width as i32;
        let mut mip_height = extent.height as i32;
        for level in 1..self.mip_levels {
            let next_mip_width = if mip_width > 1 {
                mip_width / 2
            } else {
                mip_width
            };
            let next_mip_height = if mip_height > 1 {
                mip_height / 2
            } else {
                mip_height
            };

            self.cmd_transition_image_mips_layout(
                command_buffer,
                level - 1,
                1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            let blit = vk::ImageBlit::builder()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level - 1,
                    base_array_layer: 0,
                    layer_count: self.layers,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_mip_width,
                        y: next_mip_height,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: self.layers,
                })
                .build();
            let blits = [blit];

            unsafe {
                self.context.device().cmd_blit_image(
                    command_buffer,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                )
            };

            self.cmd_transition_image_mips_layout(
                command_buffer,
                level - 1,
                1,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );

            mip_width = next_mip_width;
            mip_height = next_mip_height;
        }

        self.cmd_transition_image_mips_layout(
            command_buffer,
            self.mip_levels - 1,
            1,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
}

// Getters
impl Image {
    pub fn get_mip_levels(&self) -> u32 {
        self.mip_levels
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            if !self.managed {
                self.context.device().destroy_image(self.image, None);
            }
            if let Some(memory) = self.memory {
                self.context.device().free_memory(memory, None);
            }
        }
    }
}

fn has_stencil_component(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
}

pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    view_type: vk::ImageViewType,
    layers: u32,
    mip_levels: u32,
    base_mip_level: u32,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags,
) -> vk::ImageView {
    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: layers,
        });

    unsafe {
        device
            .create_image_view(&create_info, None)
            .expect("Failed to create image view")
    }
}

pub struct LayoutTransition<'a> {
    pub image: &'a Image,
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub mips_range: MipsRange,
}

#[derive(Clone, Copy)]
pub enum MipsRange {
    All,
    Index(u32),
    Range { first: u32, count: u32 },
}

impl MipsRange {
    fn first(&self) -> u32 {
        match self {
            Self::All => 0,
            Self::Index(index) => *index,
            Self::Range { first, .. } => *first,
        }
    }

    fn count(&self) -> Option<u32> {
        match self {
            Self::All => None,
            Self::Index(_) => Some(1),
            Self::Range { count, .. } => Some(*count),
        }
    }
}

pub fn cmd_transition_images_layouts(
    command_buffer: vk::CommandBuffer,
    transitions: &[LayoutTransition],
) {
    if transitions.is_empty() {
        return;
    }

    let context = &transitions[0].image.context;

    let barriers = transitions
        .iter()
        .map(|t| {
            let base_mip_level = t.mips_range.first();
            let level_count = t.mips_range.count().unwrap_or(t.image.mip_levels);

            t.image
                .get_barrier(base_mip_level, level_count, t.old_layout, t.new_layout)
        })
        .collect::<Vec<_>>();

    let dependency_info = vk::DependencyInfo::builder().image_memory_barriers(&barriers);

    unsafe {
        context
            .synchronization2()
            .cmd_pipeline_barrier2(command_buffer, &dependency_info)
    };
}
