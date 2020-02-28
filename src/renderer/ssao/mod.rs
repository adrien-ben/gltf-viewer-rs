mod renderpass;

use super::fullscreen::{QuadModel, QuadVertex};
use super::{create_renderer_pipeline, RendererPipelineParameters};
pub use renderpass::RenderPass as SSAORenderPass;
use std::sync::Arc;
use vulkan::ash::{version::DeviceV1_0, vk, Device};
use vulkan::{Context, Descriptors, SwapchainProperties, Texture};

const NORMALS_SAMPLER_BINDING: u32 = 0;
const DEPTH_SAMPLER_BINDING: u32 = 1;

pub struct SSAOPass {
    context: Arc<Context>,
    descriptors: Descriptors,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl SSAOPass {
    pub fn create(
        context: Arc<Context>,
        swapchain_props: SwapchainProperties,
        render_pass: &SSAORenderPass,
        normals: &Texture,
        depth: &Texture,
    ) -> Self {
        let descriptors = create_descriptors(&context, normals, depth);
        let pipeline_layout = create_pipeline_layout(context.device(), descriptors.layout());
        let pipeline = create_pipeline(
            &context,
            swapchain_props,
            render_pass.get_render_pass(),
            pipeline_layout,
        );

        SSAOPass {
            context,
            descriptors,
            pipeline_layout,
            pipeline,
        }
    }
}

impl SSAOPass {
    pub fn set_inputs(&mut self, normals: &Texture, depth: &Texture) {
        unsafe {
            self.context
                .device()
                .free_descriptor_sets(self.descriptors.pool(), self.descriptors.sets());
        }
        self.descriptors.set_sets(create_descriptor_sets(
            &self.context,
            self.descriptors.pool(),
            self.descriptors.layout(),
            normals,
            depth,
        ));
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
        )
    }

    pub fn cmd_draw(&self, command_buffer: vk::CommandBuffer, quad_model: &QuadModel) {
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

        // Bind descriptor sets
        unsafe {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &self.descriptors.sets(),
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

fn create_descriptors(context: &Arc<Context>, normals: &Texture, depth: &Texture) -> Descriptors {
    let layout = create_descriptor_set_layout(context.device());
    let pool = create_descriptor_pool(context.device());
    let sets = create_descriptor_sets(context, pool, layout, normals, depth);
    Descriptors::new(Arc::clone(context), layout, pool, sets)
}

fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
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
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}
fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 2,
    }];

    let create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(1)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    unsafe { device.create_descriptor_pool(&create_info, None).unwrap() }
}

fn create_descriptor_sets(
    context: &Arc<Context>,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    normals: &Texture,
    depth: &Texture,
) -> Vec<vk::DescriptorSet> {
    let layouts = [layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe {
        context
            .device()
            .allocate_descriptor_sets(&allocate_info)
            .unwrap()
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

    let descriptor_writes = [
        vk::WriteDescriptorSet::builder()
            .dst_set(sets[0])
            .dst_binding(NORMALS_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&normals_info)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(sets[0])
            .dst_binding(DEPTH_SAMPLER_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info)
            .build(),
    ];

    unsafe {
        context
            .device()
            .update_descriptor_sets(&descriptor_writes, &[])
    }

    sets
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
    unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
}

fn create_pipeline(
    context: &Arc<Context>,
    swapchain_properties: SwapchainProperties,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> vk::Pipeline {
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
            vertex_shader_name: "fullscreen",
            fragment_shader_name: "ssao",
            vertex_shader_specialization: None,
            fragment_shader_specialization: None,
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
