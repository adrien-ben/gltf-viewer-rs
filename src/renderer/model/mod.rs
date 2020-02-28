pub mod gbufferpass;
pub mod lightpass;

mod uniform;

use gbufferpass::GBufferPass;
use lightpass::LightPass;
use math::cgmath::Matrix4;
use model::Model;
use model::MAX_JOINTS_PER_MESH;
use std::cell::RefCell;
use std::rc::Weak;
use std::sync::Arc;
use uniform::*;
use vulkan::{mem_copy, mem_copy_aligned, Buffer, Context, SwapchainProperties};

type JointsBuffer = [Matrix4<f32>; MAX_JOINTS_PER_MESH];

pub struct ModelData {
    context: Arc<Context>,
    model: Weak<RefCell<Model>>,
    transform_ubos: Vec<Buffer>,
    skin_ubos: Vec<Buffer>,
    skin_matrices: Vec<Vec<JointsBuffer>>,
    light_buffers: Vec<Buffer>,
}

impl ModelData {
    pub fn create(
        context: Arc<Context>,

        model: Weak<RefCell<Model>>,
        swapchain_props: SwapchainProperties,
    ) -> Self {
        let model_rc = model
            .upgrade()
            .expect("Cannot create model renderer because model was dropped");

        let transform_ubos =
            create_transform_ubos(&context, &model_rc.borrow(), swapchain_props.image_count);
        let (skin_ubos, skin_matrices) =
            create_skin_ubos(&context, &model_rc.borrow(), swapchain_props.image_count);
        let light_buffers =
            create_lights_ubos(&context, &model_rc.borrow(), swapchain_props.image_count);

        Self {
            context,
            model,
            transform_ubos,
            skin_ubos,
            skin_matrices,
            light_buffers,
        }
    }

    pub fn update_buffers(&mut self, frame_index: usize) {
        let model = &self
            .model
            .upgrade()
            .expect("Cannot update buffers because model was dropped");
        let model = model.borrow();

        // Update transform buffers
        {
            let mesh_nodes = model
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
            let skins = model.skins();
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
}

pub struct ModelRenderer {
    pub data: ModelData,
    pub gbuffer_pass: GBufferPass,
    pub light_pass: LightPass,
}
