pub mod gbufferpass;
pub mod lightpass;

mod uniform;

use gbufferpass::GBufferPass;
use lightpass::LightPass;
use math::cgmath::Matrix4;
use model::MAX_JOINTS_PER_MESH;
use model::Material;
use model::Model;
use std::cell::RefCell;
use std::rc::Weak;
use std::sync::Arc;
use uniform::*;
use vulkan::ash::vk;
use vulkan::{Buffer, Context, mem_copy, mem_copy_aligned};

type JointsBuffer = [Matrix4<f32>; MAX_JOINTS_PER_MESH];

pub struct ModelData {
    context: Arc<Context>,
    model: Weak<RefCell<Model>>,
    transform_ubos: Vec<Buffer>,
    skin_ubos: Vec<Buffer>,
    skin_matrices: Vec<Vec<JointsBuffer>>,
    materials_ubo: Buffer,
    light_ubos: Vec<Buffer>,
}

pub struct ModelRenderer {
    pub data: ModelData,
    pub gbuffer_pass: GBufferPass,
    pub light_pass: LightPass,
}

impl ModelData {
    pub fn create(context: Arc<Context>, model: Weak<RefCell<Model>>, image_count: u32) -> Self {
        let model_rc = model
            .upgrade()
            .expect("Cannot create model renderer because model was dropped");

        let transform_ubos = create_transform_ubos(&context, &model_rc.borrow(), image_count);
        let (skin_ubos, skin_matrices) =
            create_skin_ubos(&context, &model_rc.borrow(), image_count);
        let materials_ubo = create_materials_ubo(&context, &model_rc.borrow());
        let light_ubos = create_lights_ubos(&context, image_count);

        Self {
            context,
            model,
            transform_ubos,
            skin_ubos,
            skin_matrices,
            materials_ubo,
            light_ubos,
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

            let elem_size = self.context.get_ubo_alignment::<JointsBuffer>();
            let buffer = &mut self.skin_ubos[frame_index];
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy_aligned(data_ptr, u64::from(elem_size), skin_matrices);
            }
        }

        // Update light buffers
        {
            let mut lights_ubo = LightsUBO::default();

            for (i, ln) in model
                .nodes()
                .nodes()
                .iter()
                .filter(|n| n.light_index().is_some())
                .map(|n| (n.transform(), n.light_index().unwrap()))
                .map(|(t, i)| (t, model.lights()[i]).into())
                .enumerate()
                .take(MAX_LIGHT_COUNT)
            {
                lights_ubo.count += 1;
                lights_ubo.lights[i] = ln;
            }

            let buffer = &mut self.light_ubos[frame_index];
            let data_ptr = buffer.map_memory();
            unsafe { mem_copy(data_ptr, &[lights_ubo]) };
        }

        // Update materials buffer
        {
            let mut ubos: Vec<MaterialUniform> = vec![Material::default().into()];
            model
                .materials()
                .iter()
                .copied()
                .map(|m| m.into())
                .for_each(|m| ubos.push(m));

            let elem_size = self.context.get_ubo_alignment::<MaterialUniform>() as vk::DeviceSize;
            let buffer = &mut self.materials_ubo;
            unsafe {
                let data_ptr = buffer.map_memory();
                mem_copy_aligned(data_ptr, elem_size, &ubos);
            }
        }
    }
}
