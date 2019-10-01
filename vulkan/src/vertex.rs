use ash::vk::{VertexInputAttributeDescription, VertexInputBindingDescription};

pub trait Vertex {
    fn get_bindings_descriptions() -> Vec<VertexInputBindingDescription>;
    fn get_attributes_descriptions() -> Vec<VertexInputAttributeDescription>;
}
