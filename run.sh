export VK_LAYER_PATH=$VULKAN_SDK/Bin
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

RUST_LOG=debug cargo run --release -- --config config.yml --debug

export VK_LAYER_PATH=
export VK_INSTANCE_LAYERS=