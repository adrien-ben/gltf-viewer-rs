export VK_LAYER_PATH=$VULKAN_SDK/Bin
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation

RUST_LOG=warn cargo run --release -- -c config.yml -f assets/models/cesium_man_with_light.glb -d

export VK_LAYER_PATH=
export VK_INSTANCE_LAYERS=