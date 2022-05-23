$env:VK_LAYER_PATH = "$env:VULKAN_SDK\Bin"
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:RUST_LOG="DEBUG"
$env:RUST_BACKTRACE=1

cargo run --release -- -c config.yml -f assets/models/cesium_man_with_light.glb -d

$env:VK_LAYER_PATH = ""
$env:VK_INSTANCE_LAYERS = ""
$env:RUST_LOG=""
$env:RUST_BACKTRACE=""