$env:RUST_LOG="INFO"

cargo run --release -- -c config.yml -f assets/models/cesium_man_with_light.glb

$env:RUST_LOG=""