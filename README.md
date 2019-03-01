# gltf-viewer-rs

This project is a [glTF 2.0][0] viewer written in Rust. Rendering is done using the [Vulkan API][1]
using [Ash][2]. It runs on Window, Linux and MacOS.

## Build it

```sh
cargo build
```

By default, building the project will trigger shader compilation for all shaders in `./assets/shaders`.
You can either skip this step altogether by setting the environnement variable `SKIP_SHADER_COMPILATION`
to `true`, or you can change the default by setting `SHADERS_DIR`. Compiled shaders will be generated at
the same location as the shader source, with the same name suffixed by `.spv`.

> To compile the shaders you'll need to have `glslangValidator` on your PATH.

Building the project with the debug profile will activate Vulkan validation layers. Activated layers are:

- VK_LAYER_LUNARG_standard_validation

## Run it

```sh
RUST_LOG=gltf_viewer_rs=warn cargo run -- C:\\dev\\glTF-Sample-Models\\2.0\\Triangle\\glTF\\Triangle.gltf
```

[0]: https://github.com/KhronosGroup/glTF
[1]: https://www.khronos.org/vulkan/
[2]: https://github.com/MaikKlein/ash