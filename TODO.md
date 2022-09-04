- bug: Texture color issues
    - https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/TextureEncodingTest
    - https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/TextureLinearInterpolationTest
    - With these two example the output is as expected (all spheres have the same color) but for some reason the text panels are displayed
    as yellow/green instead of white/black. Is seems to be a CPU side issue when loading the texture because the debug color output which
    displays the sampled texture without modification shows the same issue
- improvement: Rework pbr to include ior factor
- improvement: Keep adding extension support
- improvement: anisotropic filtering
- improvement: shadows mapping
