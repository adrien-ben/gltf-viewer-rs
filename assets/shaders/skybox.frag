#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 oPositions;

layout(binding = 0) uniform CameraUBO {
     mat4 view;
     mat4 proj;
     vec3 eye;
} cameraUBO;

layout(binding = 1) uniform samplerCube cubemapSampler;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(cubemapSampler, oPositions).rgb;

    color = color/(color + 1.0);
    color = pow(color, vec3(1.0/2.2));
    outColor = vec4(color, 1.0);
}
