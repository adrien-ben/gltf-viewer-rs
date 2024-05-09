#version 450

layout(location = 0) in vec3 oPositions;

layout(binding = 1) uniform samplerCube cubemapSampler;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(cubemapSampler, oPositions).rgb;

    outColor = vec4(color, 1.0);
}
